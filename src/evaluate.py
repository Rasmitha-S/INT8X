"""
Evaluation and benchmarking module for TFLite and Keras models.
Performs verifiable accuracy and latency evaluations using standardized warm-up and timed runs.
Provides full quantitative comparison between FP32 and INT8 models.
"""
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from src.data import load_mnist
from src.metrics import get_file_size_bytes, get_file_size_kb, load_metrics_json, update_metrics


def evaluate_tflite_model(
    tflite_model_path: str,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_latency_runs: int = 100,
    num_warmup_runs: int = 20,
) -> Dict[str, Any]:
    """
    Evaluates a TFLite model on the test dataset for accuracy and measures single-sample latency.

    Latency Methodology:
    1. Warm-up Phase: Runs `num_warmup_runs` inferences to prime CPU caches and TFLite kernels.
    2. Timed Phase: Measures `num_latency_runs` consecutive single-sample inferences using `time.perf_counter()`.
    3. Metrics: Computes mean, median, standard deviation, minimum, and maximum inference latency in milliseconds.

    Accuracy Methodology:
    Runs inference on all `len(x_test)` samples, compares argmax prediction to `y_test`, and computes exact accuracy.

    Args:
        tflite_model_path: Path to the .tflite model file.
        x_test: Test images of shape (N, 28, 28, 1), float32 in [0.0, 1.0].
        y_test: Test labels of shape (N,).
        num_latency_runs: Number of timed inference iterations (default 100).
        num_warmup_runs: Number of unmeasured warm-up iterations (default 20).

    Returns:
        Dictionary containing accuracy, latency statistics, and evaluation metadata.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_index = input_details["index"]
    output_index = output_details["index"]
    input_dtype = input_details["dtype"]
    output_dtype = output_details["dtype"]

    # Quantization parameters (scale and zero_point)
    input_scale, input_zero_point = input_details.get("quantization", (0.0, 0))
    output_scale, output_zero_point = output_details.get("quantization", (0.0, 0))

    # Helper function to preprocess single sample based on tensor dtype
    def prepare_input(img_float32: np.ndarray) -> np.ndarray:
        sample = np.expand_dims(img_float32, axis=0) if img_float32.ndim == 3 else img_float32
        if input_dtype in (np.int8, np.uint8):
            # Quantize float32 -> int8/uint8: q = (f / scale) + zero_point
            quantized = np.round(sample / input_scale + input_zero_point)
            if input_dtype == np.int8:
                return np.clip(quantized, -128, 127).astype(np.int8)
            else:
                return np.clip(quantized, 0, 255).astype(np.uint8)
        else:
            return sample.astype(np.float32)

    # --- Latency Measurement ---
    sample_input = prepare_input(x_test[0])

    # 1. Warm-up runs
    for _ in range(num_warmup_runs):
        interpreter.set_tensor(input_index, sample_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_index)

    # 2. Timed runs
    latencies_ms = []
    for _ in range(num_latency_runs):
        t_start = time.perf_counter()
        interpreter.set_tensor(input_index, sample_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_index)
        t_end = time.perf_counter()
        latencies_ms.append((t_end - t_start) * 1000.0)

    latency_arr = np.array(latencies_ms)
    latency_mean = float(np.mean(latency_arr))
    latency_median = float(np.median(latency_arr))
    latency_std = float(np.std(latency_arr))
    latency_min = float(np.min(latency_arr))
    latency_max = float(np.max(latency_arr))

    # --- Accuracy Evaluation ---
    num_test_samples = len(x_test)
    correct_count = 0

    for i in range(num_test_samples):
        inp = prepare_input(x_test[i])
        interpreter.set_tensor(input_index, inp)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_index)

        # If quantized output, dequantize: f = (q - zero_point) * scale
        if output_dtype in (np.int8, np.uint8) and output_scale > 0:
            probs = (output_data.astype(np.float32) - output_zero_point) * output_scale
        else:
            probs = output_data

        pred_label = int(np.argmax(probs))
        if pred_label == int(y_test[i]):
            correct_count += 1

    accuracy = float(correct_count / num_test_samples)

    return {
        "model_path": tflite_model_path,
        "input_dtype": str(input_dtype.__name__ if hasattr(input_dtype, "__name__") else input_dtype),
        "output_dtype": str(output_details["dtype"].__name__ if hasattr(output_details["dtype"], "__name__") else output_details["dtype"]),
        "accuracy": round(accuracy, 4),
        "correct_predictions": correct_count,
        "total_test_samples": num_test_samples,
        "latency_mean_ms": round(latency_mean, 4),
        "latency_median_ms": round(latency_median, 4),
        "latency_std_ms": round(latency_std, 4),
        "latency_min_ms": round(latency_min, 4),
        "latency_max_ms": round(latency_max, 4),
        "warmup_runs": num_warmup_runs,
        "latency_timed_runs": num_latency_runs,
        "measurement_method": "Single-sample interpreter.invoke() timed with time.perf_counter() after 20 warm-up runs.",
    }


def run_full_benchmark(
    fp32_model_path: str = "models/model_fp32.tflite",
    int8_model_path: str = "models/model_int8.tflite",
    metrics_path: str = "results/metrics.json",
    comparison_path: str = "results/comparison.json",
) -> Dict[str, Any]:
    """
    Executes the complete, rigorous benchmark comparing FP32 TFLite vs INT8 TFLite
    across all 10,000 MNIST test samples and records structured results.

    Args:
        fp32_model_path: Path to FP32 TFLite model.
        int8_model_path: Path to INT8 TFLite model.
        metrics_path: Path to metrics.json file.
        comparison_path: Path to comparison.json file.

    Returns:
        Structured comparison dictionary.
    """
    print("[1/4] Loading complete 10,000-sample MNIST test dataset...")
    (_, _), (x_test, y_test) = load_mnist()
    assert len(x_test) == 10000, f"Expected 10,000 test samples, got {len(x_test)}"

    # 1. Benchmark FP32 TFLite Model
    print(f"[2/4] Benchmarking FP32 TFLite model ({fp32_model_path}) on 10,000 samples...")
    fp32_eval = evaluate_tflite_model(
        tflite_model_path=fp32_model_path,
        x_test=x_test,
        y_test=y_test,
        num_latency_runs=100,
        num_warmup_runs=20,
    )
    fp32_size_bytes = get_file_size_bytes(fp32_model_path)
    fp32_size_kb = get_file_size_kb(fp32_model_path)

    # 2. Benchmark INT8 TFLite Model
    print(f"[3/4] Benchmarking INT8 TFLite model ({int8_model_path}) on 10,000 samples...")
    int8_eval = evaluate_tflite_model(
        tflite_model_path=int8_model_path,
        x_test=x_test,
        y_test=y_test,
        num_latency_runs=100,
        num_warmup_runs=20,
    )
    int8_size_bytes = get_file_size_bytes(int8_model_path)
    int8_size_kb = get_file_size_kb(int8_model_path)

    # 3. Calculate Comparison Metrics
    print("[4/4] Computing exact comparison and delta metrics...")
    size_reduction_pct = round(((fp32_size_bytes - int8_size_bytes) / fp32_size_bytes) * 100.0, 2)
    compression_ratio = round(fp32_size_bytes / int8_size_bytes, 2)

    acc_fp32 = fp32_eval["accuracy"]
    acc_int8 = int8_eval["accuracy"]
    acc_delta = round(acc_int8 - acc_fp32, 4)
    acc_delta_pts = round(acc_delta * 100.0, 2)
    acc_preservation = round((acc_int8 / acc_fp32) * 100.0, 2) if acc_fp32 > 0 else 100.0

    lat_fp32 = fp32_eval["latency_mean_ms"]
    lat_int8 = int8_eval["latency_mean_ms"]
    lat_diff_ms = round(lat_int8 - lat_fp32, 4)
    lat_change_pct = round(((lat_int8 - lat_fp32) / lat_fp32) * 100.0, 2) if lat_fp32 > 0 else 0.0

    benchmark_env = {
        "os": platform.system(),
        "os_release": platform.release(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "device_type": "Host Development Environment (CPU)",
    }

    comparison_data = {
        "benchmark_environment": benchmark_env,
        "dataset": "MNIST Test Set (10,000 samples)",
        "methodology": {
            "test_samples": 10000,
            "warmup_runs": 20,
            "latency_timed_runs": 100,
            "timer": "time.perf_counter()",
            "notes": "Single-sample inference measured on development host CPU. Does not represent specialized MCU hardware cycle counts.",
        },
        "fp32_baseline": {
            "model_format": "TFLite (FP32)",
            "accuracy": acc_fp32,
            "accuracy_percent": round(acc_fp32 * 100.0, 2),
            "correct_predictions": fp32_eval["correct_predictions"],
            "total_samples": fp32_eval["total_test_samples"],
            "size_bytes": fp32_size_bytes,
            "size_kb": fp32_size_kb,
            "latency_mean_ms": lat_fp32,
            "latency_median_ms": fp32_eval["latency_median_ms"],
            "latency_std_ms": fp32_eval["latency_std_ms"],
            "latency_min_ms": fp32_eval["latency_min_ms"],
            "latency_max_ms": fp32_eval["latency_max_ms"],
        },
        "int8_quantized": {
            "model_format": "TFLite (INT8 PTQ)",
            "accuracy": acc_int8,
            "accuracy_percent": round(acc_int8 * 100.0, 2),
            "correct_predictions": int8_eval["correct_predictions"],
            "total_samples": int8_eval["total_test_samples"],
            "size_bytes": int8_size_bytes,
            "size_kb": int8_size_kb,
            "latency_mean_ms": lat_int8,
            "latency_median_ms": int8_eval["latency_median_ms"],
            "latency_std_ms": int8_eval["latency_std_ms"],
            "latency_min_ms": int8_eval["latency_min_ms"],
            "latency_max_ms": int8_eval["latency_max_ms"],
        },
        "comparison_results": {
            "size_reduction_bytes": fp32_size_bytes - int8_size_bytes,
            "size_reduction_percent": size_reduction_pct,
            "compression_ratio": compression_ratio,
            "accuracy_delta": acc_delta,
            "accuracy_delta_percentage_points": acc_delta_pts,
            "accuracy_preservation_percent": acc_preservation,
            "latency_difference_ms": lat_diff_ms,
            "latency_change_percent": lat_change_pct,
        },
    }

    # Save to comparison.json
    comp_out_path = Path(comparison_path)
    comp_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(comp_out_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"Comparison data saved to {comparison_path}")

    # Update metrics.json with latest INT8 evaluation numbers
    int8_full_metrics = {
        "int8_tflite_accuracy": acc_int8,
        "int8_tflite_latency_ms": lat_int8,
        "int8_tflite_latency_median_ms": int8_eval["latency_median_ms"],
        "int8_tflite_latency_std_ms": int8_eval["latency_std_ms"],
        "int8_tflite_latency_min_ms": int8_eval["latency_min_ms"],
        "int8_tflite_latency_max_ms": int8_eval["latency_max_ms"],
        "int8_warmup_runs": int8_eval["warmup_runs"],
        "int8_timed_runs": int8_eval["latency_timed_runs"],
        "size_reduction_percent": size_reduction_pct,
        "compression_ratio": compression_ratio,
        "accuracy_delta_percentage_points": acc_delta_pts,
        "latency_change_percent": lat_change_pct,
    }
    update_metrics(int8_full_metrics, metrics_path)
    print(f"Metrics updated in {metrics_path}")

    return comparison_data


if __name__ == "__main__":
    run_full_benchmark()
