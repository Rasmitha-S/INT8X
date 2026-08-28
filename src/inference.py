"""
Inference engine for TFLite models (FP32 and INT8).
Performs dynamic quantization parameter retrieval, input quantization, interpreter execution,
output dequantization, and latency measurement.
"""
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf


def load_tflite_interpreter(model_path: str) -> tf.lite.Interpreter:
    """
    Loads and initializes a TFLite Interpreter with allocated tensors.

    Args:
        model_path: Path to the .tflite model file.

    Returns:
        Allocated tf.lite.Interpreter instance.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def get_quantization_details(interpreter: tf.lite.Interpreter) -> Dict[str, Any]:
    """
    Dynamically retrieves input and output tensor details and quantization parameters.

    Args:
        interpreter: Allocated tf.lite.Interpreter instance.

    Returns:
        Dictionary containing input/output indices, shapes, dtypes, scales, and zero_points.
    """
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details.get("quantization", (0.0, 0))
    output_scale, output_zero_point = output_details.get("quantization", (0.0, 0))

    return {
        "input_index": input_details["index"],
        "input_shape": tuple(input_details["shape"]),
        "input_dtype": input_details["dtype"],
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zero_point),
        "output_index": output_details["index"],
        "output_shape": tuple(output_details["shape"]),
        "output_dtype": output_details["dtype"],
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zero_point),
        "is_quantized": input_details["dtype"] in (np.int8, np.uint8),
    }


def quantize_input_image(
    image: np.ndarray, scale: float, zero_point: int, dtype: np.dtype = np.int8
) -> np.ndarray:
    """
    Quantizes a normalized float32 image in [0.0, 1.0] to integer representation.
    Formula: q = clip(round(image / scale + zero_point), qmin, qmax)

    Args:
        image: Float32 image array.
        scale: Quantization scale factor.
        zero_point: Quantization zero-point offset.
        dtype: Target integer dtype (default np.int8).

    Returns:
        Quantized integer array with matching dtype.
    """
    if scale <= 0.0:
        raise ValueError(f"Quantization scale must be positive, got {scale}")

    quantized = np.round(image / scale + zero_point)
    if dtype == np.int8:
        return np.clip(quantized, -128, 127).astype(np.int8)
    elif dtype == np.uint8:
        return np.clip(quantized, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported quantized dtype: {dtype}")


def dequantize_output_tensor(
    output_quantized: np.ndarray, scale: float, zero_point: int
) -> np.ndarray:
    """
    Dequantizes an integer output tensor back to floating point representation.
    Formula: f = (q - zero_point) * scale

    Args:
        output_quantized: Raw integer output from TFLite interpreter.
        scale: Quantization scale factor.
        zero_point: Quantization zero-point offset.

    Returns:
        Dequantized float32 numpy array.
    """
    if scale <= 0.0:
        return output_quantized.astype(np.float32)
    return (output_quantized.astype(np.float32) - zero_point) * scale


def run_inference(
    interpreter: tf.lite.Interpreter,
    image: np.ndarray,
) -> Dict[str, Any]:
    """
    Executes single-sample inference on an input image using the provided TFLite interpreter.
    Dynamically handles float32 and INT8 quantized models.

    Args:
        interpreter: Allocated tf.lite.Interpreter instance.
        image: 2D (28, 28), 3D (28, 28, 1), or 4D (1, 28, 28, 1) float32 image in [0.0, 1.0].

    Returns:
        Dictionary with predicted_digit, confidence, probabilities, latency_ms, and tensor info.
    """
    q_details = get_quantization_details(interpreter)
    input_idx = q_details["input_index"]
    output_idx = q_details["output_index"]

    # Normalize input shape to (1, 28, 28, 1)
    img_arr = np.array(image, dtype=np.float32)
    if img_arr.ndim == 2:
        img_arr = img_arr[np.newaxis, :, :, np.newaxis]
    elif img_arr.ndim == 3:
        if img_arr.shape[-1] == 1:
            img_arr = img_arr[np.newaxis, :, :, :]
        else:
            img_arr = img_arr[np.newaxis, :, :, np.newaxis]
    elif img_arr.ndim == 4:
        pass
    else:
        raise ValueError(f"Invalid input image shape: {img_arr.shape}")

    # Prepare input tensor based on model datatype
    if q_details["is_quantized"]:
        input_tensor = quantize_input_image(
            img_arr,
            scale=q_details["input_scale"],
            zero_point=q_details["input_zero_point"],
            dtype=q_details["input_dtype"],
        )
    else:
        input_tensor = img_arr.astype(np.float32)

    # Timed inference execution
    t_start = time.perf_counter()
    interpreter.set_tensor(input_idx, input_tensor)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_idx)
    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000.0

    # Dequantize output if quantized
    if q_details["is_quantized"]:
        dequant_output = dequantize_output_tensor(
            raw_output,
            scale=q_details["output_scale"],
            zero_point=q_details["output_zero_point"],
        )
    else:
        dequant_output = raw_output.astype(np.float32)

    probs = dequant_output.flatten()
    prob_sum = float(np.sum(probs))
    if prob_sum > 0.0:
        normalized_probs = (probs / prob_sum).tolist()
    else:
        normalized_probs = probs.tolist()

    predicted_digit = int(np.argmax(probs))
    confidence = float(np.max(normalized_probs))

    return {
        "predicted_digit": predicted_digit,
        "confidence": round(confidence, 4),
        "probabilities": [round(float(p), 4) for p in normalized_probs],
        "latency_ms": round(latency_ms, 4),
        "input_dtype": str(q_details["input_dtype"].__name__ if hasattr(q_details["input_dtype"], "__name__") else q_details["input_dtype"]),
        "output_dtype": str(q_details["output_dtype"].__name__ if hasattr(q_details["output_dtype"], "__name__") else q_details["output_dtype"]),
        "is_quantized": q_details["is_quantized"],
        "input_scale": q_details["input_scale"],
        "input_zero_point": q_details["input_zero_point"],
        "output_scale": q_details["output_scale"],
        "output_zero_point": q_details["output_zero_point"],
    }


def compare_sample_predictions(
    fp32_model_path: str,
    int8_model_path: str,
    samples: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """
    Executes a limited consistency check between FP32 and INT8 models on a subset of samples.

    Args:
        fp32_model_path: Path to FP32 TFLite model.
        int8_model_path: Path to INT8 TFLite model.
        samples: Array of test images (N, 28, 28, 1).
        labels: Array of ground-truth labels (N,).

    Returns:
        Dictionary with sample count, agreement count, agreement percentage, and sample comparison list.
    """
    fp32_interp = load_tflite_interpreter(fp32_model_path)
    int8_interp = load_tflite_interpreter(int8_model_path)

    total_samples = len(samples)
    matches = 0
    fp32_correct = 0
    int8_correct = 0
    details = []

    for i in range(total_samples):
        sample = samples[i:i+1]
        label = int(labels[i])

        fp32_res = run_inference(fp32_interp, sample)
        int8_res = run_inference(int8_interp, sample)

        fp32_pred = fp32_res["predicted_digit"]
        int8_pred = int8_res["predicted_digit"]

        is_match = (fp32_pred == int8_pred)
        if is_match:
            matches += 1
        if fp32_pred == label:
            fp32_correct += 1
        if int8_pred == label:
            int8_correct += 1

        details.append({
            "index": i,
            "ground_truth": label,
            "fp32_prediction": fp32_pred,
            "int8_prediction": int8_pred,
            "agreement": is_match,
            "fp32_confidence": fp32_res["confidence"],
            "int8_confidence": int8_res["confidence"],
        })

    agreement_rate = float(matches / total_samples) if total_samples > 0 else 0.0

    return {
        "samples_evaluated": total_samples,
        "matching_predictions": matches,
        "agreement_rate": round(agreement_rate, 4),
        "fp32_sample_accuracy": round(float(fp32_correct / total_samples), 4),
        "int8_sample_accuracy": round(float(int8_correct / total_samples), 4),
        "sample_details": details,
    }
