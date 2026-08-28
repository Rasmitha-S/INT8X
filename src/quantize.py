"""
Post-Training Quantization (PTQ) module for converting FP32 models to INT8 TFLite.
Applies full-integer quantization with calibration dataset and verifies tensor quantization parameters.
"""
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf

from src.data import get_representative_dataset_generator, load_mnist
from src.metrics import get_file_size_bytes, get_file_size_kb, update_metrics


def quantize_to_int8(
    keras_model_path: str = "models/model_fp32.keras",
    output_tflite_path: str = "models/model_int8.tflite",
    num_calibration_samples: int = 200,
    results_file: str = "results/metrics.json",
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Converts a trained FP32 Keras model to a fully quantized INT8 TFLite model using PTQ.

    Full-integer quantization parameters:
    - Optimization: Optimize.DEFAULT
    - Target ops: OpsSet.TFLITE_BUILTINS_INT8
    - Inference input dtype: tf.int8
    - Inference output dtype: tf.int8
    - Calibration: Real MNIST training samples via representative dataset generator

    Args:
        keras_model_path: Path to the trained .keras FP32 model.
        output_tflite_path: Destination path for the .tflite INT8 model.
        num_calibration_samples: Number of training samples for calibration (default 200).
        results_file: Path to metrics.json file.

    Returns:
        Tuple of (quantized model bytes, quantization metadata dictionary).
    """
    print(f"[1/4] Loading FP32 model from {keras_model_path}...")
    model = tf.keras.models.load_model(keras_model_path)

    print("[2/4] Preparing representative dataset generator from training data...")
    (x_train, _), (x_test, _) = load_mnist()
    rep_gen = get_representative_dataset_generator(x_train, num_samples=num_calibration_samples)

    print("[3/4] Converting model with full-integer INT8 Post-Training Quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_int8_bytes = converter.convert()

    # Save to disk
    out_path = Path(output_tflite_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tflite_int8_bytes)
    print(f"       Saved INT8 TFLite model to {output_tflite_path}")

    # 4. Strict tensor inspection and verification
    print("[4/4] Inspecting and verifying INT8 tensor quantization parameters...")
    interpreter = tf.lite.Interpreter(model_path=str(out_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_dtype = input_details["dtype"]
    output_dtype = output_details["dtype"]
    input_scale, input_zero_point = input_details.get("quantization", (0.0, 0))
    output_scale, output_zero_point = output_details.get("quantization", (0.0, 0))

    if input_dtype != np.int8:
        raise ValueError(f"Expected input dtype to be np.int8, got {input_dtype}")
    if output_dtype != np.int8:
        raise ValueError(f"Expected output dtype to be np.int8, got {output_dtype}")
    if input_scale <= 0.0:
        raise ValueError(f"Invalid input scale: {input_scale}")
    if output_scale <= 0.0:
        raise ValueError(f"Invalid output scale: {output_scale}")

    # Test single real sample inference
    sample_raw = x_test[0:1]  # shape (1, 28, 28, 1), float32
    sample_int8 = np.clip(
        np.round(sample_raw / input_scale + input_zero_point), -128, 127
    ).astype(np.int8)

    interpreter.set_tensor(input_details["index"], sample_int8)
    interpreter.invoke()
    output_int8 = interpreter.get_tensor(output_details["index"])

    if output_int8.shape != (1, 10) or output_int8.dtype != np.int8:
        raise ValueError(f"Unexpected output tensor shape/dtype: {output_int8.shape}, {output_int8.dtype}")

    int8_size_bytes = get_file_size_bytes(output_tflite_path)
    int8_size_kb = get_file_size_kb(output_tflite_path)

    int8_metadata = {
        "int8_tflite_size_bytes": int8_size_bytes,
        "int8_tflite_size_kb": int8_size_kb,
        "int8_input_dtype": "int8",
        "int8_output_dtype": "int8",
        "int8_input_scale": float(input_scale),
        "int8_input_zero_point": int(input_zero_point),
        "int8_output_scale": float(output_scale),
        "int8_output_zero_point": int(output_zero_point),
        "int8_quantization_verified": True,
        "int8_calibration_samples": num_calibration_samples,
    }

    update_metrics(int8_metadata, results_file)
    print(f"INT8 metadata successfully recorded in {results_file}")
    return tflite_int8_bytes, int8_metadata


if __name__ == "__main__":
    quantize_to_int8()
