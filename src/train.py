"""
Training routine for FP32 lightweight CNN on MNIST.
Trains the CNN, saves the Keras model and FP32 TFLite model, and evaluates baseline metrics.
"""
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf

from src.data import load_mnist
from src.evaluate import evaluate_tflite_model
from src.metrics import get_file_size_bytes, get_file_size_kb, update_metrics
from src.model import build_tiny_cnn, get_parameter_count


def train_and_export_fp32(
    epochs: int = 8,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    validation_split: float = 0.1,
    models_dir: str = "models",
    results_file: str = "results/metrics.json",
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Trains the Phase 3 lightweight CNN on MNIST and exports FP32 .keras and .tflite models.

    Args:
        epochs: Maximum training epochs (default 8).
        batch_size: Mini-batch size (default 64).
        learning_rate: Adam optimizer learning rate (default 0.001).
        validation_split: Fraction of training data for validation (default 0.1).
        models_dir: Directory where models are stored.
        results_file: Path to metrics.json file.

    Returns:
        Tuple of (trained tf.keras.Model, baseline metrics dictionary).
    """
    # 1. Load MNIST data
    print("[1/5] Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # 2. Build model architecture
    print("[2/5] Building lightweight CNN...")
    model = build_tiny_cnn(input_shape=(28, 28, 1), num_classes=10)
    param_count = get_parameter_count(model)
    print(f"       Total parameters: {param_count}")

    # 3. Compile and train model
    print(f"[3/5] Compiling and training model (max epochs={epochs}, batch_size={batch_size})...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=1,
    )

    # 4. Evaluate Keras model on untouched test set
    print("[4/5] Evaluating Keras model on MNIST test set...")
    keras_loss, keras_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"       Keras Test Loss: {keras_loss:.4f}, Test Accuracy: {keras_acc:.4%}")

    # Ensure output directories exist
    out_models_dir = Path(models_dir)
    out_models_dir.mkdir(parents=True, exist_ok=True)

    # Save .keras model
    keras_model_path = str(out_models_dir / "model_fp32.keras")
    model.save(keras_model_path)
    print(f"       Saved Keras model to {keras_model_path}")

    # Convert and save FP32 TFLite model
    tflite_model_path = str(out_models_dir / "model_fp32.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32_bytes = converter.convert()
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_fp32_bytes)
    print(f"       Saved FP32 TFLite model to {tflite_model_path}")

    # 5. Measure actual FP32 TFLite metrics (accuracy & latency)
    print("[5/5] Benchmarking FP32 TFLite model on test dataset...")
    tflite_eval = evaluate_tflite_model(
        tflite_model_path=tflite_model_path,
        x_test=x_test,
        y_test=y_test,
        num_latency_runs=100,
        num_warmup_runs=20,
    )

    keras_size_bytes = get_file_size_bytes(keras_model_path)
    tflite_size_bytes = get_file_size_bytes(tflite_model_path)

    baseline_metrics = {
        "fp32_parameter_count": param_count,
        "fp32_keras_accuracy": round(float(keras_acc), 4),
        "fp32_keras_loss": round(float(keras_loss), 4),
        "fp32_keras_size_bytes": keras_size_bytes,
        "fp32_keras_size_kb": get_file_size_kb(keras_model_path),
        "fp32_tflite_accuracy": tflite_eval["accuracy"],
        "fp32_tflite_size_bytes": tflite_size_bytes,
        "fp32_tflite_size_kb": get_file_size_kb(tflite_model_path),
        "fp32_tflite_latency_ms": tflite_eval["latency_mean_ms"],
        "fp32_tflite_latency_median_ms": tflite_eval["latency_median_ms"],
        "fp32_tflite_latency_std_ms": tflite_eval["latency_std_ms"],
        "fp32_tflite_latency_min_ms": tflite_eval["latency_min_ms"],
        "fp32_tflite_latency_max_ms": tflite_eval["latency_max_ms"],
        "fp32_warmup_runs": tflite_eval["warmup_runs"],
        "fp32_timed_runs": tflite_eval["latency_timed_runs"],
        "fp32_latency_measurement_method": tflite_eval["measurement_method"],
        "training_epochs_completed": len(history.history["loss"]),
        "final_train_loss": round(float(history.history["loss"][-1]), 4),
        "final_train_accuracy": round(float(history.history["accuracy"][-1]), 4),
        "final_val_loss": round(float(history.history["val_loss"][-1]), 4),
        "final_val_accuracy": round(float(history.history["val_accuracy"][-1]), 4),
    }

    update_metrics(baseline_metrics, results_file)
    print(f"Baseline metrics written to {results_file}")
    return model, baseline_metrics


if __name__ == "__main__":
    train_and_export_fp32()
