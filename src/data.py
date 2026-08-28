"""
Data pipeline for MNIST loading, normalization, and INT8 calibration.
Provides standard MNIST datasets and representative data generator for Post-Training Quantization (PTQ).
"""
import os
from pathlib import Path
from typing import Callable, Generator, List, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf


def load_mnist() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Loads and preprocesses the standard MNIST dataset.

    Returns:
        ((x_train, y_train), (x_test, y_test)):
            x_train: np.ndarray of shape (60000, 28, 28, 1), float32 in [0.0, 1.0]
            y_train: np.ndarray of shape (60000,), uint8 / int64
            x_test:  np.ndarray of shape (10000, 28, 28, 1), float32 in [0.0, 1.0]
            y_test:  np.ndarray of shape (10000,), uint8 / int64
    """
    (x_train_raw, y_train), (x_test_raw, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values from [0, 255] to [0.0, 1.0] as float32
    x_train = (x_train_raw.astype(np.float32) / 255.0)
    x_test = (x_test_raw.astype(np.float32) / 255.0)

    # Reshape images to (N, 28, 28, 1) for 2D Convolution layers
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train), (x_test, y_test)


def get_representative_dataset_generator(
    x_train: np.ndarray, num_samples: int = 200
) -> Callable[[], Generator[List[np.ndarray], None, None]]:
    """
    Creates a calibration dataset generator for TFLite INT8 Post-Training Quantization.
    Uses real MNIST training samples.

    Args:
        x_train: Training input images array of shape (N, 28, 28, 1), float32.
        num_samples: Number of calibration samples to yield (default 200).

    Returns:
        A generator function suitable for `converter.representative_dataset`.
    """
    num_samples = min(num_samples, len(x_train))

    def representative_data_gen() -> Generator[List[np.ndarray], None, None]:
        for i in range(num_samples):
            # Model input requires shape (1, 28, 28, 1) and dtype float32
            sample = np.expand_dims(x_train[i], axis=0).astype(np.float32)
            yield [sample]

    return representative_data_gen


def save_sample_digits(
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str = "assets/sample_digits",
    samples_per_digit: int = 1,
) -> List[str]:
    """
    Extracts and saves genuine MNIST test images as PNG files for testing and UI inference.

    Args:
        x_test: Test images of shape (N, 28, 28, 1), float32 in [0.0, 1.0].
        y_test: Test labels of shape (N,).
        output_dir: Target directory path.
        samples_per_digit: Number of samples to save for each digit 0-9.

    Returns:
        List of saved image file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved_paths: List[str] = []
    saved_counts = {digit: 0 for digit in range(10)}

    for idx, (img_arr, label) in enumerate(zip(x_test, y_test)):
        digit = int(label)
        if saved_counts[digit] < samples_per_digit:
            # Convert normalized float32 [0.0, 1.0] back to uint8 [0, 255]
            img_uint8 = (img_arr.squeeze() * 255.0).astype(np.uint8)
            img = Image.fromarray(img_uint8, mode="L")
            filename = f"digit_{digit}_idx{idx}.png"
            file_path = out_path / filename
            img.save(file_path)
            saved_paths.append(str(file_path))
            saved_counts[digit] += 1

        if all(count >= samples_per_digit for count in saved_counts.values()):
            break

    return saved_paths
