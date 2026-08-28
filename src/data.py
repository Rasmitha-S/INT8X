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


def preprocess_canvas_drawing(
    canvas_image_data: np.ndarray,
    min_stroke_pixels: int = 15,
) -> Tuple[np.ndarray, Image.Image]:
    """
    Preprocesses raw RGBA drawing canvas image data into a standardized MNIST-compatible
    (28, 28) normalized float32 grayscale image in [0.0, 1.0].

    Steps:
    1. Validates input array dimensions and extracts luminance/alpha.
    2. Checks for blank canvas and raises ValueError if empty.
    3. Crops the digit bounding box and scales it into a 20x20 bounding box (preserving aspect ratio).
    4. Centers the 20x20 digit inside a 28x28 canvas with zero-padding (matching MNIST standard).
    5. Normalizes pixel intensities to [0.0, 1.0].

    Args:
        canvas_image_data: Numpy array of shape (H, W, 4) or (H, W, 3) from canvas component.
        min_stroke_pixels: Minimum number of non-background pixels required to consider valid.

    Returns:
        Tuple of (normalized (28, 28) float32 numpy array in [0.0, 1.0], PIL Image for display).

    Raises:
        ValueError: If the canvas is empty or no digit was drawn.
    """
    if canvas_image_data is None or not isinstance(canvas_image_data, np.ndarray):
        raise ValueError("Invalid canvas data: empty or not a numpy array.")

    if canvas_image_data.ndim < 2:
        raise ValueError("Invalid canvas dimensions.")

    # Extract 2D grayscale representation
    if canvas_image_data.ndim == 3:
        if canvas_image_data.shape[2] == 4:
            # RGBA
            r = canvas_image_data[:, :, 0].astype(np.float32)
            g = canvas_image_data[:, :, 1].astype(np.float32)
            b = canvas_image_data[:, :, 2].astype(np.float32)
            alpha = canvas_image_data[:, :, 3].astype(np.float32)

            gray = 0.299 * r + 0.587 * g + 0.114 * b
            if np.mean(alpha) < 250.0:
                gray = np.where(alpha > 20, gray, 0.0)
        elif canvas_image_data.shape[2] == 3:
            # RGB
            r = canvas_image_data[:, :, 0].astype(np.float32)
            g = canvas_image_data[:, :, 1].astype(np.float32)
            b = canvas_image_data[:, :, 2].astype(np.float32)
            gray = 0.299 * r + 0.587 * g + 0.114 * b
        else:
            gray = canvas_image_data[:, :, 0].astype(np.float32)
    else:
        gray = canvas_image_data.astype(np.float32)

    # Check if canvas background is white/light (corners > 127) and invert
    h, w = gray.shape[:2]
    corners = [gray[0, 0], gray[0, w - 1], gray[h - 1, 0], gray[h - 1, w - 1]]
    if np.mean(corners) > 127.0:
        gray = 255.0 - gray

    # Ensure non-negative and threshold low-level noise
    gray = np.clip(gray, 0.0, 255.0)
    gray = np.where(gray > 30.0, gray, 0.0)

    # Check if canvas is essentially blank
    non_zero_pixels = int(np.count_nonzero(gray > 30.0))
    if non_zero_pixels < min_stroke_pixels or float(np.max(gray)) < 40.0:
        raise ValueError("Please draw a digit before prediction.")

    # Find bounding box of the drawn digit
    y_indices, x_indices = np.where(gray > 20.0)
    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))

    # Crop the digit
    digit_crop = gray[y_min : y_max + 1, x_min : x_max + 1]
    crop_h, crop_w = digit_crop.shape

    if crop_h == 0 or crop_w == 0:
        raise ValueError("Please draw a digit before prediction.")

    # Scale to fit inside a 20x20 box preserving aspect ratio
    max_dim = max(crop_h, crop_w)
    scale_factor = 20.0 / max_dim
    new_h = max(1, int(round(crop_h * scale_factor)))
    new_w = max(1, int(round(crop_w * scale_factor)))

    crop_img = Image.fromarray(digit_crop.astype(np.uint8), mode="L")
    resized_crop = crop_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    resized_arr = np.array(resized_crop, dtype=np.float32)

    # Center inside a blank 28x28 canvas
    canvas_28 = np.zeros((28, 28), dtype=np.float32)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    canvas_28[start_y : start_y + new_h, start_x : start_x + new_w] = resized_arr

    # Normalize to [0.0, 1.0]
    norm_28 = np.clip(canvas_28 / 255.0, 0.0, 1.0).astype(np.float32)
    display_pil = Image.fromarray((norm_28 * 255.0).astype(np.uint8), mode="L")

    return norm_28, display_pil

