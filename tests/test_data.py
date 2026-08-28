"""
Unit tests for data pipeline (src/data.py).
Verifies dataset loading, normalization, shapes, types, PTQ calibration generator, and sample image generation.
"""
import os
import shutil
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.data import load_mnist, get_representative_dataset_generator, save_sample_digits


class TestDataPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.x_train, cls.y_train), (cls.x_test, cls.y_test) = load_mnist()

    def test_dataset_shapes_and_types(self):
        # Training set
        self.assertEqual(self.x_train.shape, (60000, 28, 28, 1))
        self.assertEqual(self.y_train.shape, (60000,))
        self.assertEqual(self.x_train.dtype, np.float32)

        # Test set
        self.assertEqual(self.x_test.shape, (10000, 28, 28, 1))
        self.assertEqual(self.y_test.shape, (10000,))
        self.assertEqual(self.x_test.dtype, np.float32)

        # Labels range check (0-9)
        self.assertEqual(set(np.unique(self.y_train)), set(range(10)))
        self.assertEqual(set(np.unique(self.y_test)), set(range(10)))

    def test_normalization_range(self):
        # Pixel values must be scaled to [0.0, 1.0]
        self.assertGreaterEqual(float(self.x_train.min()), 0.0)
        self.assertLessEqual(float(self.x_train.max()), 1.0)

        self.assertGreaterEqual(float(self.x_test.min()), 0.0)
        self.assertLessEqual(float(self.x_test.max()), 1.0)

    def test_representative_dataset_generator(self):
        num_calibration_samples = 50
        gen_factory = get_representative_dataset_generator(self.x_train, num_samples=num_calibration_samples)
        gen = gen_factory()

        samples_collected = 0
        for item in gen:
            self.assertIsInstance(item, list)
            self.assertEqual(len(item), 1)
            sample = item[0]
            self.assertEqual(sample.shape, (1, 28, 28, 1))
            self.assertEqual(sample.dtype, np.float32)
            self.assertGreaterEqual(float(sample.min()), 0.0)
            self.assertLessEqual(float(sample.max()), 1.0)
            samples_collected += 1

        self.assertEqual(samples_collected, num_calibration_samples)

    def test_save_sample_digits(self):
        test_dir = "assets/sample_digits"
        saved_paths = save_sample_digits(self.x_test, self.y_test, output_dir=test_dir, samples_per_digit=1)

        self.assertEqual(len(saved_paths), 10)
        for p in saved_paths:
            self.assertTrue(os.path.exists(p))
            with Image.open(p) as img:
                self.assertEqual(img.size, (28, 28))
                self.assertEqual(img.mode, "L")


if __name__ == "__main__":
    unittest.main()
