"""
Unit tests for FP32 vs INT8 Confusion Matrix (Feature 2).
Verifies 10x10 shape, non-negativity, 10,000 sample totals, diagonal accuracy parity (98.44% FP32, 98.46% INT8),
and consistency with existing benchmark metrics.
"""
import json
import os
import unittest
from pathlib import Path
import numpy as np

import app


class TestConfusionMatrixFeature(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cm_file = "results/confusion_matrices.json"
        cls.comparison_file = "results/comparison.json"
        cls.metrics_file = "results/metrics.json"

    def test_confusion_matrix_file_exists_and_loads(self):
        self.assertTrue(os.path.exists(self.cm_file))
        data = app.load_confusion_matrix_data()
        self.assertIsInstance(data, dict)
        self.assertIn("fp32", data)
        self.assertIn("int8", data)
        self.assertIn("classes", data)
        self.assertIn("class_counts", data)
        self.assertEqual(data["classes"], list(range(10)))

    def test_fp32_matrix_integrity(self):
        with open(self.cm_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        fp32_info = data["fp32"]
        mat = np.array(fp32_info["matrix"])

        # Shape (10, 10)
        self.assertEqual(mat.shape, (10, 10))

        # All values non-negative
        self.assertTrue(np.all(mat >= 0))

        # Total entries equal 10,000
        self.assertEqual(int(mat.sum()), 10000)

        # Row sums match class counts
        row_sums = mat.sum(axis=1).tolist()
        self.assertEqual(row_sums, data["class_counts"])

        # Diagonal sum equals correct predictions (9,844)
        diag_sum = int(np.trace(mat))
        self.assertEqual(diag_sum, 9844)
        self.assertEqual(fp32_info["correct_predictions"], 9844)
        self.assertEqual(fp32_info["incorrect_predictions"], 156)
        self.assertAlmostEqual(fp32_info["accuracy_percent"], 98.44, places=2)

    def test_int8_matrix_integrity(self):
        with open(self.cm_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        int8_info = data["int8"]
        mat = np.array(int8_info["matrix"])

        # Shape (10, 10)
        self.assertEqual(mat.shape, (10, 10))

        # All values non-negative
        self.assertTrue(np.all(mat >= 0))

        # Total entries equal 10,000
        self.assertEqual(int(mat.sum()), 10000)

        # Row sums match class counts
        row_sums = mat.sum(axis=1).tolist()
        self.assertEqual(row_sums, data["class_counts"])

        # Diagonal sum equals correct predictions (9,846)
        diag_sum = int(np.trace(mat))
        self.assertEqual(diag_sum, 9846)
        self.assertEqual(int8_info["correct_predictions"], 9846)
        self.assertEqual(int8_info["incorrect_predictions"], 154)
        self.assertAlmostEqual(int8_info["accuracy_percent"], 98.46, places=2)

    def test_accuracy_consistency_with_benchmark_results(self):
        with open(self.cm_file, "r", encoding="utf-8") as f:
            cm_data = json.load(f)

        with open(self.comparison_file, "r", encoding="utf-8") as f:
            comp_data = json.load(f)

        # FP32 accuracy parity
        self.assertEqual(
            cm_data["fp32"]["accuracy_percent"],
            comp_data["fp32_baseline"]["accuracy_percent"],
        )
        self.assertEqual(
            cm_data["fp32"]["correct_predictions"],
            comp_data["fp32_baseline"]["correct_predictions"],
        )

        # INT8 accuracy parity
        self.assertEqual(
            cm_data["int8"]["accuracy_percent"],
            comp_data["int8_quantized"]["accuracy_percent"],
        )
        self.assertEqual(
            cm_data["int8"]["correct_predictions"],
            comp_data["int8_quantized"]["correct_predictions"],
        )


if __name__ == "__main__":
    unittest.main()
