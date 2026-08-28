"""
Unit tests for FP32 vs INT8 benchmarking (Phase 7).
Verifies complete 10,000-sample test coverage, accuracy and latency bounds,
mathematical consistency of comparison calculations, and JSON result integrity.
"""
import json
import os
import unittest
from pathlib import Path

import tensorflow as tf

from src.metrics import get_file_size_bytes, load_metrics_json


class TestBenchmarkResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fp32_model_path = "models/model_fp32.tflite"
        cls.int8_model_path = "models/model_int8.tflite"
        cls.metrics_path = "results/metrics.json"
        cls.comparison_path = "results/comparison.json"

    def test_model_and_result_files_exist(self):
        self.assertTrue(os.path.exists(self.fp32_model_path))
        self.assertTrue(os.path.exists(self.int8_model_path))
        self.assertTrue(os.path.exists(self.metrics_path))
        self.assertTrue(os.path.exists(self.comparison_path))

    def test_tflite_interpreters_load(self):
        fp32_interp = tf.lite.Interpreter(model_path=self.fp32_model_path)
        fp32_interp.allocate_tensors()
        self.assertIsNotNone(fp32_interp)

        int8_interp = tf.lite.Interpreter(model_path=self.int8_model_path)
        int8_interp.allocate_tensors()
        self.assertIsNotNone(int8_interp)

    def test_comparison_json_structure_and_bounds(self):
        with open(self.comparison_path, "r", encoding="utf-8") as f:
            comp = json.load(f)

        self.assertIn("fp32_baseline", comp)
        self.assertIn("int8_quantized", comp)
        self.assertIn("comparison_results", comp)
        self.assertIn("benchmark_environment", comp)
        self.assertIn("methodology", comp)

        fp32 = comp["fp32_baseline"]
        int8 = comp["int8_quantized"]
        res = comp["comparison_results"]

        # Sample count verification
        self.assertEqual(fp32["total_samples"], 10000)
        self.assertEqual(int8["total_samples"], 10000)

        # Accuracy bounds
        self.assertTrue(0.0 <= fp32["accuracy"] <= 1.0)
        self.assertTrue(0.0 <= int8["accuracy"] <= 1.0)
        self.assertGreaterEqual(fp32["accuracy"], 0.95)
        self.assertGreaterEqual(int8["accuracy"], 0.95)

        # File sizes match actual disk sizes
        actual_fp32_size = get_file_size_bytes(self.fp32_model_path)
        actual_int8_size = get_file_size_bytes(self.int8_model_path)
        self.assertEqual(fp32["size_bytes"], actual_fp32_size)
        self.assertEqual(int8["size_bytes"], actual_int8_size)
        self.assertLess(actual_int8_size, actual_fp32_size)

        # Latencies positive
        self.assertGreater(fp32["latency_mean_ms"], 0.0)
        self.assertGreater(int8["latency_mean_ms"], 0.0)

    def test_mathematical_consistency(self):
        with open(self.comparison_path, "r", encoding="utf-8") as f:
            comp = json.load(f)

        fp32 = comp["fp32_baseline"]
        int8 = comp["int8_quantized"]
        res = comp["comparison_results"]

        # Size reduction %
        expected_size_red = round(((fp32["size_bytes"] - int8["size_bytes"]) / fp32["size_bytes"]) * 100.0, 2)
        self.assertAlmostEqual(res["size_reduction_percent"], expected_size_red, places=2)

        # Compression ratio
        expected_comp_ratio = round(fp32["size_bytes"] / int8["size_bytes"], 2)
        self.assertAlmostEqual(res["compression_ratio"], expected_comp_ratio, places=2)

        # Accuracy delta
        expected_acc_delta = round(int8["accuracy"] - fp32["accuracy"], 4)
        self.assertAlmostEqual(res["accuracy_delta"], expected_acc_delta, places=4)

        # Latency change %
        expected_lat_change = round(((int8["latency_mean_ms"] - fp32["latency_mean_ms"]) / fp32["latency_mean_ms"]) * 100.0, 2)
        self.assertAlmostEqual(res["latency_change_percent"], expected_lat_change, places=2)

    def test_metrics_json_baseline_preservation(self):
        metrics = load_metrics_json(self.metrics_path)
        self.assertEqual(metrics["fp32_parameter_count"], 7834)
        self.assertIn("fp32_keras_accuracy", metrics)
        self.assertIn("fp32_tflite_accuracy", metrics)
        self.assertIn("int8_tflite_accuracy", metrics)
        self.assertIn("size_reduction_percent", metrics)
        self.assertIn("compression_ratio", metrics)


if __name__ == "__main__":
    unittest.main()
