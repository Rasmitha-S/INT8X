"""
Unit tests for the INT8 Quantization Impact Analyzer (Feature 5).
Validates FP32 vs INT8 metrics extraction, size reduction math, compression ratio,
accuracy delta, latency changes, tensor dtypes, and impact summary consistency.
"""
import json
import unittest
from pathlib import Path

from src.metrics import get_file_size_bytes, get_file_size_kb, get_quantization_impact_summary


class TestQuantizationImpactAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comparison_path = "results/comparison.json"
        cls.metrics_path = "results/metrics.json"
        cls.fp32_model_path = "models/model_fp32.tflite"
        cls.int8_model_path = "models/model_int8.tflite"

        with open(cls.comparison_path, "r", encoding="utf-8") as f:
            cls.comparison_data = json.load(f)

        with open(cls.metrics_path, "r", encoding="utf-8") as f:
            cls.metrics_data = json.load(f)

        cls.impact = get_quantization_impact_summary(cls.comparison_data)

    def test_model_file_sizes_match_benchmarks(self):
        # FP32 model binary size
        fp32_bytes = get_file_size_bytes(self.fp32_model_path)
        fp32_kb = get_file_size_kb(self.fp32_model_path)
        self.assertEqual(fp32_bytes, 35536)
        self.assertEqual(fp32_kb, 34.70)
        self.assertEqual(self.impact["fp32_size_bytes"], 35536)
        self.assertEqual(self.impact["fp32_size_kb"], 34.70)

        # INT8 model binary size
        int8_bytes = get_file_size_bytes(self.int8_model_path)
        int8_kb = get_file_size_kb(self.int8_model_path)
        self.assertEqual(int8_bytes, 13824)
        self.assertEqual(int8_kb, 13.50)
        self.assertEqual(self.impact["int8_size_bytes"], 13824)
        self.assertEqual(self.impact["int8_size_kb"], 13.50)

    def test_size_reduction_and_compression_ratio(self):
        # Size reduction math: (35536 - 13824) / 35536 = 21712 / 35536 = 61.0986% -> 61.10%
        bytes_saved = self.impact["fp32_size_bytes"] - self.impact["int8_size_bytes"]
        self.assertEqual(bytes_saved, 21712)
        self.assertEqual(self.impact["bytes_saved"], 21712)

        calc_reduction = round((bytes_saved / self.impact["fp32_size_bytes"]) * 100.0, 2)
        self.assertAlmostEqual(calc_reduction, 61.10, delta=0.01)
        self.assertAlmostEqual(self.impact["size_reduction_percent"], 61.10, delta=0.01)

        # Compression ratio: 35536 / 13824 = 2.5706 -> 2.57x
        calc_compression = round(self.impact["fp32_size_bytes"] / self.impact["int8_size_bytes"], 2)
        self.assertAlmostEqual(calc_compression, 2.57, delta=0.01)
        self.assertAlmostEqual(self.impact["compression_ratio"], 2.57, delta=0.01)

    def test_accuracy_and_delta(self):
        # FP32: 98.44%, INT8: 98.46%, Delta: +0.02 pts
        self.assertEqual(self.impact["fp32_accuracy_percent"], 98.44)
        self.assertEqual(self.impact["int8_accuracy_percent"], 98.46)
        self.assertAlmostEqual(self.impact["accuracy_delta_points"], 0.02, delta=0.001)
        self.assertFalse(self.impact["accuracy_loss_observed"])
        self.assertIn("No accuracy loss observed", self.impact["accuracy_status_statement"])

    def test_latency_metrics(self):
        # FP32: 0.0133 ms, INT8: 0.0098 ms, Change: -26.32%
        self.assertAlmostEqual(self.impact["fp32_latency_mean_ms"], 0.0133, delta=0.0005)
        self.assertAlmostEqual(self.impact["int8_latency_mean_ms"], 0.0098, delta=0.0005)
        self.assertAlmostEqual(self.impact["latency_change_percent"], -26.32, delta=0.5)

    def test_tensor_dtypes_and_shapes(self):
        self.assertEqual(self.impact["fp32_input_dtype"], "float32")
        self.assertEqual(self.impact["int8_input_dtype"], "int8")
        self.assertEqual(self.impact["fp32_output_dtype"], "float32")
        self.assertEqual(self.impact["int8_output_dtype"], "int8")
        self.assertEqual(self.impact["fp32_input_shape"], "[1, 28, 28, 1]")
        self.assertEqual(self.impact["int8_input_shape"], "[1, 28, 28, 1]")
        self.assertEqual(self.impact["fp32_output_shape"], "[1, 10]")
        self.assertEqual(self.impact["int8_output_shape"], "[1, 10]")
        self.assertEqual(self.impact["bit_reduction_factor"], 4.0)

    def test_summary_fallback_with_default_file(self):
        # Ensure loading without passing arguments works directly from results/comparison.json
        summary = get_quantization_impact_summary()
        self.assertEqual(summary["fp32_size_bytes"], 35536)
        self.assertEqual(summary["int8_size_bytes"], 13824)
        self.assertEqual(summary["int8_accuracy_percent"], 98.46)


if __name__ == "__main__":
    unittest.main()
