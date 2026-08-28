"""
Unit tests for FP32 baseline training and evaluation (Phase 4).
Verifies model persistence, TFLite interpreter loading, tensor shapes/types, and baseline metrics.
"""
import os
import unittest
import numpy as np
import tensorflow as tf

from src.metrics import load_metrics_json, get_file_size_bytes


class TestFP32Baseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.keras_path = "models/model_fp32.keras"
        cls.tflite_path = "models/model_fp32.tflite"
        cls.metrics_path = "results/metrics.json"

    def test_model_files_exist(self):
        self.assertTrue(os.path.exists(self.keras_path), f"Missing {self.keras_path}")
        self.assertTrue(os.path.exists(self.tflite_path), f"Missing {self.tflite_path}")
        self.assertTrue(os.path.exists(self.metrics_path), f"Missing {self.metrics_path}")

    def test_keras_model_loading_and_params(self):
        model = tf.keras.models.load_model(self.keras_path)
        self.assertIsNotNone(model)
        self.assertEqual(int(model.count_params()), 7834)
        self.assertEqual(model.input_shape, (None, 28, 28, 1))
        self.assertEqual(model.output_shape, (None, 10))

    def test_tflite_model_loading_and_inference(self):
        interpreter = tf.lite.Interpreter(model_path=self.tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        self.assertEqual(tuple(input_details["shape"]), (1, 28, 28, 1))
        self.assertEqual(input_details["dtype"], np.float32)
        self.assertEqual(tuple(output_details["shape"]), (1, 10))
        self.assertEqual(output_details["dtype"], np.float32)

        # Run sample inference
        dummy_input = np.ones((1, 28, 28, 1), dtype=np.float32)
        interpreter.set_tensor(input_details["index"], dummy_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])

        self.assertEqual(output.shape, (1, 10))
        self.assertAlmostEqual(float(np.sum(output)), 1.0, places=4)

    def test_metrics_json_integrity(self):
        metrics = load_metrics_json(self.metrics_path)
        self.assertIn("fp32_parameter_count", metrics)
        self.assertEqual(metrics["fp32_parameter_count"], 7834)

        self.assertIn("fp32_keras_accuracy", metrics)
        self.assertGreaterEqual(metrics["fp32_keras_accuracy"], 0.90)

        self.assertIn("fp32_tflite_accuracy", metrics)
        self.assertGreaterEqual(metrics["fp32_tflite_accuracy"], 0.90)

        self.assertIn("fp32_tflite_latency_ms", metrics)
        self.assertGreater(metrics["fp32_tflite_latency_ms"], 0.0)

        self.assertIn("fp32_tflite_size_bytes", metrics)
        self.assertEqual(metrics["fp32_tflite_size_bytes"], get_file_size_bytes(self.tflite_path))

        self.assertIn("fp32_keras_size_bytes", metrics)
        self.assertEqual(metrics["fp32_keras_size_bytes"], get_file_size_bytes(self.keras_path))


if __name__ == "__main__":
    unittest.main()
