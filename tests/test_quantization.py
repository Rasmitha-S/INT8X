"""
Unit tests for INT8 Post-Training Quantization (Phase 5).
Verifies full-integer INT8 quantization, tensor dtypes, scales, zero-points, and real inference execution.
"""
import os
import unittest
import numpy as np
import tensorflow as tf

from src.data import load_mnist
from src.metrics import get_file_size_bytes, load_metrics_json


class TestINT8Quantization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fp32_tflite_path = "models/model_fp32.tflite"
        cls.int8_tflite_path = "models/model_int8.tflite"
        cls.metrics_path = "results/metrics.json"
        (_, _), (cls.x_test, cls.y_test) = load_mnist()

    def test_model_files_exist(self):
        self.assertTrue(os.path.exists(self.int8_tflite_path), f"Missing {self.int8_tflite_path}")
        self.assertTrue(os.path.exists(self.fp32_tflite_path), f"Missing {self.fp32_tflite_path}")

    def test_int8_tflite_interpreter_and_tensors(self):
        interpreter = tf.lite.Interpreter(model_path=self.int8_tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        # Verify exact INT8 dtypes
        self.assertEqual(input_details["dtype"], np.int8, "Input tensor must be int8")
        self.assertEqual(output_details["dtype"], np.int8, "Output tensor must be int8")

        # Verify tensor shapes
        self.assertEqual(tuple(input_details["shape"]), (1, 28, 28, 1))
        self.assertEqual(tuple(output_details["shape"]), (1, 10))

        # Verify quantization parameters
        input_scale, input_zero_point = input_details["quantization"]
        output_scale, output_zero_point = output_details["quantization"]

        self.assertGreater(input_scale, 0.0)
        self.assertIsInstance(input_zero_point, (int, np.integer))
        self.assertGreater(output_scale, 0.0)
        self.assertIsInstance(output_zero_point, (int, np.integer))

    def test_real_sample_inference(self):
        interpreter = tf.lite.Interpreter(model_path=self.int8_tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        input_scale, input_zero_point = input_details["quantization"]
        output_scale, output_zero_point = output_details["quantization"]

        # Take real MNIST sample
        sample_float = self.x_test[0:1]  # (1, 28, 28, 1)
        sample_int8 = np.clip(
            np.round(sample_float / input_scale + input_zero_point), -128, 127
        ).astype(np.int8)

        interpreter.set_tensor(input_details["index"], sample_int8)
        interpreter.invoke()
        output_int8 = interpreter.get_tensor(output_details["index"])

        self.assertEqual(output_int8.shape, (1, 10))
        self.assertEqual(output_int8.dtype, np.int8)

        # Dequantize output
        probs = (output_int8.astype(np.float32) - output_zero_point) * output_scale
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=2)
        predicted_class = int(np.argmax(probs))
        self.assertEqual(predicted_class, int(self.y_test[0]))

    def test_model_size_reduction(self):
        fp32_size = get_file_size_bytes(self.fp32_tflite_path)
        int8_size = get_file_size_bytes(self.int8_tflite_path)

        self.assertGreater(int8_size, 0)
        self.assertLess(int8_size, fp32_size, "INT8 model must be smaller than FP32 model")

        # Verify distinct file contents
        with open(self.fp32_tflite_path, "rb") as f1, open(self.int8_tflite_path, "rb") as f2:
            self.assertNotEqual(f1.read(), f2.read(), "INT8 model must not be identical bytes to FP32 model")

    def test_metrics_json_contains_int8_metadata(self):
        metrics = load_metrics_json(self.metrics_path)
        self.assertTrue(metrics.get("int8_quantization_verified", False))
        self.assertEqual(metrics.get("int8_input_dtype"), "int8")
        self.assertEqual(metrics.get("int8_output_dtype"), "int8")
        self.assertIn("int8_tflite_size_bytes", metrics)
        self.assertIn("int8_input_scale", metrics)
        self.assertIn("int8_input_zero_point", metrics)

        # Ensure FP32 baseline values are intact
        self.assertIn("fp32_tflite_accuracy", metrics)
        self.assertEqual(metrics["fp32_parameter_count"], 7834)


if __name__ == "__main__":
    unittest.main()
