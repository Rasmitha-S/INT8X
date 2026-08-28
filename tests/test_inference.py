"""
Unit tests for the inference engine (Phase 6).
Verifies dynamic quantization handling, input quantization, output dequantization,
single-sample execution across shapes, and FP32 vs INT8 agreement on test samples.
"""
import os
import unittest
import numpy as np

from src.data import load_mnist
from src.inference import (
    load_tflite_interpreter,
    get_quantization_details,
    quantize_input_image,
    dequantize_output_tensor,
    run_inference,
    compare_sample_predictions,
)


class TestInferenceEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fp32_model_path = "models/model_fp32.tflite"
        cls.int8_model_path = "models/model_int8.tflite"
        (_, _), (cls.x_test, cls.y_test) = load_mnist()

    def test_interpreter_loading(self):
        fp32_interp = load_tflite_interpreter(self.fp32_model_path)
        int8_interp = load_tflite_interpreter(self.int8_model_path)
        self.assertIsNotNone(fp32_interp)
        self.assertIsNotNone(int8_interp)

    def test_dynamic_quantization_details(self):
        int8_interp = load_tflite_interpreter(self.int8_model_path)
        q_int8 = get_quantization_details(int8_interp)

        self.assertTrue(q_int8["is_quantized"])
        self.assertEqual(q_int8["input_dtype"], np.int8)
        self.assertEqual(q_int8["output_dtype"], np.int8)
        self.assertGreater(q_int8["input_scale"], 0.0)
        self.assertGreater(q_int8["output_scale"], 0.0)
        self.assertEqual(q_int8["input_shape"], (1, 28, 28, 1))
        self.assertEqual(q_int8["output_shape"], (1, 10))

        fp32_interp = load_tflite_interpreter(self.fp32_model_path)
        q_fp32 = get_quantization_details(fp32_interp)
        self.assertFalse(q_fp32["is_quantized"])
        self.assertEqual(q_fp32["input_dtype"], np.float32)
        self.assertEqual(q_fp32["output_dtype"], np.float32)

    def test_input_quantization_math(self):
        scale = 1.0 / 255.0
        zero_point = -128

        # 0.0 float -> -128 int8
        val_0 = quantize_input_image(np.array([0.0], dtype=np.float32), scale, zero_point)
        self.assertEqual(val_0[0], -128)
        self.assertEqual(val_0.dtype, np.int8)

        # 1.0 float -> 127 int8
        val_1 = quantize_input_image(np.array([1.0], dtype=np.float32), scale, zero_point)
        self.assertEqual(val_1[0], 127)
        self.assertEqual(val_1.dtype, np.int8)

        # Values outside [0, 1] must clip safely to [-128, 127]
        val_clipped = quantize_input_image(np.array([-10.0, 10.0], dtype=np.float32), scale, zero_point)
        self.assertEqual(val_clipped[0], -128)
        self.assertEqual(val_clipped[1], 127)

    def test_output_dequantization_math(self):
        scale = 1.0 / 256.0
        zero_point = -128

        # -128 raw int8 -> 0.0 float
        out_0 = dequantize_output_tensor(np.array([-128], dtype=np.int8), scale, zero_point)
        self.assertAlmostEqual(float(out_0[0]), 0.0, places=5)

        # 128 (or ~127) raw int8 -> ~1.0 float
        out_1 = dequantize_output_tensor(np.array([128], dtype=np.int16), scale, zero_point)
        self.assertAlmostEqual(float(out_1[0]), 1.0, places=5)

    def test_int8_inference_various_input_shapes(self):
        int8_interp = load_tflite_interpreter(self.int8_model_path)
        sample_3d = self.x_test[0]  # (28, 28, 1)
        sample_2d = sample_3d.squeeze()  # (28, 28)
        sample_4d = self.x_test[0:1]  # (1, 28, 28, 1)

        for inp in [sample_2d, sample_3d, sample_4d]:
            res = run_inference(int8_interp, inp)
            self.assertIn("predicted_digit", res)
            self.assertIn("confidence", res)
            self.assertIn("latency_ms", res)
            self.assertIn("probabilities", res)

            self.assertIsInstance(res["predicted_digit"], int)
            self.assertTrue(0 <= res["predicted_digit"] <= 9)
            self.assertTrue(0.0 <= res["confidence"] <= 1.0)
            self.assertGreater(res["latency_ms"], 0.0)
            self.assertEqual(len(res["probabilities"]), 10)
            self.assertAlmostEqual(sum(res["probabilities"]), 1.0, places=3)
            self.assertEqual(res["predicted_digit"], int(self.y_test[0]))

    def test_fp32_vs_int8_limited_agreement(self):
        # Limited verification subset of 50 samples
        subset_samples = self.x_test[:50]
        subset_labels = self.y_test[:50]

        comparison = compare_sample_predictions(
            self.fp32_model_path,
            self.int8_model_path,
            subset_samples,
            subset_labels,
        )

        self.assertEqual(comparison["samples_evaluated"], 50)
        self.assertGreaterEqual(comparison["matching_predictions"], 45)
        self.assertGreaterEqual(comparison["agreement_rate"], 0.90)
        self.assertGreaterEqual(comparison["fp32_sample_accuracy"], 0.90)
        self.assertGreaterEqual(comparison["int8_sample_accuracy"], 0.90)
        self.assertEqual(len(comparison["sample_details"]), 50)


if __name__ == "__main__":
    unittest.main()
