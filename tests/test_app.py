"""
Unit tests for the Streamlit application (Phase 9).
Verifies artifact presence, loader functions, image preprocessing, inference execution,
and absence of hardcoded OS-specific paths.
"""
import json
import os
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import app


class TestStreamlitApp(unittest.TestCase):
    def test_required_artifacts_exist(self):
        self.assertTrue(os.path.exists("models/model_fp32.tflite"))
        self.assertTrue(os.path.exists("models/model_int8.tflite"))
        self.assertTrue(os.path.exists("results/metrics.json"))
        self.assertTrue(os.path.exists("results/comparison.json"))
        self.assertTrue(os.path.exists("tinyml/model_analysis.json"))
        self.assertTrue(os.path.exists("tinyml/model_data.h"))
        self.assertTrue(os.path.exists("tinyml/model_data.cc"))

    def test_app_data_loaders(self):
        comp = app.load_comparison_data()
        self.assertIsInstance(comp, dict)
        self.assertIn("fp32_baseline", comp)
        self.assertIn("int8_quantized", comp)
        self.assertIn("comparison_results", comp)

        metrics = app.load_metrics_data()
        self.assertIsInstance(metrics, dict)
        self.assertIn("fp32_tflite_accuracy", metrics)
        self.assertIn("int8_tflite_accuracy", metrics)

        tinyml_data = app.load_tinyml_analysis()
        self.assertIsInstance(tinyml_data, dict)
        self.assertIn("verified", tinyml_data)
        self.assertIn("estimated", tinyml_data)
        self.assertIn("not_verified", tinyml_data)

    def test_sample_digits_available(self):
        samples = app.get_available_sample_images()
        self.assertGreaterEqual(len(samples), 10)
        for s in samples:
            self.assertTrue(s.exists())
            with Image.open(s) as img:
                self.assertEqual(img.size, (28, 28))

    def test_app_preprocessing_and_inference(self):
        samples = app.get_available_sample_images()
        self.assertGreater(len(samples), 0)

        with open(samples[0], "rb") as f:
            norm_arr, display_img = app.preprocess_uploaded_image(f)

        self.assertEqual(norm_arr.shape, (28, 28))
        self.assertGreaterEqual(float(norm_arr.min()), 0.0)
        self.assertLessEqual(float(norm_arr.max()), 1.0)
        self.assertEqual(display_img.size, (28, 28))

        interpreter = app.get_interpreter("models/model_int8.tflite")
        res = app.run_inference(interpreter, norm_arr)

        self.assertIn("predicted_digit", res)
        self.assertTrue(0 <= res["predicted_digit"] <= 9)
        self.assertIn("confidence", res)
        self.assertTrue(0.0 <= res["confidence"] <= 1.0)
        self.assertIn("latency_ms", res)
        self.assertGreater(res["latency_ms"], 0.0)

    def test_no_hardcoded_absolute_windows_paths(self):
        with open("app.py", "r", encoding="utf-8") as f:
            app_code = f.read()

        # Ensure no hardcoded C: / C:\ or absolute machine paths in app.py
        self.assertNotIn("C:\\", app_code)
        self.assertNotIn("C:/", app_code)
        self.assertNotIn("c:\\", app_code)
        self.assertNotIn("c:/", app_code)
        self.assertNotIn("rasmi", app_code.lower())


if __name__ == "__main__":
    unittest.main()
