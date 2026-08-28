"""
Unit tests for Canvas Drawing Preprocessing & Inference (Feature 3).
Verifies canvas RGBA preprocessing, centering, normalization, blank detection,
and INT8 model inference execution.
"""
import unittest
import numpy as np
from PIL import Image

from src.data import preprocess_canvas_drawing
from src.inference import load_tflite_interpreter, run_inference


class TestDrawingPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.interpreter = load_tflite_interpreter("models/model_int8.tflite")

    def test_synthetic_canvas_drawing_preprocessing(self):
        # Create a 200x200 white canvas with a black drawn stroke (e.g. a vertical bar representing digit '1')
        canvas_rgba = np.full((200, 200, 4), 255, dtype=np.uint8)
        # Draw a vertical line from (50, 90) to (150, 110) in black (0, 0, 0, 255)
        canvas_rgba[50:150, 95:105, 0:3] = 0

        norm_arr, display_img = preprocess_canvas_drawing(canvas_rgba)

        # Output shape must be (28, 28)
        self.assertEqual(norm_arr.shape, (28, 28))
        self.assertEqual(norm_arr.dtype, np.float32)

        # Output range must be in [0.0, 1.0]
        self.assertGreaterEqual(float(norm_arr.min()), 0.0)
        self.assertLessEqual(float(norm_arr.max()), 1.0)

        # Max pixel should have high intensity
        self.assertGreater(float(norm_arr.max()), 0.5)

        # PIL image should be 28x28
        self.assertEqual(display_img.size, (28, 28))
        self.assertEqual(display_img.mode, "L")

    def test_blank_canvas_detection(self):
        # Completely white canvas
        white_canvas = np.full((200, 200, 4), 255, dtype=np.uint8)
        with self.assertRaises(ValueError):
            _ = preprocess_canvas_drawing(white_canvas)

        # Completely transparent canvas
        transparent_canvas = np.zeros((200, 200, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            _ = preprocess_canvas_drawing(transparent_canvas)

        # Canvas with just a single stray pixel (below min_stroke_pixels)
        stray_canvas = np.full((200, 200, 4), 255, dtype=np.uint8)
        stray_canvas[100, 100, 0:3] = 0
        with self.assertRaises(ValueError):
            _ = preprocess_canvas_drawing(stray_canvas)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            _ = preprocess_canvas_drawing(None)

        with self.assertRaises(ValueError):
            _ = preprocess_canvas_drawing(np.array([1, 2, 3]))

    def test_inference_on_preprocessed_drawing(self):
        # Create a synthetic digit '1'
        canvas_rgba = np.full((200, 200, 4), 255, dtype=np.uint8)
        canvas_rgba[40:160, 95:105, 0:3] = 0

        norm_arr, _ = preprocess_canvas_drawing(canvas_rgba)
        res = run_inference(self.interpreter, norm_arr)

        self.assertIn("predicted_digit", res)
        self.assertTrue(0 <= res["predicted_digit"] <= 9)
        self.assertIn("confidence", res)
        self.assertTrue(0.0 <= res["confidence"] <= 1.0)
        self.assertIn("latency_ms", res)
        self.assertGreater(res["latency_ms"], 0.0)
        self.assertEqual(len(res["probabilities"]), 10)


if __name__ == "__main__":
    unittest.main()
