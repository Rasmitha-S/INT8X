"""
End-to-End smoke test verifying all 10 sample digits through full inference pipeline (Phase 10).
"""
import unittest
from pathlib import Path
from PIL import Image

import app


class TestEndToEndSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.interpreter = app.get_interpreter("models/model_int8.tflite")
        cls.samples = app.get_available_sample_images()

    def test_all_ten_sample_digits(self):
        self.assertEqual(len(self.samples), 10, "Expected exactly 10 sample digits")

        for sample_path in self.samples:
            expected_digit = int(sample_path.name.split("_")[1])
            with open(sample_path, "rb") as f:
                norm_img, preview_img = app.preprocess_uploaded_image(f)

            res = app.run_inference(self.interpreter, norm_img)
            self.assertEqual(
                res["predicted_digit"],
                expected_digit,
                f"Mismatch on {sample_path.name}: expected {expected_digit}, got {res['predicted_digit']}",
            )
            self.assertGreater(res["confidence"], 0.50)
            self.assertGreater(res["latency_ms"], 0.0)

    def test_invalid_image_handling(self):
        # Create a corrupt/empty stream test
        with self.assertRaises(Exception):
            _ = app.preprocess_uploaded_image(b"not an image")


if __name__ == "__main__":
    unittest.main()
