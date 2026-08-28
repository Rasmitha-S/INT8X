"""
Environment validation test for PS09 TinyQuant.
Verifies all required libraries and basic TFLite functionality.
"""
import sys
import unittest


class TestEnvironment(unittest.TestCase):
    def test_python_version(self):
        self.assertGreaterEqual(sys.version_info, (3, 9))

    def test_imports(self):
        import tensorflow as tf
        import numpy as np
        import sklearn
        import streamlit as st
        import plotly
        import PIL

        self.assertTrue(tf.__version__.startswith("2."))
        self.assertTrue(hasattr(tf, "lite"))
        self.assertTrue(hasattr(tf.lite, "TFLiteConverter"))
        self.assertTrue(hasattr(tf.lite, "Interpreter"))

    def test_tflite_interpreter_dummy(self):
        import tensorflow as tf
        import numpy as np

        # Create a simple 1-layer model and convert to TFLite to verify TFLite runtime
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(1,))
        ])
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        self.assertGreater(len(tflite_model), 0)

        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertEqual(len(input_details), 1)


if __name__ == "__main__":
    unittest.main()
