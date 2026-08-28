"""
Unit tests for lightweight CNN model (src/model.py).
Verifies model building, parameter limits (< 10,000), input/output shapes, layer hierarchy, forward pass, and metadata.
"""
import unittest
import numpy as np
import tensorflow as tf

from src.model import build_tiny_cnn, get_parameter_count, get_model_metadata


class TestModelArchitecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_tiny_cnn(input_shape=(28, 28, 1), num_classes=10)

    def test_model_build_and_shapes(self):
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.input_shape, (None, 28, 28, 1))
        self.assertEqual(self.model.output_shape, (None, 10))

    def test_parameter_count_limit(self):
        param_count = get_parameter_count(self.model)
        # Non-negotiable constraint: < 10,000 parameters
        self.assertLess(param_count, 10000)
        self.assertGreater(param_count, 0)
        self.assertEqual(param_count, 7834)

    def test_required_layers_presence(self):
        layer_types = [layer.__class__.__name__ for layer in self.model.layers]
        
        # Check required layer types
        self.assertIn("InputLayer", layer_types)
        self.assertEqual(layer_types.count("Conv2D"), 2)
        self.assertEqual(layer_types.count("MaxPooling2D"), 2)
        self.assertIn("Flatten", layer_types)
        self.assertEqual(layer_types.count("Dense"), 2)

        # Check final activation is softmax
        final_dense = [l for l in self.model.layers if isinstance(l, tf.keras.layers.Dense)][-1]
        self.assertEqual(final_dense.activation.__name__, "softmax")

    def test_forward_pass(self):
        # Create a batch of 2 dummy MNIST images (2, 28, 28, 1)
        dummy_input = np.random.uniform(0.0, 1.0, size=(2, 28, 28, 1)).astype(np.float32)
        predictions = self.model(dummy_input, training=False).numpy()

        self.assertEqual(predictions.shape, (2, 10))
        # Softmax outputs must be in [0.0, 1.0] and sum to ~1.0
        self.assertTrue(np.all(predictions >= 0.0))
        self.assertTrue(np.all(predictions <= 1.0))
        for row in predictions:
            self.assertAlmostEqual(float(np.sum(row)), 1.0, places=5)

    def test_model_metadata(self):
        meta = get_model_metadata(self.model)
        self.assertIn("total_parameters", meta)
        self.assertEqual(meta["total_parameters"], 7834)
        self.assertEqual(meta["input_shape"], (None, 28, 28, 1))
        self.assertEqual(meta["output_shape"], (None, 10))
        self.assertIsInstance(meta["layers"], list)
        self.assertEqual(len(meta["layers"]), 8)


if __name__ == "__main__":
    unittest.main()
