"""
Lightweight CNN architecture definition for TinyML deployment on MNIST.
Designed to have minimal parameter count (< 10,000) and full compatibility with TFLite and TFLite Micro.
"""
from typing import Any, Dict, Tuple
import tensorflow as tf


def build_tiny_cnn(
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
) -> tf.keras.Model:
    """
    Constructs a lightweight TinyML-compatible CNN for MNIST classification.

    Architecture:
        Input: (28, 28, 1)
        Conv2D(8, 3x3, relu) -> (26, 26, 8)
        MaxPooling2D(2x2)     -> (13, 13, 8)
        Conv2D(16, 3x3, relu)-> (11, 11, 16)
        MaxPooling2D(2x2)     -> (5, 5, 16)
        Flatten               -> (400)
        Dense(16, relu)       -> (16)
        Dense(num_classes, softmax) -> (10)

    Args:
        input_shape: Dimensions of input images, default (28, 28, 1).
        num_classes: Number of output classification categories, default 10.

    Returns:
        Compiled or uncompiled tf.keras.Model instance.
    """
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_image")
    
    # Layer 1: Conv2D + ReLU + MaxPool
    x = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=(3, 3),
        activation="relu",
        padding="valid",
        name="conv2d_1",
    )(inputs)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name="max_pool_1",
    )(x)

    # Layer 2: Conv2D + ReLU + MaxPool
    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        activation="relu",
        padding="valid",
        name="conv2d_2",
    )(x)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name="max_pool_2",
    )(x)

    # Classification head
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(
        16,
        activation="relu",
        name="dense_1",
    )(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="output_probabilities",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="tiny_mnist_cnn")
    return model


def get_parameter_count(model: tf.keras.Model) -> int:
    """
    Computes total trainable + non-trainable parameters in the model.

    Args:
        model: tf.keras.Model instance.

    Returns:
        Total parameter count as an integer.
    """
    return int(model.count_params())


def get_model_metadata(model: tf.keras.Model) -> Dict[str, Any]:
    """
    Extracts structural metadata of the CNN model.

    Args:
        model: tf.keras.Model instance.

    Returns:
        Dictionary containing input_shape, output_shape, total_params, layer_names, and layer_types.
    """
    return {
        "model_name": model.name,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "total_parameters": get_parameter_count(model),
        "trainable_parameters": sum(
            tf.keras.backend.count_params(w) for w in model.trainable_weights
        ),
        "layers": [
            {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": tuple(layer.output.shape) if hasattr(layer, "output") and layer.output is not None else tuple(getattr(layer, "shape", ())),
                "params": int(layer.count_params()),
            }
            for layer in model.layers
        ],
    }
