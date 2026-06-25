# Framework-agnostic utilities for Fashion-MNIST classifier

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from shared import print_metrics, log_results_csv


def load_data():
    """Load Fashion-MNIST via Keras, normalize to [0,1], return numpy arrays."""
    from tensorflow.keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return x_train, y_train, x_test, y_test


def get_labels():
    """Return the 10 Fashion-MNIST class name strings."""
    return ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
