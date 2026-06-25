# Keras model builder for Fashion-MNIST classifier

from tensorflow import keras


def build():
    """Flatten(28,28) → Dense(1024,gelu) → BN → Drop(0.4) → Dense(512,gelu) → BN → Drop(0.3) → Dense(256,gelu) → BN → Drop(0.3) → Dense(10,softmax)"""
    return keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(1024, activation="gelu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(512, activation="gelu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation="gelu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax"),
    ])
