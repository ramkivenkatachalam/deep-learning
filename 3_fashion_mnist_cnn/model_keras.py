# Keras model builder for Fashion-MNIST CNN classifier

from tensorflow import keras


def build():
    """3 conv blocks (64→128→256, 2×Conv2D+BN+MaxPool+Drop) → Dense(512)+BN+Drop → Dense(10)"""
    input = keras.Input(shape=(28, 28, 1))
    h = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(input)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(h)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.MaxPooling2D((2, 2))(h)
    h = keras.layers.Dropout(0.25)(h)

    h = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(h)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(h)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.MaxPooling2D((2, 2))(h)
    h = keras.layers.Dropout(0.25)(h)

    h = keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(h)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(h)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.MaxPooling2D((2, 2))(h)
    h = keras.layers.Dropout(0.25)(h)

    h = keras.layers.Flatten()(h)
    h = keras.layers.Dense(512, activation="relu")(h)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.Dropout(0.5)(h)
    output = keras.layers.Dense(10, activation="softmax")(h)
    return keras.Model(input, output)
