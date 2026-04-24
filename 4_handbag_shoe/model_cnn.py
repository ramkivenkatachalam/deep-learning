# CNN model builders for handbag-shoe classifier

import keras


def build_cnn():
    """Basic CNN: Rescaling → Conv2D(32) → MaxPool → Conv2D(32) → MaxPool → Flatten → Dense(1, sigmoid)."""
    input = keras.Input(shape=(224, 224, 3))

    h = keras.layers.Rescaling(1.0 / 255)(input)

    h = keras.layers.Conv2D(32, kernel_size=(2, 2), activation="relu", name="Conv_1")(h)
    h = keras.layers.MaxPool2D()(h)

    h = keras.layers.Conv2D(32, kernel_size=(2, 2), activation="relu", name="Conv_2")(h)
    h = keras.layers.MaxPool2D()(h)

    h = keras.layers.Flatten()(h)
    output = keras.layers.Dense(1, activation="sigmoid")(h)

    return keras.Model(input, output)


def build_cnn_augmented():
    """CNN with data augmentation layers prepended."""
    input = keras.Input(shape=(224, 224, 3))

    h = keras.layers.RandomFlip("horizontal")(input)
    h = keras.layers.RandomRotation(0.1)(h)
    h = keras.layers.RandomZoom(0.2)(h)

    h = keras.layers.Rescaling(1.0 / 255)(h)

    h = keras.layers.Conv2D(32, kernel_size=(2, 2), activation="relu", name="Conv_1")(h)
    h = keras.layers.MaxPool2D()(h)

    h = keras.layers.Conv2D(32, kernel_size=(2, 2), activation="relu", name="Conv_2")(h)
    h = keras.layers.MaxPool2D()(h)

    h = keras.layers.Flatten()(h)
    output = keras.layers.Dense(1, activation="sigmoid")(h)

    return keras.Model(input, output)
