# CNN model builders for handbag-shoe classifier

import keras
from keras.layers import (
    Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization,
    Rescaling, RandomFlip, RandomRotation, RandomZoom, ReLU, Add, GlobalAveragePooling2D,
)


def build_cnn():
    """CNN: Rescaling → 2x Conv2D(32, 3x3)+MaxPool → Flatten → Dense(128) → Dropout → Dense(1)."""
    input = keras.Input(shape=(224, 224, 3))

    h = Rescaling(1.0 / 255)(input)

    h = Conv2D(32, kernel_size=(3, 3), activation="relu", name="Conv_1")(h)
    h = MaxPool2D()(h)

    h = Conv2D(32, kernel_size=(3, 3), activation="relu", name="Conv_2")(h)
    h = MaxPool2D()(h)

    h = Flatten()(h)
    h = Dense(128, activation="relu")(h)
    h = Dropout(0.5)(h)
    output = Dense(1, activation="sigmoid")(h)

    return keras.Model(input, output)


def build_cnn_augmented():
    """CNN with data augmentation layers prepended."""
    input = keras.Input(shape=(224, 224, 3))

    h = RandomFlip("horizontal")(input)
    h = RandomRotation(0.05)(h)
    h = RandomZoom(0.1)(h)

    h = Rescaling(1.0 / 255)(h)

    h = Conv2D(32, kernel_size=(3, 3), activation="relu", name="Conv_1")(h)
    h = MaxPool2D()(h)

    h = Conv2D(32, kernel_size=(3, 3), activation="relu", name="Conv_2")(h)
    h = MaxPool2D()(h)

    h = Flatten()(h)
    h = Dense(128, activation="relu")(h)
    h = Dropout(0.5)(h)
    output = Dense(1, activation="sigmoid")(h)

    return keras.Model(input, output)


def _residual_block(x, filters):
    """Two 3x3 convs with a skip connection. Projects shortcut if channels don't match."""
    shortcut = x

    h = Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    h = Conv2D(filters, (3, 3), padding="same")(h)

    # Project shortcut if input channels != output channels
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1))(shortcut)

    h = Add()([h, shortcut])
    h = ReLU()(h)
    return h


def build_cnn_residual():
    """CNN with residual (skip) connections.

    Rescaling → ResBlock(32) → Pool → ResBlock(64) → Pool → ResBlock(128)
    → GlobalAvgPool → Dense(128) → Dropout → Dense(1, sigmoid)
    """
    input = keras.Input(shape=(224, 224, 3))

    h = Rescaling(1.0 / 255)(input)

    h = _residual_block(h, 32)
    h = MaxPool2D()(h)

    h = _residual_block(h, 64)
    h = MaxPool2D()(h)

    h = _residual_block(h, 128)
    h = MaxPool2D()(h)
    h = Flatten()(h)

    h = Dense(128, activation="relu")(h)
    h = Dropout(0.5)(h)
    output = Dense(1, activation="sigmoid")(h)

    return keras.Model(input, output)
