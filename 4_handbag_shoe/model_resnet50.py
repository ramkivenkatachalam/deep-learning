# ResNet50 transfer learning for handbag-shoe classifier
# Two approaches: cached feature extraction (fast) and end-to-end (enables fine-tuning).

import keras
import numpy as np


def get_resnet50_base():
    """Return a frozen, headless ResNet50 pretrained on ImageNet."""
    base = keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base.trainable = False
    return base


def extract_features(dataset, resnet50_base):
    """Run dataset through frozen ResNet50 base, return (features, labels) numpy arrays.

    Expects dataset with integer labels and raw pixel values [0, 255].
    ResNet50 preprocess_input handles normalization.
    """
    all_features = []
    all_labels = []
    for images, labels in dataset:
        preprocessed = keras.applications.resnet50.preprocess_input(images)
        features = resnet50_base(preprocessed, training=False)
        all_features.append(features.numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)


def build_resnet50_head():
    """Classification head: Input(7,7,2048) → Flatten → Dense(256) → Dropout → Dense(1, sigmoid)."""
    input = keras.Input(shape=(7, 7, 2048))
    h = keras.layers.Flatten()(input)
    h = keras.layers.Dense(256, activation="relu")(h)
    h = keras.layers.Dropout(0.5)(h)
    output = keras.layers.Dense(1, activation="sigmoid")(h)
    return keras.Model(input, output)


def build_resnet50_e2e():
    """End-to-end ResNet50 model with augmentation. Enables fine-tuning."""
    base = keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base.trainable = False

    input = keras.Input(shape=(224, 224, 3))
    h = keras.layers.RandomFlip("horizontal")(input)
    h = keras.layers.RandomRotation(0.1)(h)
    h = keras.layers.RandomZoom(0.2)(h)
    h = keras.applications.resnet50.preprocess_input(h)
    h = base(h, training=False)
    h = keras.layers.Flatten()(h)
    h = keras.layers.Dense(256, activation="relu")(h)
    h = keras.layers.Dropout(0.5)(h)
    output = keras.layers.Dense(1, activation="sigmoid")(h)

    return keras.Model(input, output)
