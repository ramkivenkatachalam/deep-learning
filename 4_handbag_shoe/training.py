# Handbag-Shoe binary classifier
# Based on MIT 15.773 Hands-On Deep Learning
# Original colab: https://colab.research.google.com/drive/1q42h-IJTmxjKeG5f4E_2v7jQW0vDZi0T
#
# Dataset: Handbags vs Shoes (small custom dataset, 224x224 RGB)
# Models: cnn, cnn_augmented, resnet50, resnet50_e2e
# Usage: uv run python training.py [cnn|cnn_augmented|resnet50|resnet50_e2e]

import sys
import time

import keras

from common import prepare_data, load_datasets, print_metrics, log_results_csv
from model_cnn import build_cnn, build_cnn_augmented
from model_resnet50 import get_resnet50_base, extract_features, build_resnet50_head, build_resnet50_e2e

keras.utils.set_random_seed(42)

# --- Configuration ---

MODEL = "cnn"
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
BATCH_SIZE = 32

if len(sys.argv) > 1:
    MODEL = sys.argv[1]

# --- Data ---

base_dir = prepare_data("./data")
train_dataset, validation_dataset, test_dataset = load_datasets(base_dir, batch_size=BATCH_SIZE)

# --- Build model and train ---

t0 = time.time()

if MODEL == "cnn":
    model = build_cnn()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    history = model.fit(
        train_dataset, epochs=NUM_EPOCHS, validation_data=validation_dataset
    )

elif MODEL == "cnn_augmented":
    model = build_cnn_augmented()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    history = model.fit(
        train_dataset, epochs=NUM_EPOCHS, validation_data=validation_dataset
    )

elif MODEL == "resnet50":
    resnet50_base = get_resnet50_base()

    print("Extracting features with ResNet50...")
    train_features, train_labels = extract_features(train_dataset, resnet50_base)
    val_features, val_labels = extract_features(validation_dataset, resnet50_base)
    test_features, test_labels = extract_features(test_dataset, resnet50_base)

    model = build_resnet50_head()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    history = model.fit(
        train_features, train_labels,
        epochs=NUM_EPOCHS,
        validation_data=(val_features, val_labels),
    )

elif MODEL == "resnet50_e2e":
    model = build_resnet50_e2e()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    history = model.fit(
        train_dataset, epochs=NUM_EPOCHS, validation_data=validation_dataset
    )

else:
    print(f"Unknown model: {MODEL}. Choose from: cnn, cnn_augmented, resnet50, resnet50_e2e")
    sys.exit(1)

training_seconds = round(time.time() - t0, 1)
num_params = model.count_params()

# --- Evaluation ---

if MODEL == "resnet50":
    test_loss, test_accuracy = model.evaluate(test_features, test_labels)
else:
    test_loss, test_accuracy = model.evaluate(test_dataset)

print_metrics(history, test_loss, test_accuracy, num_params, training_seconds)

h = history.history
log_results_csv(
    test_accuracy, test_loss,
    h["val_accuracy"][-1], h["val_loss"][-1],
    num_params, training_seconds,
    description=MODEL,
)
