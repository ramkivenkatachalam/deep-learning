# Fashion-MNIST CNN classifier
# Based on MIT 15.773 Hands-On Deep Learning (Spring 2024)
# Original colab: https://colab.research.google.com/drive/1Ll2iV7bXLqGqbrLQDHzsRqetWNks_M6q
#
# Dataset: Fashion-MNIST (60k train / 10k test, 28x28 grayscale, 10 classes)
# This is the only file modified during autoresearch experiments.

import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import subprocess
import csv
import os

keras.utils.set_random_seed(41)

# --- Data loading and preprocessing ---

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

# Add channel dimension for Conv2D
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# --- Model architecture ---

input = keras.Input(shape=(28, 28, 1))
h = keras.layers.Conv2D(32, kernel_size=(2, 2), activation="relu")(input)
h = keras.layers.MaxPooling2D(pool_size=(2, 2))(h)
h = keras.layers.Conv2D(32, kernel_size=(2, 2), activation="relu")(h)
h = keras.layers.MaxPooling2D(pool_size=(2, 2))(h)
h = keras.layers.Flatten()(h)
h = keras.layers.Dense(256, activation="relu")(h)
output = keras.layers.Dense(10, activation="softmax")(h)
model = keras.Model(input, output)

model.summary()

num_params = model.count_params()

# --- Training ---

num_epochs = 10
batch_size = 64

optimizer = keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

t0 = time.time()
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size,
                    validation_split=0.1, verbose=True)
training_seconds = round(time.time() - t0, 1)

history_dict = history.history

# --- Evaluation and structured output ---
# Stats block is parsed by experiment tooling (grep "^test_accuracy:" run.log)

test_loss, test_accuracy = model.evaluate(x_test, y_test)

train_loss = history_dict["loss"][-1]
train_accuracy = history_dict["accuracy"][-1]
val_loss = history_dict["val_loss"][-1]
val_accuracy = history_dict["val_accuracy"][-1]

print("---")
print(f"test_accuracy:    {test_accuracy:.4f}")
print(f"test_loss:        {test_loss:.4f}")
print(f"val_accuracy:     {val_accuracy:.4f}")
print(f"val_loss:         {val_loss:.4f}")
print(f"train_accuracy:   {train_accuracy:.4f}")
print(f"train_loss:       {train_loss:.4f}")
print(f"num_params:       {num_params}")
print(f"num_epochs:       {num_epochs}")
print(f"training_seconds: {training_seconds}")

# --- Auto-log results to CSV ---

try:
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    commit = "uncommitted"

results_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")
write_header = not os.path.exists(results_file)

with open(results_file, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["commit", "test_accuracy", "test_loss", "val_accuracy", "val_loss", "num_params", "training_seconds", "status", "description"])
    writer.writerow([commit, f"{test_accuracy:.4f}", f"{test_loss:.4f}", f"{val_accuracy:.4f}", f"{val_loss:.4f}", num_params, training_seconds, "pending", ""])
