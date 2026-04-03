# Fashion-MNIST 10-class image classifier
# Based on MIT 15.773 Hands-On Deep Learning (Spring 2024)
# Original colab: https://colab.research.google.com/drive/14Vv8YgcflIVF_cY9YFsx7Ae774nCuvLn
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

labels = ["T-shirt/top",
          "Trouser",
          "Pullover",
          "Dress",
          "Coat",
          "Sandal",
          "Shirt",
          "Sneaker",
          "Bag",
          "Ankle boot"]

x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for Conv2D: (N, 28, 28) -> (N, 28, 28, 1)
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# Manual train/val split for proper augmentation
val_split = 0.2
num_val = int(len(x_train) * val_split)
x_val, y_val = x_train[:num_val], y_train[:num_val]
x_tr, y_tr = x_train[num_val:], y_train[num_val:]

# --- Model architecture ---

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
model = keras.Model(input, output)

model.summary()

num_params = model.count_params()

# --- Training ---

num_epochs = 60
batch_size = 128
steps_per_epoch = len(x_tr) // batch_size
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.002, decay_steps=num_epochs * steps_per_epoch, warmup_target=0.002, warmup_steps=3 * steps_per_epoch
)
optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=5e-4)
# Convert to one-hot for label smoothing
y_tr_oh = keras.utils.to_categorical(y_tr, 10)
y_val_oh = keras.utils.to_categorical(y_val, 10)
y_test_oh = keras.utils.to_categorical(y_test, 10)

loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Data augmentation via tf.data (train only)
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.pad(image, [[2, 2], [2, 2], [0, 0]], mode='REFLECT')
    image = tf.image.random_crop(image, size=[28, 28, 1])
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((x_tr, y_tr_oh))
train_ds = train_ds.shuffle(len(x_tr)).map(augment, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val_oh)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

t0 = time.time()
history = model.fit(train_ds, epochs=num_epochs, validation_data=val_ds, verbose=True)
training_seconds = round(time.time() - t0, 1)

history_dict = history.history

# --- Evaluation and structured output ---
# Stats block is parsed by experiment tooling (grep "^test_accuracy:" run.log)

test_loss, test_accuracy = model.evaluate(x_test, y_test_oh)

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
# Appends a row after each run; status starts as "pending",
# updated to "keep" or "discard" by the experiment loop.

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
