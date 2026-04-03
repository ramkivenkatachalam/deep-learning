# Heart disease binary classifier
# Based on MIT 15.773 Hands-On Deep Learning (Spring 2024)
# https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/
# Original colab: https://colab.research.google.com/drive/1flLafeFpy8JjLN4H_ertcs5wJE3--TdQ
#
# Dataset: UCI Heart Disease (303 samples, 13 features, binary target)
# This is the only file modified during autoresearch experiments.

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import time
import subprocess
import csv
import os

keras.utils.set_random_seed(41)

# --- Data loading and preprocessing ---

df = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/heart.csv')

categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'ca', 'thal']
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
target_col = 'target'

# One-hot encode categorical features (expands from 13 to 21 input features)
df = pd.get_dummies(df, columns=categorical_cols)

# 80/20 train/test split (fixed seed for reproducibility)
test_df = df.sample(frac=0.2, random_state=41)
train_df = df.drop(test_df.index)

# Standardize numerical features using training set statistics
means = train_df[numerical_cols].mean()
stds = train_df[numerical_cols].std()

train_df[numerical_cols] = (train_df[numerical_cols] - means) / stds
test_df[numerical_cols] = (test_df[numerical_cols] - means) / stds

feature_cols = [c for c in train_df.columns if c != target_col]

train_X = train_df[feature_cols].to_numpy().astype(float)
train_Y = train_df[target_col].to_numpy().astype(float)

test_X = test_df[feature_cols].to_numpy().astype(float)
test_Y = test_df[target_col].to_numpy().astype(float)

# --- Model architecture ---
# Best config found via autoresearch: Dense(16)+Dropout(0.3)
# Baseline was Dense(8) with no dropout

ncols = train_X.shape[1]
input = keras.Input(shape=(ncols, ))
h = keras.layers.Dense(16, activation='relu', name="Hidden")(input)
h = keras.layers.Dropout(0.3)(h)
output = keras.layers.Dense(1, activation='sigmoid', name="Output")(h)
model = keras.Model(input, output)

model.summary()

num_params = model.count_params()

# --- Training ---
# LR=0.0005 (lower than default 0.001) found to improve generalization

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

num_epochs = 500
t0 = time.time()
history = model.fit(train_X, train_Y, epochs=num_epochs, verbose=True, validation_split=0.2, batch_size=32)
training_seconds = round(time.time() - t0, 1)

history_dict = history.history

# --- Evaluation and structured output ---
# Stats block is parsed by experiment tooling (grep "^test_accuracy:" run.log)

test_loss, test_accuracy = model.evaluate(test_X, test_Y)

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
