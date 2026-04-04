#!/bin/bash
# Usage: ./new_project.sh "project-name"
# Creates a new numbered project folder with template files.

set -e

if [ -z "$1" ]; then
    echo "Usage: ./new_project.sh \"project-name\""
    echo "Example: ./new_project.sh \"mnist-digits\""
    exit 1
fi

NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Find next number
LAST=$(ls -d "$SCRIPT_DIR"/[0-9]*/ 2>/dev/null | sed 's|.*/\([0-9]*\)_.*|\1|' | sort -n | tail -1)
NEXT=$(( ${LAST:-0} + 1 ))

FOLDER="${SCRIPT_DIR}/${NEXT}_${NAME}"

if [ -d "$FOLDER" ]; then
    echo "Error: $FOLDER already exists"
    exit 1
fi

mkdir -p "$FOLDER"

# Create CLAUDE.md template
cat > "$FOLDER/CLAUDE.md" << 'TEMPLATE'
# PROJECT_TITLE

TODO: Add course/project reference here.

## Dataset
- TODO: Describe dataset (samples, features, classes)
- Train/test split: TODO
- Source: TODO

## Current baseline
- Architecture: TODO
- Optimizer: TODO
- Epochs: TODO
- test_accuracy: TBD (run baseline to establish)

## Ideas to try
- TODO: Add experiment ideas
TEMPLATE

sed -i '' "s/PROJECT_TITLE/${NAME}/g" "$FOLDER/CLAUDE.md"

# Create training.py template
cat > "$FOLDER/training.py" << 'TEMPLATE'
# TODO: Add project description
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
# TODO: Load and preprocess your dataset
# train_X, train_Y = ...
# test_X, test_Y = ...

# --- Model architecture ---
# TODO: Define your baseline model
# model = keras.Model(...)

model.summary()
num_params = model.count_params()

# --- Training ---
# TODO: Compile and train
# model.compile(...)

num_epochs = 20  # TODO: adjust
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
TEMPLATE

echo "Created ${NEXT}_${NAME}/"
echo "  ${NEXT}_${NAME}/CLAUDE.md     — fill in dataset and baseline info"
echo "  ${NEXT}_${NAME}/training.py   — fill in TODOs with your baseline model"
echo ""
echo "Next steps:"
echo "  1. Edit training.py with your data loading and baseline model"
echo "  2. Run: cd ${NEXT}_${NAME} && uv run python training.py"
echo "  3. Tune: /autoresearch ${NEXT}_${NAME}"
