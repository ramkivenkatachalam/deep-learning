# Shared utilities for handbag-shoe classifier
# Data download/split, dataset loading, plotting, metrics/logging

import os
import shutil
import pathlib
import urllib.request
import zipfile
import csv
import subprocess

import keras
import numpy as np
import matplotlib.pyplot as plt


DATA_URL = "https://www.dropbox.com/s/w07liww46kgxo1m/handbags-shoes.zip?dl=1"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def prepare_data(data_dir="./data"):
    """Download zip, unzip, split into train(50)/val(25)/test(remaining) per class.

    Idempotent: skips if train/validation/test dirs already exist.
    """
    base_dir = pathlib.Path(data_dir) / "handbags-shoes"

    if (base_dir / "train").exists() and (base_dir / "validation").exists() and (base_dir / "test").exists():
        print(f"Data already prepared at {base_dir}")
        return base_dir

    zip_path = pathlib.Path(data_dir) / "handbags-shoes.zip"
    os.makedirs(data_dir, exist_ok=True)

    if not zip_path.exists():
        print("Downloading handbags-shoes dataset...")
        urllib.request.urlretrieve(DATA_URL, zip_path)

    if not base_dir.exists():
        print("Unzipping dataset...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)

    for category in ("handbags", "shoes"):
        fnames = sorted(os.listdir(base_dir / category))

        dir = base_dir / "train" / category
        os.makedirs(dir, exist_ok=True)
        for fname in fnames[:50]:
            shutil.copyfile(src=base_dir / category / fname, dst=dir / fname)

        dir = base_dir / "validation" / category
        os.makedirs(dir, exist_ok=True)
        for fname in fnames[50:75]:
            shutil.copyfile(src=base_dir / category / fname, dst=dir / fname)

        dir = base_dir / "test" / category
        os.makedirs(dir, exist_ok=True)
        for fname in fnames[75:]:
            shutil.copyfile(src=base_dir / category / fname, dst=dir / fname)

    print(f"Data prepared at {base_dir}")
    return base_dir


def load_datasets(base_dir, batch_size=BATCH_SIZE):
    """Load train/val/test datasets with integer labels (no one-hot)."""
    base_dir = pathlib.Path(base_dir)

    train_dataset = keras.utils.image_dataset_from_directory(
        directory=base_dir / "train",
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
    )
    validation_dataset = keras.utils.image_dataset_from_directory(
        directory=base_dir / "validation",
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
    )
    test_dataset = keras.utils.image_dataset_from_directory(
        directory=base_dir / "test",
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
    )
    return train_dataset, validation_dataset, test_dataset


def plot_loss_curves(history):
    plt.clf()
    history_dict = history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "bo", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_acc_curves(history):
    plt.clf()
    history_dict = history.history
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def print_metrics(history, test_loss, test_accuracy, num_params, training_seconds):
    """Print structured metrics matching repo conventions."""
    h = history.history
    print("---")
    print(f"test_accuracy:    {test_accuracy:.4f}")
    print(f"test_loss:        {test_loss:.4f}")
    print(f"val_accuracy:     {h['val_accuracy'][-1]:.4f}")
    print(f"val_loss:         {h['val_loss'][-1]:.4f}")
    print(f"train_accuracy:   {h['accuracy'][-1]:.4f}")
    print(f"train_loss:       {h['loss'][-1]:.4f}")
    print(f"num_params:       {num_params}")
    print(f"num_epochs:       {len(h['loss'])}")
    print(f"training_seconds: {training_seconds}")


def log_results_csv(test_accuracy, test_loss, val_accuracy, val_loss, num_params, training_seconds, description=""):
    """Append results to results.csv in the script directory."""
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
        writer.writerow([commit, f"{test_accuracy:.4f}", f"{test_loss:.4f}", f"{val_accuracy:.4f}", f"{val_loss:.4f}", num_params, training_seconds, "pending", description])
