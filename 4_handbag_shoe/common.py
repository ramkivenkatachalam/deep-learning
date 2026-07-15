# Shared utilities for handbag-shoe classifier
# Data download/split, dataset loading, plotting, metrics/logging

import os
import sys
import shutil
import pathlib
import urllib.request
import zipfile

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from shared import print_metrics, log_results_csv


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
    """Load train/val/test Keras datasets with integer labels (no one-hot)."""
    import keras
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


def load_datasets_torch(base_dir, batch_size=BATCH_SIZE, augment=False):
    """Load train/val/test PyTorch DataLoaders. Images scaled to [0,1]."""
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    base_dir = pathlib.Path(base_dir)

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(18),
            transforms.RandomAffine(0, scale=(0.9, 1.1)),
            transforms.ToTensor(),  # [0,255] → [0,1]
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])

    eval_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(str(base_dir / "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(base_dir / "validation"), transform=eval_transform)
    test_dataset = datasets.ImageFolder(str(base_dir / "test"), transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_datasets_torch_resnet(base_dir, batch_size=BATCH_SIZE, augment=False):
    """Load PyTorch DataLoaders with ResNet50 normalization."""
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    base_dir = pathlib.Path(base_dir)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(18),
            transforms.RandomAffine(0, scale=(0.9, 1.1)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            normalize,
        ])

    eval_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(str(base_dir / "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(base_dir / "validation"), transform=eval_transform)
    test_dataset = datasets.ImageFolder(str(base_dir / "test"), transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


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


