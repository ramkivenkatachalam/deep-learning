# PyTorch training and evaluation utilities for Fashion-MNIST CNN

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class AugmentedDataset(Dataset):
    """Wraps numpy arrays with optional per-sample augmentation transforms."""
    def __init__(self, x, y, transform=None):
        # x: (N, 1, H, W) float32 tensor
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img, label = self.x[idx], self.y[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def train_model(model, x_train, y_train, num_epochs=80, batch_size=128,
                learning_rate=2e-3, weight_decay=5e-4, validation_split=0.2,
                warmup_epochs=3, augment=True, verbose=True):
    """Train the model; returns a history dict with loss/accuracy keys."""
    # Split off validation set (first 20%, matching Keras ordering)
    n = len(x_train)
    n_val = int(n * validation_split)
    x_val, y_val = x_train[:n_val], y_train[:n_val]
    x_tr, y_tr = x_train[n_val:], y_train[n_val:]

    # Convert NHWC → NCHW
    x_tr = np.transpose(x_tr, (0, 3, 1, 2))
    x_val = np.transpose(x_val, (0, 3, 1, 2))

    # Augmentation: RandomHorizontalFlip + reflect-pad(2) + RandomCrop(28)
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.Pad(2, padding_mode='reflect'),
        T.RandomCrop(28),
    ]) if augment else None

    train_ds = AugmentedDataset(x_tr, y_tr, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    vX = torch.tensor(x_val, dtype=torch.float32)
    vY = torch.tensor(y_val, dtype=torch.long)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(num_epochs):
        # Manual warmup: linear ramp from 0 to target LR over first warmup_epochs
        if epoch < warmup_epochs:
            warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        model.train()
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(xb)
            epoch_correct += (logits.argmax(1) == yb).sum().item()
            epoch_total += len(xb)

        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(vX)
            val_loss = criterion(val_logits, vY).item()
            val_acc = (val_logits.argmax(1) == vY).float().mean().item()

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        # Step scheduler after warmup
        if epoch >= warmup_epochs:
            scheduler.step()

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - "
                  f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

    return history


def evaluate_model(model, x_test, y_test):
    """Returns (loss, accuracy) on test data. Expects NHWC input, converts to NCHW."""
    model.eval()
    with torch.no_grad():
        x = np.transpose(x_test, (0, 3, 1, 2))
        xt = torch.tensor(x, dtype=torch.float32)
        yt = torch.tensor(y_test, dtype=torch.long)
        logits = model(xt)
        loss = nn.CrossEntropyLoss()(logits, yt).item()
        acc = (logits.argmax(1) == yt).float().mean().item()
    return loss, acc
