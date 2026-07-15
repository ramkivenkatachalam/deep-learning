# PyTorch training and evaluation utilities for handbag-shoe classifier
# Supports two data modes: "loader" (DataLoaders) and "numpy" (extracted features)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def train_model(model, train_data, val_data, num_epochs=20, batch_size=32,
                learning_rate=1e-3, data_mode="loader", verbose=True):
    """Train a binary classifier.

    data_mode="loader": train_data/val_data are DataLoaders
    data_mode="numpy": train_data=(x, y), val_data=(x, y) numpy arrays
    Returns a history dict with loss/accuracy keys.
    """
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Build val tensors for numpy mode
    if data_mode == "numpy":
        x_train, y_train = train_data
        x_val, y_val = val_data
        tX = torch.tensor(x_train, dtype=torch.float32)
        tY = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        vX = torch.tensor(x_val, dtype=torch.float32)
        vY = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        train_loader = DataLoader(TensorDataset(tX, tY), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(vX, vY), batch_size=batch_size, shuffle=False)
    else:
        train_loader = train_data
        val_loader = val_data

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

        for xb, yb in train_loader:
            if data_mode == "loader":
                yb = yb.float().unsqueeze(1)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(xb)
            epoch_correct += ((preds > 0.5).float() == yb).sum().item()
            epoch_total += len(xb)

        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total

        # Validation
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                if data_mode == "loader":
                    yb = yb.float().unsqueeze(1)
                preds = model(xb)
                val_loss_sum += criterion(preds, yb).item() * len(xb)
                val_correct += ((preds > 0.5).float() == yb).sum().item()
                val_total += len(xb)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - "
                  f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

    return history


def evaluate_model(model, test_data, data_mode="loader"):
    """Returns (loss, accuracy) on test data."""
    criterion = nn.BCELoss()
    model.eval()

    if data_mode == "numpy":
        x_test, y_test = test_data
        test_loader = DataLoader(
            TensorDataset(
                torch.tensor(x_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
            ),
            batch_size=32, shuffle=False,
        )
    else:
        test_loader = test_data

    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            if data_mode == "loader":
                yb = yb.float().unsqueeze(1)
            preds = model(xb)
            total_loss += criterion(preds, yb).item() * len(xb)
            total_correct += ((preds > 0.5).float() == yb).sum().item()
            total += len(xb)

    return total_loss / total, total_correct / total
