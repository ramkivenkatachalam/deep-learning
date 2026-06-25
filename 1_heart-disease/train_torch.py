# PyTorch training and evaluation utilities

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def train_model(model, train_X, train_Y, num_epochs=500, batch_size=32,
                learning_rate=0.0005, validation_split=0.2, verbose=True):
    """Train the model; returns a history dict with loss/accuracy keys."""
    # Split off validation set
    n = len(train_X)
    n_val = int(n * validation_split)
    indices = np.random.RandomState(41).permutation(n)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    tX = torch.tensor(train_X[train_idx], dtype=torch.float32)
    tY = torch.tensor(train_Y[train_idx], dtype=torch.float32)
    vX = torch.tensor(train_X[val_idx], dtype=torch.float32)
    vY = torch.tensor(train_Y[val_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(tX, tY), batch_size=batch_size, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

        for xb, yb in train_loader:
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
        with torch.no_grad():
            val_preds = model(vX)
            val_loss = criterion(val_preds, vY).item()
            val_acc = ((val_preds > 0.5).float() == vY).float().mean().item()

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - "
                  f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

    return history


def evaluate_model(model, test_X, test_Y):
    """Returns (loss, accuracy) on test data."""
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(test_X, dtype=torch.float32)
        yt = torch.tensor(test_Y, dtype=torch.float32)
        preds = model(xt)
        loss = nn.BCELoss()(preds, yt).item()
        acc = ((preds > 0.5).float() == yt).float().mean().item()
    return loss, acc
