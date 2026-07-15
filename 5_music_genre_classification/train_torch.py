# PyTorch training and evaluation utilities for music genre classification

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def train_model(model, x_train, y_train, x_val, y_val,
                num_epochs=10, batch_size=32, learning_rate=1e-3, verbose=True):
    """Train the model; returns a history dict with loss/accuracy keys."""
    tX = torch.tensor(x_train, dtype=torch.float32)
    tY = torch.tensor(y_train, dtype=torch.long)
    vX = torch.tensor(x_val, dtype=torch.float32)
    vY = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(tX, tY), batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(num_epochs):
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

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - "
                  f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

    return history


def evaluate_model(model, x_test, y_test):
    """Returns (loss, accuracy) on test data."""
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x_test, dtype=torch.float32)
        yt = torch.tensor(y_test, dtype=torch.long)
        logits = model(xt)
        loss = nn.CrossEntropyLoss()(logits, yt).item()
        acc = (logits.argmax(1) == yt).float().mean().item()
    return loss, acc
