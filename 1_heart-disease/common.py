# Framework-agnostic utilities for heart disease classifier

import numpy as np
import pandas as pd
import subprocess
import csv
import os


def load_data(random_state=41):
    """Load UCI Heart Disease dataset, preprocess, and split into train/test numpy arrays."""
    df = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/heart.csv')

    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'ca', 'thal']
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    target_col = 'target'

    # One-hot encode categorical features (expands from 13 to 21 input features)
    df = pd.get_dummies(df, columns=categorical_cols)

    # 80/20 train/test split (fixed seed for reproducibility)
    test_df = df.sample(frac=0.2, random_state=random_state)
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

    return train_X, train_Y, test_X, test_Y


def print_metrics(test_accuracy, test_loss, val_accuracy, val_loss,
                  train_accuracy, train_loss, num_params, num_epochs,
                  training_seconds):
    """Print structured metrics block (parsed by experiment tooling)."""
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


def log_results_csv(test_accuracy, test_loss, val_accuracy, val_loss,
                    num_params, training_seconds, framework="keras", model="default"):
    """Append a row to results.csv with status='pending'."""
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
            writer.writerow(["commit", "test_accuracy", "test_loss", "val_accuracy", "val_loss",
                             "num_params", "training_seconds", "status", "description", "model", "framework"])
        writer.writerow([commit, f"{test_accuracy:.4f}", f"{test_loss:.4f}",
                         f"{val_accuracy:.4f}", f"{val_loss:.4f}",
                         num_params, training_seconds, "pending", "", model, framework])
