# Shared utilities for music genre classifier
# Data download, label prep, metrics printing, CSV logging

import os
import csv
import subprocess

import pandas as pd


TRAIN_URL = "https://www.dropbox.com/scl/fi/ito6bnl2yaf1uw0uqibzf/lyric_genre_train.csv?rlkey=04dkn5un2djza8x0bdmfnlw3u&st=y47qh8i4&dl=1"
VAL_URL = "https://www.dropbox.com/scl/fi/xmywjzqsaa8n5sn1bs0t9/lyric_genre_val.csv?rlkey=hggbeo0s1iaxjpa6z80429xl9&st=6i7d8eau&dl=1"
TEST_URL = "https://www.dropbox.com/scl/fi/fnocl69w9ojs9s5zb0xvf/lyric_genre_test.csv?rlkey=z4hjopw7vaihoh948cbb5mvdp&st=xwond7dp&dl=1"


def load_data(data_dir="./data"):
    """Download train/val/test CSVs and return DataFrames.

    Caches locally to data_dir to avoid re-downloading.
    """
    os.makedirs(data_dir, exist_ok=True)

    urls = {"train": TRAIN_URL, "val": VAL_URL, "test": TEST_URL}
    dfs = {}

    for split, url in urls.items():
        path = os.path.join(data_dir, f"lyric_genre_{split}.csv")
        if os.path.exists(path):
            dfs[split] = pd.read_csv(path, index_col=0)
        else:
            print(f"Downloading {split} data...")
            df = pd.read_csv(url, index_col=0)
            df.to_csv(path)
            dfs[split] = df

    print(f"Train: {dfs['train'].shape[0]}  Val: {dfs['val'].shape[0]}  Test: {dfs['test'].shape[0]}")
    return dfs["train"], dfs["val"], dfs["test"]


def prepare_labels(df):
    """Convert Genre column to one-hot numpy array."""
    return pd.get_dummies(df.Genre).to_numpy(dtype="uint8")


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
