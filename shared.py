# Shared utilities for all deep learning projects

import subprocess
import csv
import os


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
    """Append a row to results.csv with status='pending'.

    Determines the caller's directory to write results.csv alongside the
    project's training.py (not the root shared.py location).
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = "uncommitted"

    # Walk up the call stack to find the caller's directory
    import inspect
    caller_file = inspect.stack()[1].filename
    results_file = os.path.join(os.path.dirname(os.path.abspath(caller_file)), "results.csv")

    write_header = not os.path.exists(results_file)

    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["commit", "test_accuracy", "test_loss", "val_accuracy", "val_loss",
                             "num_params", "training_seconds", "status", "description", "model", "framework"])
        writer.writerow([commit, f"{test_accuracy:.4f}", f"{test_loss:.4f}",
                         f"{val_accuracy:.4f}", f"{val_loss:.4f}",
                         num_params, training_seconds, "pending", "", model, framework])
