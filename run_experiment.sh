#!/bin/bash
# Usage: ../run_experiment.sh "commit message" [model] [flags...]
# Run from inside an assignment folder (e.g. 1_heart-disease/).
# Commits training.py (and model files if present), runs training, and prints key metrics.
#
# Single-model:  ../run_experiment.sh "add dropout"
# Multi-model:   ../run_experiment.sh "widen conv layers" cnn
# With flags:    ../run_experiment.sh "add dropout" --torch
#                ../run_experiment.sh "widen conv layers" cnn --torch

set -e

MSG="${1:-experiment}"
shift
ARGS=("$@")

# Stage training.py and any model files that have changes
git add training.py
git diff --cached --quiet model_*.py 2>/dev/null || git add model_*.py 2>/dev/null
git commit -m "$MSG"

uv run python training.py "${ARGS[@]}" > run.log 2>&1

echo "=== RESULTS ==="
grep "^test_accuracy:\|^test_loss:\|^val_accuracy:\|^val_loss:\|^num_params:\|^training_seconds:" run.log
