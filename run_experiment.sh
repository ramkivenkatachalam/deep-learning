#!/bin/bash
# Usage: ../run_experiment.sh "commit message" [model]
# Run from inside an assignment folder (e.g. 1_heart-disease/).
# Commits training.py (and model files if present), runs training, and prints key metrics.
#
# Single-model:  ../run_experiment.sh "add dropout"
# Multi-model:   ../run_experiment.sh "widen conv layers" cnn

set -e

MSG="${1:-experiment}"
MODEL="${2:-}"

# Stage training.py and any model files that have changes
git add training.py
git diff --cached --quiet model_*.py 2>/dev/null || git add model_*.py 2>/dev/null
git commit -m "$MSG"

if [ -n "$MODEL" ]; then
    uv run python training.py "$MODEL" > run.log 2>&1
else
    uv run python training.py > run.log 2>&1
fi

echo "=== RESULTS ==="
grep "^test_accuracy:\|^test_loss:\|^val_accuracy:\|^val_loss:\|^num_params:\|^training_seconds:" run.log
