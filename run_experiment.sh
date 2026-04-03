#!/bin/bash
# Usage: ../run_experiment.sh "commit message"
# Run from inside an assignment folder (e.g. 1_heart-disease/).
# Commits training.py, runs training, and prints key metrics.

set -e

MSG="${1:-experiment}"

git add training.py
git commit -m "$MSG"

uv run python training.py > run.log 2>&1

echo "=== RESULTS ==="
grep "^test_accuracy:\|^test_loss:\|^val_accuracy:\|^val_loss:\|^num_params:\|^training_seconds:" run.log
