# Deep Learning Class Assignments

Each assignment lives in a numbered folder with a `training.py` file:
- `1_heart-disease/` — UCI Heart Disease binary classifier
- `2_fashion-mnist/` — Fashion-MNIST 10-class image classifier

## Autoresearch Protocol

Run `/autoresearch <folder>` (e.g. `/autoresearch 1_heart-disease`) to start autonomous experiments. Key points:
- `cd` into the assignment folder before starting
- Only modify `training.py` in that folder
- Run with `uv run python training.py`
- Use `../run_experiment.sh "description"` to commit + run + show results

## Shared files (root level)
- `run_experiment.sh` — commit + run helper
- `plot_results.py` — visualize results.csv from any assignment folder
- `pyproject.toml` — shared Python dependencies
