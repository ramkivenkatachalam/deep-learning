# Deep Learning

Projects for MIT Deep Learning certification. Each project starts with a baseline model, then uses an autonomous experiment harness ([autoresearch](#autoresearch)) to iteratively tune architecture, hyperparameters, and regularization.

## Results

| Assignment | Dataset | Task | Baseline | Best | Experiments |
|---|---|---|---|---|---|
| [1_heart-disease](1_heart-disease/) | UCI Heart Disease (303 samples, 13 features) | Binary classification | 90.16% | **95.08%** | 44 |
| [2_fashion-mnist](2_fashion-mnist/) | Fashion-MNIST (70k images, 28x28) | 10-class MLP | 73.93% | **90.34%** | 20 |
| [3_fashion_mnist_cnn](3_fashion_mnist_cnn/) | Fashion-MNIST (70k images, 28x28) | 10-class CNN | 87.01% | **94.87%** | 8 |

## Setup

```bash
uv sync
```

## Starting a new project

```bash
./new_project.sh "project-name"
```

This creates a numbered folder (e.g. `3_project-name/`) with template `CLAUDE.md` and `training.py` files. Fill in the TODOs with your dataset, baseline model, and training config, then run `/autoresearch` to start tuning.

## Running a training script

```bash
cd 1_heart-disease
uv run python training.py
```

Or use the helper script to commit and run in one step:

```bash
../run_experiment.sh "description of change"
```

## Autoresearch

Autoresearch is an autonomous experiment protocol that systematically explores model improvements. It works by:

1. **Starting with a baseline** — run the unmodified `training.py` to establish initial metrics
2. **Proposing changes** — modify architecture, optimizer, hyperparameters, regularization, preprocessing, etc.
3. **Evaluating** — each change is committed, run, and scored on test accuracy
4. **Keeping or reverting** — improvements are kept; regressions are reverted via `git reset`
5. **Logging** — every experiment is recorded in `results.csv` with metrics, status, and description

All experiments are tracked in `results.csv` (per assignment folder) and can be visualized with `plot_results.py`.

### Starting autoresearch

Using the Claude Code slash command:

```
/autoresearch 1_heart-disease
```

- If `results.csv` has existing data, autoresearch **continues** from where it left off — it reviews past experiments and proposes new ideas informed by what worked and what didn't.
- If no history exists, it **starts fresh** by running the baseline first.

### Visualizing results

```bash
uv run python plot_results.py 1_heart-disease
```

This generates an `experiments.png` chart showing all experiments, which ones were kept, and the running best accuracy.

## Project structure

```
├── new_project.sh             # Scaffold a new project
├── run_experiment.sh          # Commit + run helper
├── plot_results.py            # Results visualization
├── pyproject.toml             # Shared dependencies
├── 1_heart-disease/
│   ├── CLAUDE.md              # Assignment context and ideas
│   ├── training.py            # Training script
│   ├── results.csv            # Experiment log (gitignored)
│   └── experiments.png        # Results chart
├── 2_fashion-mnist/
│   ├── CLAUDE.md
│   ├── training.py
│   ├── results.csv
│   └── experiments.png
└── 3_fashion_mnist_cnn/
    ├── CLAUDE.md
    ├── training.py
    ├── results.csv
    └── experiments.png
```
