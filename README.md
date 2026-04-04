# Deep Learning

Assignments for MIT 15.773 Hands-On Deep Learning (Spring 2024), structured for autonomous experimentation with the autoresearch protocol (`/autoresearch <folder>`).

## Assignments

| Folder | Dataset | Task | Baseline |
|--------|---------|------|----------|
| `1_heart-disease/` | UCI Heart Disease (303 samples, 13 features) | Binary classification | 95.08% |
| `2_fashion-mnist/` | Fashion-MNIST (70k images, 28x28) | 10-class classification | 87.46% |

## Setup

```bash
uv sync
```

## Running an experiment

```bash
cd 1_heart-disease
uv run python training.py
```

Or use the helper script to commit and run in one step:

```bash
cd 1_heart-disease
../run_experiment.sh "description of change"
```

## Visualizing results

```bash
uv run python plot_results.py 1_heart-disease
```

## Project structure

```
├── run_experiment.sh      # Commit + run helper
├── plot_results.py        # Results visualization
├── pyproject.toml         # Shared dependencies
├── 1_heart-disease/
│   ├── CLAUDE.md          # Assignment context and ideas
│   └── training.py        # Training script
└── 2_fashion-mnist/
    ├── CLAUDE.md
    └── training.py
```
