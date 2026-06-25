# Deep Learning

Projects for the [MIT Hands On Deep Learning course](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/). Each project starts with a baseline model based on the notebook covered in the class, then uses an autonomous experiment harness ([autoresearch](#autoresearch)) to iteratively tune architecture, hyperparameters, and regularization. The other extension is that we also have PyTorch implementations for each of the projects in addition to Keras.

## Results

| Assignment | Dataset | Task | Baseline | Best | Experiments |
|---|---|---|---|---|---|
| [1_heart-disease](1_heart-disease/) | UCI Heart Disease (303 samples, 13 features) | Binary classification (Keras) | 90.16% | **93.44%** | 10 |
| | | Binary classification (PyTorch) | 86.89% | **91.80%** | 16 |
| [2_fashion-mnist](2_fashion-mnist/) | Fashion-MNIST (70k images, 28x28) | 10-class MLP | 73.93% | **90.34%** | 20 |
| [3_fashion_mnist_cnn](3_fashion_mnist_cnn/) | Fashion-MNIST (70k images, 28x28) | 10-class CNN | 87.01% | **94.87%** | 8 |
| [4_handbag_shoe](4_handbag_shoe/) | Handbags vs Shoes (224x224 RGB) | Binary CNN | 76.92% | **82.05%** | 6 |
| | | Binary CNN + augmentation | 61.54% | **82.05%** | 4 |
| | | Binary ResNet50 (feature extract) | 100% | **100%** | — |
| | | Binary ResNet50 (end-to-end) | 100% | **100%** | — |
| [5_music_genre_classification](5_music_genre_classification/) | Song Lyrics (3 genres, BoW) | Unigram Dense | 73.06% | **74.08%** | — |
| | | Bigram Dense | 74.25% | **74.25%** | — |
| [6_standalone_word_embeddings](6_standalone_word_embeddings/) | Song Lyrics (3 genres, embeddings) | GloVe Frozen | 61.20% | **71.29%** | 22 |
| | | GloVe Fine-tune | 70.55% | **72.94%** | 8 |
| | | Custom Embedding | 69.25% | **72.70%** | 7 |

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
uv run python training.py           # Keras (default)
uv run python training.py --torch   # PyTorch
```

For multi-model projects, pass the model name:

```bash
cd 4_handbag_shoe
uv run python training.py cnn
uv run python training.py cnn --torch   # PyTorch variant
```

Or use the helper script to commit and run in one step:

```bash
../run_experiment.sh "description of change"
../run_experiment.sh "widen conv layers" cnn         # multi-model
../run_experiment.sh "try AdamW" --torch              # PyTorch
../run_experiment.sh "try AdamW" cnn --torch          # multi-model + PyTorch
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
/autoresearch 1_heart-disease              # single-model project
/autoresearch 4_handbag_shoe cnn           # multi-model project
/autoresearch 1_heart-disease --torch      # PyTorch variant
/autoresearch 4_handbag_shoe cnn --torch   # multi-model + PyTorch
```

For multi-model projects, autoresearch tunes one model at a time — modifying both `training.py` (hyperparameters) and the model's builder file (architecture). Each folder's `CLAUDE.md` maps model names to their files.

- If `results.csv` has existing data, autoresearch **continues** from where it left off — it reviews past experiments and proposes new ideas informed by what worked and what didn't.
- If no history exists, it **starts fresh** by running the baseline first.

### Visualizing results

```bash
uv run python plot_results.py 1_heart-disease
```

This generates an `experiments.png` chart showing all experiments, which ones were kept, and the running best accuracy.

For multi-model projects, generate per-model charts:

```bash
uv run python plot_results_by_model.py 6_standalone_word_embeddings
```

## PyTorch support

Each project can optionally support PyTorch alongside Keras via the `--torch` flag. The pattern:

- `model_keras.py` — Keras model builder (architecture only)
- `model_torch.py` — PyTorch model architecture (`nn.Module` subclass)
- `train_torch.py` — PyTorch training loop and evaluation utilities
- `common.py` — Framework-agnostic code (data loading, metrics, CSV logging)
- `training.py` — Entry point that dispatches to the appropriate framework

Currently supported: `1_heart-disease/`. Other training projects (2-6) are Keras-only for now — see [PYTORCH_PLAN.md](PYTORCH_PLAN.md) for the rollout plan.

## Project structure

### Training projects (with autoresearch support)

```
├── new_project.sh             # Scaffold a new project
├── run_experiment.sh          # Commit + run helper
├── plot_results.py            # Results visualization (all experiments)
├── plot_results_by_model.py   # Per-model charts (multi-model projects)
├── pyproject.toml             # Shared dependencies
├── PYTORCH_PLAN.md            # Plan for adding PyTorch support to all projects
├── 1_heart-disease/           # Binary classifier (Keras + PyTorch)
│   ├── CLAUDE.md
│   ├── common.py              # Shared: data loading, metrics, logging
│   ├── model_keras.py         # Keras model builder
│   ├── model_torch.py         # PyTorch model architecture
│   ├── train_torch.py         # PyTorch training loop and evaluation
│   ├── training.py            # Entry point (--torch flag selects framework)
│   ├── course_notebook.ipynb  # Keras course notebook
│   └── course_notebook_torch.ipynb  # PyTorch course notebook
├── 2_fashion-mnist/           # 10-class MLP (Keras only)
│   ├── CLAUDE.md
│   ├── training.py
│   └── course_notebook.ipynb
├── 3_fashion_mnist_cnn/       # 10-class CNN (Keras only)
│   ├── CLAUDE.md
│   └── training.py
├── 4_handbag_shoe/            # Binary CNN + ResNet50 transfer learning (Keras only)
│   ├── CLAUDE.md
│   ├── common.py
│   ├── model_cnn.py
│   ├── model_resnet50.py
│   ├── training.py
│   ├── course.ipynb
│   ├── course_notebook_cnn.ipynb
│   └── course_notebook_transfer_learning.ipynb
├── 5_music_genre_classification/  # BoW text classifier (Keras only)
│   ├── CLAUDE.md
│   ├── common.py
│   ├── model_dense.py
│   ├── training.py
│   └── course_notebook.ipynb
└── 6_standalone_word_embeddings/  # Embedding text classifier (Keras only)
    ├── CLAUDE.md
    ├── common.py
    ├── model_embedding.py
    ├── training.py
    ├── course_notebook.ipynb
    └── embeddings.ipynb
```

### Course notebooks (no training.py)

```
├── 7_transformers/            # Custom transformer for NER/slot-filling (Keras)
│   └── course_notebook.ipynb
├── 8_huggingface_pretrained/  # Pre-trained model inference (HuggingFace/PyTorch)
│   └── course_notebook.ipynb
├── 10_rag/                    # Retrieval-Augmented Generation (OpenAI API)
│   └── course_notebook.ipynb
├── 10.5_lora_finetuning/      # LoRA fine-tuning Gemma 2B (Keras 3)
│   └── course_notebook.ipynb
└── 11_diffusion/              # Stable Diffusion + CV models (PyTorch)
    ├── course_notebook_stable_diffusion.ipynb
    ├── course_notebook_noisy_images.ipynb
    └── course_notebook_cv_models.ipynb
```

### Homework assignments

```
├── hw1/                       # FER emotion classification (CNN + VGG19)
│   ├── assignment.ipynb
│   ├── solutions.ipynb
│   └── fer_cnn.ipynb
└── hw2/                       # 20 Newsgroups text classification (BoW + GloVe + BERT)
    ├── assignment.ipynb
    └── solutions.ipynb
```
