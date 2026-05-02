# Standalone Word Embeddings

Music genre classification from song lyrics using word embedding models. Compares pre-trained GloVe embeddings (frozen vs fine-tuned) against randomly initialized embeddings.

Based on MIT 15.773 Hands-On Deep Learning.

## Models

| Model | Embedding | Trainable | Description |
|-------|-----------|-----------|-------------|
| `glove_frozen` | GloVe 300d | No | Pre-trained embeddings held fixed during training |
| `glove_finetune` | GloVe 300d | Yes | Pre-trained embeddings updated during training |
| `custom` | Random init 300d | Yes | Embeddings learned from scratch |

## Results

| Model | Baseline | Best | Experiments |
|-------|----------|------|-------------|
| `glove_frozen` | 61.20% | **71.29%** | 22 |
| `glove_finetune` | 70.55% | **72.94%** | 8 |
| `custom` | 69.25% | **72.70%** | 7 |

## Setup

GloVe embeddings must be downloaded to this directory:

```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip -q glove.6B.zip glove.6B.300d.txt
```

## Usage

```bash
uv run python training.py glove_frozen
uv run python training.py glove_finetune
uv run python training.py custom
```

## File structure

```
├── CLAUDE.md              # Autoresearch context
├── README.md              # This file
├── common.py              # Data loading, GloVe loading, metrics, logging
├── model_embedding.py     # Embedding model builders
├── training.py            # Dispatcher: selects model via CLI arg
├── embeddings.ipynb       # Original course notebook
└── results.csv            # Experiment log (gitignored)
```
