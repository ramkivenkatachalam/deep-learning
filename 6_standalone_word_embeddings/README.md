# Standalone Word Embeddings

Music genre classification from song lyrics using word embedding models. Compares pre-trained GloVe embeddings (frozen vs fine-tuned) against randomly initialized embeddings.

Based on MIT 15.773 Hands-On Deep Learning.

## Models

| Model | Embedding | Trainable | Description |
|-------|-----------|-----------|-------------|
| `glove_frozen` | GloVe 100d | No | Pre-trained embeddings held fixed during training |
| `glove_finetune` | GloVe 100d | Yes | Pre-trained embeddings updated during training |
| `custom` | Random init 100d | Yes | Embeddings learned from scratch |

All models share: Embedding → GlobalAveragePooling1D → Dense(8, relu) → Dense(3, softmax)

## Results

| Model | Test Accuracy | Val Accuracy |
|-------|--------------|--------------|
| `glove_frozen` | 61.20% | 61.94% |
| `glove_finetune` | 71.24% | 71.67% |
| `custom` | 70.81% | 71.13% |

## Setup

GloVe embeddings must be downloaded to this directory:

```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip -q glove.6B.zip glove.6B.100d.txt
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
