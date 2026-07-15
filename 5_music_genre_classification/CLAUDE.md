# Music Genre Classification

Based on MIT 15.773 Hands-On Deep Learning.

## Dataset
- Song lyrics → 3 genre classes
- Dropbox CSV download (train/val/test)
- Input: multi-hot bag-of-words encoding of lyrics

## Models

| Model | Keras builder | PyTorch builder | Description |
|-------|--------------|-----------------|-------------|
| `unigram` | `model_dense.py` | `model_dense_torch.py` | Unigram multi-hot (5k tokens) → Dense(16, relu) → Dropout(0.3) → Dense(3) |
| `bigram` | `model_dense.py` | `model_dense_torch.py` | Bigram multi-hot (20k tokens) → Dense(8, relu) → Dropout(0.5) → Dense(3) |

## File structure
- `training.py` — main entry point, framework dispatch
- `common.py` — data loading, labels, text vectorization (framework-agnostic)
- `model_dense.py` — Keras model builders
- `model_dense_torch.py` — PyTorch model architectures
- `train_torch.py` — PyTorch training loop and evaluation

## Usage
```bash
uv run python training.py unigram           # Keras (default)
uv run python training.py bigram            # Keras
uv run python training.py unigram --torch   # PyTorch
uv run python training.py bigram --torch    # PyTorch
```

## Ideas to try
- Wider hidden layers (16, 32, 64)
- Deeper networks (add second hidden layer)
- Different dropout rates
- More epochs
- Learning rate tuning
- Different max_tokens values
- TF-IDF output mode instead of multi_hot
- Trigrams (ngrams=3)
- Embedding-based models
