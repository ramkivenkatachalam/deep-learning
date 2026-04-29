# Music Genre Classification

Based on MIT 15.773 Hands-On Deep Learning.

## Dataset
- Song lyrics → 3 genre classes
- Dropbox CSV download (train/val/test)
- Input: multi-hot bag-of-words encoding of lyrics

## Models

| Model | Builder file | Description |
|-------|-------------|-------------|
| `unigram` | `model_dense.py` | Unigram multi-hot (5k tokens) → Dense(16, relu) → Dropout(0.3) → Dense(3, softmax) |
| `bigram` | `model_dense.py` | Bigram multi-hot (20k tokens) → Dense(8, relu) → Dropout(0.5) → Dense(3, softmax) |

## Usage
```bash
uv run python training.py unigram
uv run python training.py bigram
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
