# Standalone Word Embeddings

Based on MIT 15.773 Hands-On Deep Learning.

## Dataset
- Song lyrics → 3 genre classes (same as project 5)
- Dropbox CSV download (train/val/test)
- Input: integer sequences (TextVectorization int mode, 5000 tokens, seq len 300)

## Models

| Model | Keras builder | PyTorch builder | Description |
|-------|--------------|-----------------|-------------|
| `glove_frozen` | `model_embedding.py` | `model_embedding_torch.py` | GloVe 300d frozen → concat(AvgPool, MaxPool) → Dense(64) → Dense(3) |
| `glove_finetune` | `model_embedding.py` | `model_embedding_torch.py` | GloVe 300d trainable → AvgPool → Dense(8) → Dense(3) |
| `custom` | `model_embedding.py` | `model_embedding_torch.py` | Random-init 300d trainable → AvgPool → Dense(16) → Dropout(0.3) → Dense(3) |

## File structure
- `training.py` — main entry point, framework dispatch
- `common.py` — data loading, labels, GloVe loading, text vectorization (framework-agnostic)
- `model_embedding.py` — Keras model builders
- `model_embedding_torch.py` — PyTorch model architectures
- `train_torch.py` — PyTorch training loop and evaluation

## Usage
```bash
uv run python training.py glove_frozen              # Keras (default)
uv run python training.py glove_finetune             # Keras
uv run python training.py custom                     # Keras
uv run python training.py glove_frozen --torch       # PyTorch
uv run python training.py glove_finetune --torch     # PyTorch
uv run python training.py custom --torch             # PyTorch
```

## Ideas to try
- Larger embedding dim (200d, 300d GloVe)
- Wider/deeper dense layers
- Add Dropout after GlobalAvgPool or Dense
- Different max_tokens (10000, 20000)
- Different max_length (150, 500)
- Learning rate tuning
- More epochs
- Add a second Dense hidden layer
- Try LSTM/GRU instead of GlobalAvgPool
