# Standalone Word Embeddings

Based on MIT 15.773 Hands-On Deep Learning.

## Dataset
- Song lyrics → 3 genre classes (same as project 5)
- Dropbox CSV download (train/val/test)
- Input: integer sequences (TextVectorization int mode, 5000 tokens, seq len 300)

## Models

| Model | Builder file | Description |
|-------|-------------|-------------|
| `glove_frozen` | `model_embedding.py` | GloVe 100d frozen → GlobalAvgPool → Dense(8, relu) → Dense(3, softmax) |
| `glove_finetune` | `model_embedding.py` | GloVe 100d trainable → GlobalAvgPool → Dense(8, relu) → Dense(3, softmax) |
| `custom` | `model_embedding.py` | Random-init 100d trainable → GlobalAvgPool → Dense(8, relu) → Dense(3, softmax) |

## Usage
```bash
uv run python training.py glove_frozen
uv run python training.py glove_finetune
uv run python training.py custom
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
