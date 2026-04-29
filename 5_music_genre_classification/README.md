# 5 — Music Genre Classification

Classify song lyrics into 3 music genres using bag-of-words dense networks.

## Dataset
- Song lyrics from Dropbox CSVs (train/val/test split)
- 3 genre classes, one-hot encoded
- Text vectorized as multi-hot bag-of-words (unigram or bigram)

## Models

| Model | Input | Architecture |
|-------|-------|-------------|
| unigram | 5k-token multi-hot | Dense(16) → Dropout(0.3) → softmax(3) |
| bigram | 20k-token bigram multi-hot | Dense(8) → Dropout(0.5) → softmax(3) |

## Results

| Model | Test Acc | Val Acc | Params | Notes |
|-------|----------|---------|--------|-------|
| unigram | 0.7408 | 0.7461 | 80,067 | Dense(16)+Dropout(0.3), 10 epochs |
| bigram | 0.7425 | 0.7504 | 160,035 | Dense(8)+Dropout(0.5), 10 epochs |

## Observations

- Bigram slightly outperforms unigram (~1% test accuracy) despite using 2x params
- Wider hidden layers (64) cause overfitting; Dense(16) with dropout is the sweet spot for unigram
- The bigram model's original Dense(8)+Dropout(0.5) proved hard to beat — wider/deeper variants all performed worse
- Both models show train-test gap (~4-6%), suggesting the text classification task is inherently noisy
