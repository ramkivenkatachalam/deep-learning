# Handbag vs Shoe Classifier

Binary image classifier comparing CNNs trained from scratch against ResNet50 transfer learning, based on MIT 15.773 Hands-On Deep Learning.

## Dataset

- **Source**: Custom handbags vs shoes dataset (Dropbox)
- **Images**: 224x224 RGB
- **Split**: 50 train / 25 validation / remaining test per class
- Auto-downloaded and split on first run

## Models

| Model | Test Acc | Val Acc | Params | Time |
|-------|----------|---------|--------|------|
| `cnn` | 0.7692 | 0.7755 | 101K | 5.9s |
| `cnn_augmented` | 0.6154 | 0.7347 | 101K | 5.0s |
| `resnet50` | 1.0000 | 1.0000 | 25.7M | 15s |
| `resnet50_e2e` | 1.0000 | 1.0000 | 49.3M | 18s |

### Model descriptions

- **cnn** — Basic CNN: Rescaling(1/255) → 2x [Conv2D(32, 2x2) + MaxPool] → Dense(1, sigmoid)
- **cnn_augmented** — Same CNN with RandomFlip/RandomRotation/RandomZoom prepended
- **resnet50** — Frozen ResNet50 feature extraction: pre-computes (7,7,2048) features, trains a small head (Dense(256) → Dropout → Dense(1)). Fast because features are cached once.
- **resnet50_e2e** — End-to-end ResNet50 with augmentation layers. Slower per epoch but enables fine-tuning by unfreezing backbone layers.

## Usage

```bash
uv run python training.py cnn
uv run python training.py cnn_augmented
uv run python training.py resnet50
uv run python training.py resnet50_e2e
```

## Observations

- Both ResNet50 variants achieve perfect accuracy — the dataset is small and trivially separable for ImageNet-pretrained features.
- The basic CNN overfits heavily (100% train accuracy, 77% test) due to the tiny training set (50 per class).
- Data augmentation hurts the basic CNN here — too few parameters and too little data to benefit from the added variation.
- Transfer learning is the clear winner for small-dataset image classification.

## File structure

```
common.py          — Data download/split, dataset loading, plotting, metrics/logging
model_cnn.py       — build_cnn() and build_cnn_augmented()
model_resnet50.py  — Feature extraction + end-to-end ResNet50 builders
training.py        — Dispatcher: selects model, trains, evaluates, logs to results.csv
course.ipynb       — Original Colab notebook (unchanged)
```
