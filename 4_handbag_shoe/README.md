# Handbag vs Shoe Classifier

Binary image classifier comparing CNNs trained from scratch against ResNet50 transfer learning, based on MIT 15.773 Hands-On Deep Learning.

## Dataset

- **Source**: Custom handbags vs shoes dataset (Dropbox)
- **Images**: 224x224 RGB
- **Split**: 50 train / 25 validation / remaining test per class
- Auto-downloaded and split on first run

## Results

![Model Comparison](experiments.png)

### CNN (`cnn`)

Rescaling(1/255) → 2x [Conv2D(32, 3x3) + MaxPool] → Dense(128, relu) → Dropout(0.5) → Dense(1, sigmoid)

| | Baseline | After tuning |
|--|----------|-------------|
| Test Accuracy | 76.92% | **82.05%** |
| Val Accuracy | 77.55% | 71.43% |
| Params | 101K | 12.5M |

**What helped:** 3x3 kernels (from 2x2), adding Dense(128) hidden layer with Dropout(0.5).
**What didn't help:** more filters (64/128), 3rd conv block, BatchNorm, L2 regularization, lower LR, more epochs.

### CNN + Augmentation (`cnn_augmented`)

RandomFlip + RandomRotation(0.05) + RandomZoom(0.1) → same CNN as above

| | Baseline | After tuning |
|--|----------|-------------|
| Test Accuracy | 61.54% | **82.05%** |
| Val Accuracy | 73.47% | 81.63% |
| Params | 101K | 12.5M |

**What helped:** same architecture improvements as CNN, reducing augmentation intensity (0.1→0.05 rotation, 0.2→0.1 zoom), 40 epochs (from 20).
**What didn't help:** 60 epochs, lower LR, flip-only augmentation.

### ResNet50 Feature Extraction (`resnet50`)

Frozen ResNet50 pre-computes (7,7,2048) features once, then trains a small head: Dense(256) → Dropout(0.5) → Dense(1, sigmoid). Fast because features are cached.

| Metric | Value |
|--------|-------|
| Test Accuracy | **100%** |
| Val Accuracy | 100% |
| Params | 25.7M |
| Training Time | 15s |

### ResNet50 End-to-End (`resnet50_e2e`)

End-to-end model with augmentation layers → frozen ResNet50 → Dense head. Slower per epoch but enables fine-tuning by unfreezing backbone layers.

| Metric | Value |
|--------|-------|
| Test Accuracy | **100%** |
| Val Accuracy | 100% |
| Params | 49.3M |
| Training Time | 18s |

## Observations

- Both ResNet50 variants achieve perfect accuracy — the dataset is small and trivially separable for ImageNet-pretrained features.
- The basic CNN overfits heavily (100% train, 77% test) due to the tiny training set (50 per class). Adding Dropout and a hidden layer helped the most.
- Data augmentation initially hurt because intensity was too high for this small dataset. Reducing rotation/zoom intensity brought it up to match the plain CNN.
- BatchNorm, L2 regularization, and wider/deeper architectures all hurt — the dataset is simply too small to benefit from added capacity.
- Transfer learning is the clear winner for small-dataset image classification.

## Usage

```bash
uv run python training.py cnn
uv run python training.py cnn_augmented
uv run python training.py resnet50
uv run python training.py resnet50_e2e
```

## File structure

```
common.py          — Data download/split, dataset loading, plotting, metrics/logging
model_cnn.py       — build_cnn() and build_cnn_augmented()
model_resnet50.py  — Feature extraction + end-to-end ResNet50 builders
training.py        — Dispatcher: selects model, trains, evaluates, logs to results.csv
plot_models.py     — Generate model comparison chart
course.ipynb       — Original Colab notebook (unchanged)
```
