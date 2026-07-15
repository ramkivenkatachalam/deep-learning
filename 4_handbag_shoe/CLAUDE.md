# Handbag-Shoe Binary Classifier

Based on MIT 15.773 Hands-On Deep Learning.

## Dataset
- Custom handbags vs shoes dataset (Dropbox download)
- Images: 224x224 RGB
- Split: 50 train / 25 val / remaining test per class
- Binary classification (handbags=0, shoes=1)

## Models

| Model | Keras builder | PyTorch builder | Description |
|-------|--------------|-----------------|-------------|
| `cnn` | `model_cnn.py` | `model_cnn_torch.py` | Basic CNN: 2x Conv2D(32, 3x3)+MaxPool → Dense(128) → Dropout → Dense(1, sigmoid) |
| `cnn_augmented` | `model_cnn.py` | `model_cnn_torch.py` | Same CNN with data augmentation (flip, rotate, zoom/affine) |
| `cnn_residual` | `model_cnn.py` | `model_cnn_torch.py` | 3x ResBlock(32→64→128)+Pool → Dense(128) → Dropout → Dense(1, sigmoid) |
| `resnet50` | `model_resnet50.py` | `model_resnet50_torch.py` | Frozen ResNet50 feature extraction → Dense(256) → Dropout → Dense(1, sigmoid) |
| `resnet50_e2e` | `model_resnet50.py` | `model_resnet50_torch.py` | End-to-end ResNet50 with augmentation |

## File structure
- `training.py` — main entry point, framework dispatch
- `common.py` — data download/split, dataset loading (Keras & PyTorch), metrics
- `model_cnn.py` — Keras CNN builders
- `model_cnn_torch.py` — PyTorch CNN architectures
- `model_resnet50.py` — Keras ResNet50 transfer learning
- `model_resnet50_torch.py` — PyTorch ResNet50 transfer learning
- `train_torch.py` — PyTorch training loop and evaluation

## Usage
```bash
uv run python training.py cnn                  # Keras (default)
uv run python training.py cnn_augmented         # Keras
uv run python training.py cnn_residual          # Keras
uv run python training.py resnet50              # Keras
uv run python training.py resnet50_e2e          # Keras
uv run python training.py cnn --torch           # PyTorch
uv run python training.py resnet50 --torch      # PyTorch
```

## Ideas to try
- Larger conv kernels (3x3)
- More filters (64, 128)
- BatchNormalization
- Dropout regularization
- Learning rate scheduling
- ResNet50 fine-tuning (unfreeze top layers)
- Different pretrained backbones (EfficientNet, MobileNet)
- More training data augmentation strategies
