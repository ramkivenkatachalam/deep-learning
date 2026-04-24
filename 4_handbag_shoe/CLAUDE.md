# Handbag-Shoe Binary Classifier

Based on MIT 15.773 Hands-On Deep Learning.

## Dataset
- Custom handbags vs shoes dataset (Dropbox download)
- Images: 224x224 RGB
- Split: 50 train / 25 val / remaining test per class
- Binary classification (handbags=0, shoes=1)

## Models

| Model | Builder file | Description |
|-------|-------------|-------------|
| `cnn` | `model_cnn.py` | Basic CNN: Rescaling → 2x Conv2D(32, 2x2)+MaxPool → Dense(1, sigmoid) |
| `cnn_augmented` | `model_cnn.py` | Same CNN with RandomFlip/RandomRotation/RandomZoom |
| `resnet50` | `model_resnet50.py` | Frozen ResNet50 feature extraction → Dense(256) → Dropout → Dense(1, sigmoid) |
| `resnet50_e2e` | `model_resnet50.py` | End-to-end ResNet50 with augmentation, enables fine-tuning |

## Usage
```bash
uv run python training.py cnn
uv run python training.py cnn_augmented
uv run python training.py resnet50
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
