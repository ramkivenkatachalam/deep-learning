# Handbag-Shoe Binary Classifier

Based on MIT 15.773 Hands-On Deep Learning.

## Dataset
- Custom handbags vs shoes dataset (Dropbox download)
- Images: 224x224 RGB
- Split: 50 train / 25 val / remaining test per class
- Binary classification (handbags=0, shoes=1)

## Models
- `cnn` — Basic CNN: Rescaling → 2x Conv2D(32, 2x2)+MaxPool → Dense(1, sigmoid)
- `cnn_augmented` — Same CNN with RandomFlip/RandomRotation/RandomZoom
- `resnet50` — Frozen ResNet50 feature extraction → Dense(256) → Dropout → Dense(1, sigmoid)

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
