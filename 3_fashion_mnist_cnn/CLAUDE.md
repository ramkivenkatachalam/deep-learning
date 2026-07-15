# Fashion-MNIST CNN Classifier

Based on MIT 15.773 Hands-On Deep Learning (Spring 2024).

## Dataset
- Fashion-MNIST: 60,000 training / 10,000 test images, 28x28 grayscale
- 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- Pixel values normalized to [0, 1]
- Validation split: 0.2 of training data

## Usage
```
python training.py           # Keras (default)
python training.py --torch   # PyTorch
```

## File structure
- `training.py` — main entry point, framework dispatch
- `common.py` — data loading, metrics output, CSV logging (framework-agnostic)
- `model_keras.py` — Keras CNN model builder
- `model_torch.py` — PyTorch CNN model architecture
- `train_torch.py` — PyTorch training loop with augmentation and evaluation

## Current architecture
- 3 conv blocks: 64→128→256 filters, each 2×Conv(3×3)+BN+MaxPool(2×2)+Dropout(0.25)
- Dense head: Flatten → Dense(512)+BN+Dropout(0.5) → Dense(10)
- Data augmentation: RandomHorizontalFlip + reflect-pad(2) + RandomCrop(28)
- Optimizer: AdamW, lr=2e-3, weight_decay=5e-4
- LR schedule: CosineDecay with 3-epoch warmup
- Loss: CrossEntropy with label_smoothing=0.1
- Epochs: 80, batch_size: 128

## Ideas to try
- Larger conv kernels (5x5 in first block)
- More filters (512 in block 3)
- GELU activation
- Deeper architectures (4 conv blocks)
- Cutout / CutMix augmentation
- Mixup training
- More aggressive dropout
- Squeeze-and-Excitation blocks
