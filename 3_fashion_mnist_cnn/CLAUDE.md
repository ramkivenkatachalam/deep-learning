# Fashion-MNIST CNN Classifier

Based on MIT 15.773 Hands-On Deep Learning (Spring 2024).

## Dataset
- Fashion-MNIST: 60,000 training / 10,000 test images, 28x28 grayscale
- 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- Pixel values normalized to [0, 1]
- Validation split: 0.1 of training data

## Current baseline
- Architecture: Conv2D(32,2x2) → MaxPool → Conv2D(32,2x2) → MaxPool → Flatten → Dense(256) → Dense(10)
- Optimizer: Adam, Loss: sparse_categorical_crossentropy
- Epochs: 10, Batch: 64, Validation split: 0.1
- test_accuracy: TBD (run baseline to establish)

## Ideas to try
- Larger conv kernels (3x3)
- More filters (64, 128)
- BatchNormalization after conv layers
- Dropout regularization
- Data augmentation (flips, shifts)
- Learning rate scheduling (CosineDecay)
- AdamW with weight decay
- Label smoothing
- Deeper architectures (3 conv blocks)
- GELU activation
- More epochs
