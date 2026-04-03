# Fashion-MNIST Image Classifier

Based on MIT 15.773 Hands-On Deep Learning (Spring 2024).

## Dataset
- Fashion-MNIST: 60,000 training / 10,000 test images, 28x28 grayscale
- 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- Pixel values normalized to [0, 1]
- Validation split: 0.2 of training data

## Current baseline
- Architecture: Input(28,28) → Flatten → Dense(128, relu) → Dense(128, relu) → Dense(10, softmax)
- Optimizer: Adam (default), Loss: sparse_categorical_crossentropy
- Epochs: 20, Batch: 64, Validation split: 0.2
- test_accuracy: TBD (run baseline to establish)

## Ideas to try
- Convolutional layers (Conv2D + MaxPool)
- Dropout regularization
- Batch normalization
- Learning rate tuning
- More/fewer epochs
- Different optimizers (SGD+momentum, AdamW)
- Data augmentation (shifts, flips)
- Different activation functions
- L2 weight regularization
- Learning rate scheduling
- Deeper/wider architectures
