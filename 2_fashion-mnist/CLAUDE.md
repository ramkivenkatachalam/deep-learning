# Fashion-MNIST Image Classifier

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
- `common.py` — data loading, labels, metrics output, CSV logging (framework-agnostic)
- `model_keras.py` — Keras model builder
- `model_torch.py` — PyTorch model architecture
- `train_torch.py` — PyTorch training loop and evaluation

## Constraints
- **NO CNN layers.** Do not use Conv2D, MaxPooling2D, or any convolutional layers. MLP/Dense-only architectures.

## Ideas to try
- Dropout regularization
- Batch normalization
- Learning rate tuning
- More/fewer epochs
- Different optimizers (SGD+momentum, AdamW)
- Data augmentation (shifts, flips)
- Different activation functions
- L2 weight regularization
- Learning rate scheduling
- Deeper/wider dense architectures
- Mixture of Experts
- Skip connections between dense layers
