# Heart Disease Binary Classifier

Based on MIT 15.773 Hands-On Deep Learning (Spring 2024).

## Dataset
- UCI Heart Disease: 303 samples, 13 features (expands to 21 after one-hot encoding), binary target
- Train/test split: 80/20 with random_state=41
- Source: http://storage.googleapis.com/download.tensorflow.org/data/heart.csv

## Current baseline
- Architecture: Input(21) → Dense(16, relu) → Dropout(0.3) → Dense(1, sigmoid)
- Optimizer: Adam (LR=0.0005), Loss: binary_crossentropy
- Epochs: 500, Batch: 32, Validation split: 0.2
- test_accuracy: 0.9508

## Ideas to try
- More/fewer hidden units
- Additional hidden layers (deeper networks)
- Different dropout rates
- Learning rate scheduling
- Different optimizers (SGD+momentum, RMSprop, AdamW)
- Early stopping with patience
- Batch normalization
- Feature engineering / feature selection
- Different activation functions (tanh, LeakyReLU, swish)
- L1/L2 weight regularization
- Class weighting
- Different batch sizes
