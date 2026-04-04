# Fashion-MNIST Image Classifier

10-class image classification on [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (70,000 grayscale images, 28x28). Part of MIT 15.773 Hands-On Deep Learning (Spring 2024).

## Results

| | Accuracy | Architecture |
|---|---|---|
| **Baseline** | 87.46% | Flatten → Dense(128) x2 → Dense(10) |
| **Best** | **94.94%** | 3 Conv blocks (64-128-256) → Dense(512) → Dense(10) |

**25 experiments** were run to reach the best result.

![Experiment results](experiments.png)

## Dataset

- **Images**: 60,000 train / 10,000 test, 28x28 grayscale, normalized to [0, 1]
- **Classes**: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- **Augmentation**: Random horizontal flip + reflect padding with random crop (via tf.data)
- **Validation**: 20% of training data (no augmentation)

## Best model

```
Conv2D(64)x2 + BatchNorm + MaxPool + Dropout(0.25)
Conv2D(128)x2 + BatchNorm + MaxPool + Dropout(0.25)
Conv2D(256)x2 + BatchNorm + MaxPool + Dropout(0.25)
Dense(512) + BatchNorm + Dropout(0.5)
Dense(10, softmax)
```

- **Optimizer**: AdamW, cosine decay LR schedule (initial 0.002, 3-step warmup), weight decay 5e-4
- **Loss**: Categorical crossentropy with label smoothing 0.1
- **Epochs**: 80, batch size 128
- **Parameters**: 2,335,178

## Key findings

**What improved accuracy:**
- Switching from dense-only to CNN (87.46% → 89.24%)
- Adding BatchNorm after conv layers (89.24% → 91.37%)
- Deeper architecture with double conv blocks (91.37% → 93.09%)
- ReduceLROnPlateau + larger batch size (93.09% → 94.00%)
- Wider conv filters (64-128-256) + Dense(512) (94.00% → 94.49%)
- Weight decay + more epochs (94.49% → 94.66%)
- tf.data augmentation (flip + crop) at 80 epochs (94.66% → 94.94%)

**What didn't help:**
- Higher learning rates without scheduling
- 100 epochs (diminishing returns, risk of overfitting)
- Label smoothing alone (marginal, kept as part of final config)
