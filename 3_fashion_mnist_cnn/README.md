# Fashion-MNIST CNN Classifier

10-class image classification on [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (70,000 grayscale images, 28x28). Part of MIT 15.773 Hands-On Deep Learning (Spring 2024).

## Results

| | Accuracy | Architecture |
|---|---|---|
| **Baseline** | 87.01% | Conv2D(32)x2 → MaxPool → Dense(256) → Dense(10) |
| **Best** | **94.87%** | 3 conv blocks (64-128-256) + Dense(512) + augmentation |

**8 experiments** were run.

![Experiment results](experiments.png)

## Dataset

- **Images**: 60,000 train / 10,000 test, 28x28 grayscale, normalized to [0, 1]
- **Classes**: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- **Validation**: 20% of training data

## Best model

```
Input(28, 28, 1)
Conv2D(64, 3x3, relu) + BatchNorm + Conv2D(64, 3x3, relu) + BatchNorm + MaxPool + Dropout(0.25)
Conv2D(128, 3x3, relu) + BatchNorm + Conv2D(128, 3x3, relu) + BatchNorm + MaxPool + Dropout(0.25)
Conv2D(256, 3x3, relu) + BatchNorm + Conv2D(256, 3x3, relu) + BatchNorm + MaxPool + Dropout(0.25)
Flatten → Dense(512, relu) + BatchNorm + Dropout(0.5)
Dense(10, softmax)
```

- **Optimizer**: AdamW, cosine decay LR schedule (initial 2e-3, 3-epoch warmup), weight decay 5e-4
- **Loss**: Categorical crossentropy with label smoothing 0.1
- **Augmentation**: Random horizontal flip + reflect pad (2px) + random crop
- **Epochs**: 80, batch size 128
- **Parameters**: 2,335,178

## Key findings

**What improved accuracy:**
- Deeper/wider conv blocks (32 → 64-128-256) with BatchNorm + Dropout
- AdamW with CosineDecay LR + warmup (93.32% → 94.01%)
- tf.data augmentation: flip + reflect pad + random crop (94.01% → 94.71%)
- More epochs with higher LR and warmup (94.71% → 94.87%)

**What didn't help:**
- GELU activation (94.76% but 18x slower on CPU — not worth it)
- Stronger augmentation with brightness jitter (worse)
- Lower dropout + higher weight decay (test accuracy worse)
