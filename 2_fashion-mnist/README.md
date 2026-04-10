# Fashion-MNIST Image Classifier (MLP Only)

10-class image classification on [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (70,000 grayscale images, 28x28). Part of MIT 15.773 Hands-On Deep Learning (Spring 2024).

## Results

| | Accuracy | Architecture |
|---|---|---|
| **Baseline** | 73.93% | Flatten → Dense(128) x2 → Dense(10) |
| **Best** | **90.34%** | Flatten → Dense(1024) → Dense(512) → Dense(256) → Dense(10) |

**20 experiments** were run (MLP only, no CNN).

![Experiment results](experiments.png)

## Dataset

- **Images**: 60,000 train / 10,000 test, 28x28 grayscale, normalized to [0, 1]
- **Classes**: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- **Validation**: 20% of training data

## Best model

```
Flatten(28x28)
Dense(1024, gelu) + BatchNorm + Dropout(0.4)
Dense(512, gelu) + BatchNorm + Dropout(0.3)
Dense(256, gelu) + BatchNorm + Dropout(0.3)
Dense(10, softmax)
```

- **Optimizer**: AdamW, cosine decay LR schedule (initial 1e-3, 3-epoch warmup), weight decay 5e-4
- **Loss**: Categorical crossentropy with label smoothing 0.1
- **Epochs**: 80, batch size 128
- **Parameters**: 1,469,706

## Key findings

**What improved accuracy:**
- GELU activation (single biggest win: 83.98% → 87.64%)
- CosineDecay LR schedule with warmup (88.68% → 90.05%)
- Wider layers (256 → 512 → 1024 first layer)
- BatchNorm + Dropout (73.93% → 83.43%)
- AdamW with weight decay + label smoothing

**What didn't help:**
- Data augmentation (hurts MLP — can't learn spatial invariance)
- MLP-Mixer with patch mixing (underperformed simple wide MLP)
- Skip/residual connections
- SGD+momentum (worse than Adam for this MLP)
- Very deep or very wide networks (diminishing returns)
