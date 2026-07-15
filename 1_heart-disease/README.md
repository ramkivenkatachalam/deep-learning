# Heart Disease Binary Classifier

Binary classification on the [UCI Heart Disease dataset](http://storage.googleapis.com/download.tensorflow.org/data/heart.csv) (303 samples, 13 features). Part of MIT 15.773 Hands-On Deep Learning (Spring 2024).

## Dataset

- **Samples**: 303 (242 train / 61 test, 80/20 split, seed=41)
- **Features**: 13 raw → 21 after one-hot encoding categorical columns (sex, cp, fbs, restecg, exang, ca, thal)
- **Preprocessing**: Numerical features (age, trestbps, chol, thalach, oldpeak) standardized using training set statistics
- **Target**: Binary (heart disease present or not)

## Best models

**Keras**: `Input(21) → Dense(16, relu) → Dropout(0.3) → Dense(1, sigmoid)`
- **Optimizer**: Adam, learning rate 0.0005
- **Loss**: Binary crossentropy
- **Epochs**: 500, batch size 32, validation split 0.2

**PyTorch**: `Input(21) → Dense(16, relu) → Dropout(0.3) → Dense(8, relu) → Dense(1, sigmoid)`
- **Optimizer**: AdamW, learning rate 0.0003, weight decay 0.05
- **Loss**: BCELoss
- **Epochs**: 500, batch size 32, validation split 0.2

## Results

| Framework | Accuracy | Architecture | Params |
|---|---|---|---|
| **Keras** | **93.44%** | Dense(16, relu) → Dropout(0.3) → Dense(1, sigmoid) | 497 |
| **PyTorch** | **91.80%** | Dense(16, relu) → Dropout(0.3) → Dense(8, relu) → Dense(1, sigmoid) | 625 |

![Experiment results](experiments.png)

## Usage

```bash
python training.py           # Keras (default)
python training.py --torch   # PyTorch
```


## Key findings

### What improved accuracy

- **Adding a second hidden layer Dense(8) — PyTorch 86.89% → 91.80%** — The single biggest jump. With only 21 input features and a tiny dataset, a single Dense(16) layer can't learn enough non-linear feature combinations. The extra Dense(8) layer lets the network compose features hierarchically — first layer finds patterns, second layer combines them. This was the key to closing the gap with Keras.

- **AdamW with weight decay — PyTorch** — Switching from Adam to AdamW with weight_decay=0.05 didn't change test accuracy (still 91.80%) but significantly improved validation loss (0.29 → 0.24). Weight decay penalizes large weights, pushing toward simpler solutions. On a dataset this small, preventing the model from memorizing a few outlier patients matters.

- **Lower learning rate 0.0005 → 0.0003 — PyTorch** — Again same accuracy but better loss. The deeper architecture needed a gentler learning rate to converge to a better minimum rather than bouncing around it.

- **Keras baseline already optimal at 93.44%** — The original single-layer architecture was already the best for Keras. 10 experiments (wider layers, deeper networks, AdamW, L2, different batch sizes, more epochs) all matched or worsened accuracy. When a simple model already fits the data well, adding complexity just adds ways to overfit.

### What didn't help

- **Batch normalization** — With only 242 training samples and batch size 32, each batch has ~7 samples. BatchNorm statistics are too noisy at this scale to be useful, and added parameters just increase overfitting risk.
- **L2 regularization** — Hurt accuracy (93.44% → 88.52% in Keras). The models are already small enough that weight magnitudes aren't a problem.
- **He weight initialization** — Dropped PyTorch to 83.61%. PyTorch's default Kaiming uniform init already uses a similar strategy; switching to the exact He normal variant changed the starting point in a way that the optimizer couldn't recover from on this tiny dataset.
- **LeakyReLU, wider/narrower layers** — No improvement. The bottleneck isn't activation saturation or model capacity — it's the limited data.
- **More epochs (500 → 800–1000)** — Same or worse accuracy with increasing validation loss. Classic overfitting on a small dataset.
- **Removing Dropout** — Accuracy dropped (91.80% → 86.89% in PyTorch). With so few samples, Dropout is essential to prevent co-adaptation of neurons.
- **Larger batch sizes (32 → 64)** — Reduced accuracy in Keras (93.44% → 86.89%). Smaller batches provide more gradient noise, which acts as implicit regularization — important when you only have 242 training samples.

### Why PyTorch trails Keras by ~1.6%

The same single-layer architecture gets 93.44% in Keras but only 86.89% in PyTorch. Testing with matched validation splits (Keras takes the last 20%, PyTorch uses a random permutation) closed the gap partially (86.89% → 88.52%), but ~5 points remain due to different default weight initialization between the frameworks. With only 303 samples, these small differences get amplified. PyTorch needed a deeper architecture to compensate.
