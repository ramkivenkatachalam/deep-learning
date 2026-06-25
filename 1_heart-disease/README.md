# Heart Disease Binary Classifier

Binary classification on the [UCI Heart Disease dataset](http://storage.googleapis.com/download.tensorflow.org/data/heart.csv) (303 samples, 13 features). Part of MIT 15.773 Hands-On Deep Learning (Spring 2024).

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

## Key findings

**Keras (10 experiments):**
- Original baseline was already well-optimized at 93.44%
- No change improved accuracy — wider layers, deeper networks, different optimizers, regularization, batch sizes all equal or worse

**PyTorch (16 experiments):**
- Baseline started at 86.89% with the same architecture as Keras
- Adding a second hidden layer Dense(8) jumped to 91.80%
- AdamW with weight decay and lower LR improved loss without changing accuracy
- Remaining ~1.6% gap vs Keras likely due to different random init and val split mechanics

**What didn't help (either framework):**
- Batch normalization, L2 regularization, He weight init
- LeakyReLU, wider/narrower layers, different batch sizes
- More epochs beyond 500 (overfitting)
