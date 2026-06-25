# Heart disease binary classifier
# Based on MIT 15.773 Hands-On Deep Learning (Spring 2024)
# https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/
# Original colab: https://colab.research.google.com/drive/1flLafeFpy8JjLN4H_ertcs5wJE3--TdQ
#
# Dataset: UCI Heart Disease (303 samples, 13 features, binary target)
# Usage:
#   python training.py           # Keras (default)
#   python training.py --torch   # PyTorch

import sys
import time
from common import load_data, print_metrics, log_results_csv

# --- Parse flags ---
use_torch = "--torch" in sys.argv

# --- Data loading (framework-agnostic) ---
train_X, train_Y, test_X, test_Y = load_data(random_state=41)
input_dim = train_X.shape[1]

# --- Hyperparameters ---
num_epochs = 500
batch_size = 32
learning_rate = 0.0003 if use_torch else 0.0005

# --- Framework dispatch ---
if use_torch:
    import torch
    torch.manual_seed(41)
    from model_torch import build
    from train_torch import train_model, evaluate_model

    model = build(input_dim)
    print(model)
    num_params = sum(p.numel() for p in model.parameters())

    t0 = time.time()
    history = train_model(model, train_X, train_Y,
                          num_epochs=num_epochs, batch_size=batch_size,
                          learning_rate=learning_rate, validation_split=0.2)
    training_seconds = round(time.time() - t0, 1)

    test_loss, test_accuracy = evaluate_model(model, test_X, test_Y)
else:
    import tensorflow as tf
    from tensorflow import keras
    keras.utils.set_random_seed(41)
    from model_keras import build

    model = build(input_dim)
    model.summary()
    num_params = model.count_params()

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])

    t0 = time.time()
    history_obj = model.fit(train_X, train_Y, epochs=num_epochs, verbose=True,
                            validation_split=0.2, batch_size=batch_size)
    training_seconds = round(time.time() - t0, 1)

    history = history_obj.history
    test_loss, test_accuracy = model.evaluate(test_X, test_Y)

# --- Results (shared) ---
train_loss = history["loss"][-1]
train_accuracy = history["accuracy"][-1]
val_loss = history["val_loss"][-1]
val_accuracy = history["val_accuracy"][-1]

print_metrics(test_accuracy, test_loss, val_accuracy, val_loss,
              train_accuracy, train_loss, num_params, num_epochs,
              training_seconds)

log_results_csv(test_accuracy, test_loss, val_accuracy, val_loss,
                num_params, training_seconds,
                framework="torch" if use_torch else "keras")
