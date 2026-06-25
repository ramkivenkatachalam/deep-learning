# Fashion-MNIST 10-class image classifier
# Based on MIT 15.773 Hands-On Deep Learning (Spring 2024)
# Original colab: https://colab.research.google.com/drive/14Vv8YgcflIVF_cY9YFsx7Ae774nCuvLn
#
# Dataset: Fashion-MNIST (60k train / 10k test, 28x28 grayscale, 10 classes)
# Usage:
#   python training.py           # Keras (default)
#   python training.py --torch   # PyTorch

import sys
import time
from common import load_data, get_labels, print_metrics, log_results_csv

# --- Parse flags ---
use_torch = "--torch" in sys.argv

# --- Data loading (framework-agnostic) ---
x_train, y_train, x_test, y_test = load_data()

# --- Hyperparameters ---
num_epochs = 80
batch_size = 128
weight_decay = 5e-4
learning_rate = 1e-3

# --- Framework dispatch ---
if use_torch:
    import torch
    torch.manual_seed(41)
    from model_torch import build
    from train_torch import train_model, evaluate_model

    model = build()
    print(model)
    num_params = sum(p.numel() for p in model.parameters())

    t0 = time.time()
    history = train_model(model, x_train, y_train,
                          num_epochs=num_epochs, batch_size=batch_size,
                          learning_rate=learning_rate, weight_decay=weight_decay,
                          validation_split=0.2, warmup_epochs=3)
    training_seconds = round(time.time() - t0, 1)

    test_loss, test_accuracy = evaluate_model(model, x_test, y_test)
else:
    import tensorflow as tf
    from tensorflow import keras
    keras.utils.set_random_seed(41)
    from model_keras import build

    model = build()
    model.summary()
    num_params = model.count_params()

    # Manual val split for label smoothing
    val_split = 0.2
    num_val = int(len(x_train) * val_split)
    x_val, y_val = x_train[:num_val], y_train[:num_val]
    x_tr, y_tr = x_train[num_val:], y_train[num_val:]

    y_tr_oh = keras.utils.to_categorical(y_tr, 10)
    y_val_oh = keras.utils.to_categorical(y_val, 10)
    y_test_oh = keras.utils.to_categorical(y_test, 10)

    steps_per_epoch = len(x_tr) // batch_size
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate, decay_steps=num_epochs * steps_per_epoch,
        warmup_target=learning_rate, warmup_steps=3 * steps_per_epoch
    )
    optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    t0 = time.time()
    history_obj = model.fit(x_tr, y_tr_oh, epochs=num_epochs, batch_size=batch_size,
                            validation_data=(x_val, y_val_oh), verbose=True)
    training_seconds = round(time.time() - t0, 1)

    history = history_obj.history
    test_loss, test_accuracy = model.evaluate(x_test, y_test_oh)

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
