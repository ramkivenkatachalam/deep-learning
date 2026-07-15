# Music Genre Classification from Song Lyrics
# Based on MIT 15.773 Hands-On Deep Learning
#
# Dataset: Song lyrics → 3 genre classes (multi-hot bag-of-words)
# Models: unigram, bigram
# Usage:
#   uv run python training.py [unigram|bigram]           # Keras (default)
#   uv run python training.py [unigram|bigram] --torch   # PyTorch

import sys
import time

from common import load_data, prepare_labels, vectorize_text, print_metrics, log_results_csv

# --- Parse flags ---
use_torch = "--torch" in sys.argv
args = [a for a in sys.argv[1:] if a != "--torch"]

# --- Configuration ---

MODEL = args[0] if args else "unigram"
NUM_EPOCHS = {"unigram": 10, "bigram": 10}
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_TOKENS = {"unigram": 5000, "bigram": 20000}
NGRAMS = {"unigram": None, "bigram": 2}

VALID_MODELS = ["unigram", "bigram"]
if MODEL not in VALID_MODELS:
    print(f"Unknown model: {MODEL}. Choose from: {', '.join(VALID_MODELS)}")
    sys.exit(1)

num_epochs = NUM_EPOCHS[MODEL]
max_tokens = MAX_TOKENS[MODEL]

# --- Data ---

train_df, val_df, test_df = load_data()

y_train = prepare_labels(train_df)
y_val = prepare_labels(val_df)
y_test = prepare_labels(test_df)

# --- Text Vectorization (shared across frameworks) ---

x_train, x_val, x_test = vectorize_text(
    train_df.Lyric, val_df.Lyric, test_df.Lyric,
    max_tokens=max_tokens, ngrams=NGRAMS[MODEL],
)

# --- Framework dispatch ---

t0 = time.time()

if use_torch:
    import torch
    torch.manual_seed(42)
    from model_dense_torch import UnigramNet, BigramNet
    from train_torch import train_model, evaluate_model

    if MODEL == "unigram":
        model = UnigramNet(max_tokens=max_tokens)
    else:
        model = BigramNet(max_tokens=max_tokens)

    print(model)
    num_params = sum(p.numel() for p in model.parameters())

    # PyTorch CrossEntropyLoss needs integer labels
    y_train_int = y_train.argmax(axis=1)
    y_val_int = y_val.argmax(axis=1)
    y_test_int = y_test.argmax(axis=1)

    history = train_model(model, x_train, y_train_int, x_val, y_val_int,
                          num_epochs=num_epochs, batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE)
    training_seconds = round(time.time() - t0, 1)

    test_loss, test_accuracy = evaluate_model(model, x_test, y_test_int)

else:
    import keras
    keras.utils.set_random_seed(42)
    from model_dense import build_unigram, build_bigram

    if MODEL == "unigram":
        model = build_unigram(max_tokens=max_tokens)
    else:
        model = build_bigram(max_tokens=max_tokens)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history_obj = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=num_epochs,
        batch_size=BATCH_SIZE,
    )
    training_seconds = round(time.time() - t0, 1)

    history = history_obj.history
    num_params = model.count_params()

    test_loss, test_accuracy = model.evaluate(x_test, y_test)

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
                framework="torch" if use_torch else "keras", model=MODEL)
