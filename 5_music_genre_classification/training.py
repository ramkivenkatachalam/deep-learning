# Music Genre Classification from Song Lyrics
# Based on MIT 15.773 Hands-On Deep Learning
#
# Dataset: Song lyrics → 3 genre classes (multi-hot bag-of-words)
# Models: unigram, bigram
# Usage: uv run python training.py [unigram|bigram]

import sys
import time

import keras

from common import load_data, prepare_labels, print_metrics, log_results_csv
from model_dense import build_unigram, build_bigram

keras.utils.set_random_seed(42)

# --- Configuration ---

MODEL = "unigram"
NUM_EPOCHS = {"unigram": 10, "bigram": 10}
LEARNING_RATE = 1e-3
BATCH_SIZE = 32

if len(sys.argv) > 1:
    MODEL = sys.argv[1]

num_epochs = NUM_EPOCHS.get(MODEL, 10)

# --- Data ---

train_df, val_df, test_df = load_data()

y_train = prepare_labels(train_df)
y_val = prepare_labels(val_df)
y_test = prepare_labels(test_df)

# --- Text Vectorization ---

if MODEL == "unigram":
    max_tokens = 5000
    text_vectorization = keras.layers.TextVectorization(
        output_mode="multi_hot", max_tokens=max_tokens, dtype="float32"
    )
elif MODEL == "bigram":
    max_tokens = 20000
    text_vectorization = keras.layers.TextVectorization(
        output_mode="multi_hot", max_tokens=max_tokens, dtype="float32", ngrams=2
    )
else:
    print(f"Unknown model: {MODEL}. Choose from: unigram, bigram")
    sys.exit(1)

text_vectorization.adapt(train_df.Lyric)

x_train = text_vectorization(train_df.Lyric)
x_val = text_vectorization(val_df.Lyric)
x_test = text_vectorization(test_df.Lyric)

# --- Build model and train ---

t0 = time.time()

if MODEL == "unigram":
    model = build_unigram(max_tokens=max_tokens)
elif MODEL == "bigram":
    model = build_bigram(max_tokens=max_tokens)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=num_epochs,
    batch_size=BATCH_SIZE,
)

training_seconds = round(time.time() - t0, 1)
num_params = model.count_params()

# --- Evaluation ---

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print_metrics(history, test_loss, test_accuracy, num_params, training_seconds)

h = history.history
log_results_csv(
    test_accuracy, test_loss,
    h["val_accuracy"][-1], h["val_loss"][-1],
    num_params, training_seconds,
    description=MODEL,
)
