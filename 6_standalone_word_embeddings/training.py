# Standalone Word Embeddings for Music Genre Classification
# Based on MIT 15.773 Hands-On Deep Learning
#
# Dataset: Song lyrics → 3 genre classes (word embedding models)
# Models: glove_frozen, glove_finetune, custom
# Usage: uv run python training.py [glove_frozen|glove_finetune|custom]

import sys
import time

import keras

from common import (
    load_data, prepare_labels, load_glove_embeddings,
    build_embedding_matrix, print_metrics, log_results_csv,
)
from model_embedding import build_glove_frozen, build_glove_finetune, build_custom

keras.utils.set_random_seed(42)

# --- Configuration ---

MODEL = "glove_frozen"
NUM_EPOCHS = {"glove_frozen": 30, "glove_finetune": 20, "custom": 10}
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_TOKENS = 5000
MAX_LENGTH = 300
EMBEDDING_DIM = 300

if len(sys.argv) > 1:
    MODEL = sys.argv[1]

VALID_MODELS = ["glove_frozen", "glove_finetune", "custom"]
if MODEL not in VALID_MODELS:
    print(f"Unknown model: {MODEL}. Choose from: {', '.join(VALID_MODELS)}")
    sys.exit(1)

num_epochs = NUM_EPOCHS.get(MODEL, 10)

# --- Data ---

train_df, val_df, test_df = load_data()

y_train = prepare_labels(train_df)
y_val = prepare_labels(val_df)
y_test = prepare_labels(test_df)

# --- Text Vectorization (int mode for embeddings) ---

text_vectorization = keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=MAX_LENGTH,
)
text_vectorization.adapt(train_df.Lyric)

x_train = text_vectorization(train_df.Lyric)
x_val = text_vectorization(val_df.Lyric)
x_test = text_vectorization(test_df.Lyric)

# --- GloVe embeddings (skip for custom model) ---

embedding_matrix = None
if MODEL in ("glove_frozen", "glove_finetune"):
    embeddings_index = load_glove_embeddings(EMBEDDING_DIM)
    vocabulary = text_vectorization.get_vocabulary()
    embedding_matrix = build_embedding_matrix(vocabulary, embeddings_index, MAX_TOKENS, EMBEDDING_DIM)

# --- Build model ---

t0 = time.time()

if MODEL == "glove_frozen":
    model = build_glove_frozen(embedding_matrix, MAX_TOKENS, EMBEDDING_DIM, MAX_LENGTH)
elif MODEL == "glove_finetune":
    model = build_glove_finetune(embedding_matrix, MAX_TOKENS, EMBEDDING_DIM, MAX_LENGTH)
elif MODEL == "custom":
    model = build_custom(MAX_TOKENS, EMBEDDING_DIM, MAX_LENGTH)

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
