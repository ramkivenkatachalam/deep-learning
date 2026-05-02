# Embedding model builders for word embedding genre classification
# Input: int sequences → Embedding → GlobalAvgPool → Dense → 3-class softmax

import keras


def build_glove_frozen(embedding_matrix, max_tokens, embedding_dim, max_length):
    """GloVe pre-trained embedding (frozen) → GlobalAvgPool → Dense(8) → Dense(3)."""
    embedding_layer = keras.layers.Embedding(
        max_tokens,
        embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=False,
        name="Embedding",
    )

    input = keras.Input(shape=(max_length,), name="input")
    x = embedding_layer(input)
    avg = keras.layers.GlobalAveragePooling1D()(x)
    mx = keras.layers.GlobalMaxPooling1D()(x)
    x = keras.layers.Concatenate()([avg, mx])
    x = keras.layers.Dense(64, activation="relu")(x)
    output = keras.layers.Dense(3, activation="softmax")(x)
    return keras.Model(inputs=input, outputs=output)


def build_glove_finetune(embedding_matrix, max_tokens, embedding_dim, max_length):
    """GloVe pre-trained embedding (trainable) → GlobalAvgPool → Dense(8) → Dense(3)."""
    embedding_layer = keras.layers.Embedding(
        max_tokens,
        embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=True,
        name="Embedding",
    )

    input = keras.Input(shape=(max_length,), name="input")
    x = embedding_layer(input)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(3, activation="softmax")(x)
    return keras.Model(inputs=input, outputs=output)


def build_custom(max_tokens, embedding_dim, max_length):
    """Random-init embedding (trainable) → GlobalAvgPool → Dense(8) → Dense(3)."""
    embedding_layer = keras.layers.Embedding(
        max_tokens,
        embedding_dim,
        trainable=True,
        name="Embedding",
    )

    input = keras.Input(shape=(max_length,), name="input")
    x = embedding_layer(input)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(3, activation="softmax")(x)
    return keras.Model(inputs=input, outputs=output)
