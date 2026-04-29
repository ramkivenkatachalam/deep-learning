# Dense model builders for music genre classification
# Input: multi-hot bag-of-words vector → Dense layers → 3-class softmax

import keras


def build_unigram(max_tokens=5000):
    """Unigram bag-of-words → Dense(16, relu) → Dropout(0.3) → Dense(3, softmax)."""
    input = keras.Input(shape=(max_tokens,), name="input_layer")
    h = keras.layers.Dense(16, activation="relu", name="hidden_layer")(input)
    h = keras.layers.Dropout(0.3, name="dropout")(h)
    output = keras.layers.Dense(3, activation="softmax", name="output_layer")(h)
    return keras.Model(inputs=input, outputs=output)


def build_bigram(max_tokens=20000):
    """Bigram bag-of-words → Dense(8, relu) → Dropout(0.5) → Dense(3, softmax)."""
    input = keras.Input(shape=(max_tokens,), name="input_layer")
    h = keras.layers.Dense(8, activation="relu", name="hidden_layer")(input)
    h = keras.layers.Dropout(0.5, name="dropout")(h)
    output = keras.layers.Dense(3, activation="softmax", name="output_layer")(h)
    return keras.Model(inputs=input, outputs=output)
