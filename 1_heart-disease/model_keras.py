# Keras model builder for heart disease classifier

from tensorflow import keras


def build(input_dim):
    """Input → Dense(16, relu) → Dropout(0.3) → Dense(1, sigmoid)"""
    inp = keras.Input(shape=(input_dim,))
    h = keras.layers.Dense(16, activation='relu', name="Hidden")(inp)
    h = keras.layers.Dropout(0.3)(h)
    out = keras.layers.Dense(1, activation='sigmoid', name="Output")(h)
    return keras.Model(inp, out)
