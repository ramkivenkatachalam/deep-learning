# Shared utilities for music genre classifier
# Data download, label prep, text vectorization

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from shared import print_metrics, log_results_csv


TRAIN_URL = "https://www.dropbox.com/scl/fi/ito6bnl2yaf1uw0uqibzf/lyric_genre_train.csv?rlkey=04dkn5un2djza8x0bdmfnlw3u&st=y47qh8i4&dl=1"
VAL_URL = "https://www.dropbox.com/scl/fi/xmywjzqsaa8n5sn1bs0t9/lyric_genre_val.csv?rlkey=hggbeo0s1iaxjpa6z80429xl9&st=6i7d8eau&dl=1"
TEST_URL = "https://www.dropbox.com/scl/fi/fnocl69w9ojs9s5zb0xvf/lyric_genre_test.csv?rlkey=z4hjopw7vaihoh948cbb5mvdp&st=xwond7dp&dl=1"


def load_data(data_dir="./data"):
    """Download train/val/test CSVs and return DataFrames.

    Caches locally to data_dir to avoid re-downloading.
    """
    os.makedirs(data_dir, exist_ok=True)

    urls = {"train": TRAIN_URL, "val": VAL_URL, "test": TEST_URL}
    dfs = {}

    for split, url in urls.items():
        path = os.path.join(data_dir, f"lyric_genre_{split}.csv")
        if os.path.exists(path):
            dfs[split] = pd.read_csv(path, index_col=0)
        else:
            print(f"Downloading {split} data...")
            df = pd.read_csv(url, index_col=0)
            df.to_csv(path)
            dfs[split] = df

    print(f"Train: {dfs['train'].shape[0]}  Val: {dfs['val'].shape[0]}  Test: {dfs['test'].shape[0]}")
    return dfs["train"], dfs["val"], dfs["test"]


def prepare_labels(df):
    """Convert Genre column to one-hot numpy array."""
    return pd.get_dummies(df.Genre).to_numpy(dtype="uint8")


def vectorize_text(train_lyrics, val_lyrics, test_lyrics, max_tokens=5000, ngrams=None):
    """Multi-hot vectorization via Keras TextVectorization. Returns numpy arrays."""
    import keras

    text_vectorization = keras.layers.TextVectorization(
        output_mode="multi_hot", max_tokens=max_tokens, dtype="float32",
        ngrams=ngrams,
    )
    text_vectorization.adapt(train_lyrics)

    x_train = text_vectorization(train_lyrics).numpy()
    x_val = text_vectorization(val_lyrics).numpy()
    x_test = text_vectorization(test_lyrics).numpy()

    return x_train, x_val, x_test
