# Framework-agnostic utilities for heart disease classifier

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from shared import print_metrics, log_results_csv


def load_data(random_state=41):
    """Load UCI Heart Disease dataset, preprocess, and split into train/test numpy arrays."""
    df = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/heart.csv')

    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'ca', 'thal']
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    target_col = 'target'

    # One-hot encode categorical features (expands from 13 to 21 input features)
    df = pd.get_dummies(df, columns=categorical_cols)

    # 80/20 train/test split (fixed seed for reproducibility)
    test_df = df.sample(frac=0.2, random_state=random_state)
    train_df = df.drop(test_df.index)

    # Standardize numerical features using training set statistics
    means = train_df[numerical_cols].mean()
    stds = train_df[numerical_cols].std()

    train_df[numerical_cols] = (train_df[numerical_cols] - means) / stds
    test_df[numerical_cols] = (test_df[numerical_cols] - means) / stds

    feature_cols = [c for c in train_df.columns if c != target_col]

    train_X = train_df[feature_cols].to_numpy().astype(float)
    train_Y = train_df[target_col].to_numpy().astype(float)
    test_X = test_df[feature_cols].to_numpy().astype(float)
    test_Y = test_df[target_col].to_numpy().astype(float)

    return train_X, train_Y, test_X, test_Y


