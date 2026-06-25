# PyTorch model builder for heart disease classifier

import torch.nn as nn


class HeartDiseaseNet(nn.Module):
    """Input → Dense(16, relu) → Dropout(0.3) → Dense(8, relu) → Dense(1, sigmoid)"""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build(input_dim):
    return HeartDiseaseNet(input_dim)
