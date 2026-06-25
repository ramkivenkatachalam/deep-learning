# PyTorch model builder for Fashion-MNIST classifier

import torch.nn as nn


class FashionMNISTNet(nn.Module):
    """Flatten → Linear(784,1024) → GELU → BN → Drop(0.4) → Linear(1024,512) → GELU → BN → Drop(0.3) → Linear(512,256) → GELU → BN → Drop(0.3) → Linear(256,10)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 1024),
            nn.GELU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


def build():
    return FashionMNISTNet()
