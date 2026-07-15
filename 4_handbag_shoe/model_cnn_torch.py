# PyTorch CNN model builders for handbag-shoe classifier

import torch
import torch.nn as nn


class CNN(nn.Module):
    """CNN: 2x Conv2d(32, 3x3)+MaxPool → Flatten → Dense(128) → Dropout → Dense(1, sigmoid).

    Input: (batch, 3, 224, 224) with values in [0,1].
    Conv path: 224 → Conv(3) → 222 → Pool(2) → 111 → Conv(3) → 109 → Pool(2) → 54
    Flatten: 32 * 54 * 54 = 93312
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 54 * 54, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    """Two 3x3 convs with skip connection. Projects shortcut if channels don't match."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        h = self.relu(self.conv1(x))
        h = self.conv2(h)
        return self.relu(h + shortcut)


class CNNResidual(nn.Module):
    """CNN with residual connections.

    ResBlock(3→32) → Pool → ResBlock(32→64) → Pool → ResBlock(64→128) → Pool → Flatten
    → Dense(128) → Dropout → Dense(1, sigmoid).

    Input: (batch, 3, 224, 224) with values in [0,1].
    Path: 224 → ResBlock(pad=1) → 224 → Pool → 112 → ResBlock → 112 → Pool → 56
        → ResBlock → 56 → Pool → 28
    Flatten: 128 * 28 * 28 = 100352
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ResidualBlock(3, 32),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
