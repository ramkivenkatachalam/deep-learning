# PyTorch model builder for Fashion-MNIST CNN classifier

import torch.nn as nn


class FashionCNN(nn.Module):
    """3 conv blocks (64→128→256, 2×Conv2d+BN+MaxPool+Drop) → Linear(512)+BN+Drop → Linear(10)"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1×28×28 → 64×14×14
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            # Block 2: 64×14×14 → 128×7×7
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            # Block 3: 128×7×7 → 256×3×3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build():
    return FashionCNN()
