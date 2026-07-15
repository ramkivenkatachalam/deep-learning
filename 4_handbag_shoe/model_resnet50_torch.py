# PyTorch ResNet50 transfer learning for handbag-shoe classifier

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


def get_resnet50_base():
    """Return a frozen, headless ResNet50 pretrained on ImageNet.

    Removes avgpool and fc layers. Output shape: (batch, 2048, 7, 7).
    """
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Remove avgpool and fc
    modules = list(resnet.children())[:-2]
    base = nn.Sequential(*modules)
    for param in base.parameters():
        param.requires_grad = False
    base.eval()
    return base


def extract_features(data_loader, resnet50_base):
    """Run DataLoader through frozen ResNet50 base, return (features, labels) numpy arrays.

    Expects DataLoader with ResNet50 normalization already applied.
    Output features shape: (N, 2048, 7, 7).
    """
    all_features = []
    all_labels = []
    resnet50_base.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            features = resnet50_base(images)
            all_features.append(features.numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)


class ResNet50Head(nn.Module):
    """Classification head: Input(2048,7,7) → Flatten → Dense(256) → Dropout → Dense(1, sigmoid)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class ResNet50E2E(nn.Module):
    """End-to-end ResNet50 with frozen base + classification head.

    Input: normalized images (batch, 3, 224, 224).
    """
    def __init__(self):
        super().__init__()
        self.base = get_resnet50_base()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.base(x)
        return self.head(features)
