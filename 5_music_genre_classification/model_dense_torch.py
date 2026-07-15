# PyTorch dense model builders for music genre classification
# Input: multi-hot bag-of-words vector → Dense layers → 3-class (no softmax, uses CrossEntropyLoss)

import torch.nn as nn


class UnigramNet(nn.Module):
    """Unigram BoW → Linear(16, relu) → Dropout(0.3) → Linear(3)."""
    def __init__(self, max_tokens=5000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(max_tokens, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        return self.net(x)


class BigramNet(nn.Module):
    """Bigram BoW → Linear(8, relu) → Dropout(0.5) → Linear(3)."""
    def __init__(self, max_tokens=20000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(max_tokens, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8, 3),
        )

    def forward(self, x):
        return self.net(x)
