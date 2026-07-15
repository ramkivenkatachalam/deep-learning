# PyTorch embedding model builders for word embedding genre classification
# Input: int sequences → nn.Embedding → pooling → Dense → 3-class (no softmax)

import torch
import torch.nn as nn


class MaskedAvgPool(nn.Module):
    """Average pooling that ignores padding (index 0)."""
    def forward(self, x, mask):
        # x: (batch, seq_len, embed_dim), mask: (batch, seq_len) bool, True = non-padding
        mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        summed = (x * mask_expanded).sum(dim=1)     # (batch, embed_dim)
        lengths = mask_expanded.sum(dim=1).clamp(min=1)  # (batch, 1)
        return summed / lengths


class MaskedMaxPool(nn.Module):
    """Max pooling that ignores padding (index 0)."""
    def forward(self, x, mask):
        # x: (batch, seq_len, embed_dim), mask: (batch, seq_len) bool
        mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)
        x = x.masked_fill(~mask_expanded, float('-inf'))
        return x.max(dim=1).values  # (batch, embed_dim)


class GloveFrozenNet(nn.Module):
    """GloVe frozen → concat(AvgPool, MaxPool) → Dense(64) → Dense(3)."""
    def __init__(self, embedding_matrix):
        super().__init__()
        num_embeddings, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), freeze=True, padding_idx=0
        )
        self.avg_pool = MaskedAvgPool()
        self.max_pool = MaskedMaxPool()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        mask = x != 0
        embedded = self.embedding(x)
        avg = self.avg_pool(embedded, mask)
        mx = self.max_pool(embedded, mask)
        pooled = torch.cat([avg, mx], dim=1)
        return self.classifier(pooled)


class GloveFinetuneNet(nn.Module):
    """GloVe trainable → AvgPool → Dense(8) → Dense(3)."""
    def __init__(self, embedding_matrix):
        super().__init__()
        num_embeddings, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), freeze=False, padding_idx=0
        )
        self.avg_pool = MaskedAvgPool()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
        )

    def forward(self, x):
        mask = x != 0
        embedded = self.embedding(x)
        pooled = self.avg_pool(embedded, mask)
        return self.classifier(pooled)


class CustomEmbeddingNet(nn.Module):
    """Random-init trainable → AvgPool → Dense(16) → Dropout(0.3) → Dense(3)."""
    def __init__(self, max_tokens, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(max_tokens, embedding_dim, padding_idx=0)
        self.avg_pool = MaskedAvgPool()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        mask = x != 0
        embedded = self.embedding(x)
        pooled = self.avg_pool(embedded, mask)
        return self.classifier(pooled)
