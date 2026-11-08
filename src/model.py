"""
fastText classifier (Joulin et al., 2016, Section 2) in PyTorch.

    x_doc  = mean of the embeddings of its unigram ids and hashed n-gram ids  (matrix A)
    logits = B x_doc                                                          (matrix B)
    p(y)   = softmax(logits)

This is a linear model with a rank constraint (the hidden "text representation" has
h = embedding_dim units). `nn.EmbeddingBag(mode="mean")` computes the averaged bag
in one fused CPU kernel, and `sparse=True` means each step only touches the rows of
A that occur in the batch - the property that makes fastText fast.

Initialisation follows the C++ code (FastText::train -> createInput/createOutput):
A ~ U(-1/dim, 1/dim), B = 0, no bias term.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configs.config import Config


class FastTextClassifier(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, num_classes: int,
                 init_uniform_scale: float = 0.1, zero_init_output: bool = True, sparse: bool = True):
        super().__init__()
        self.embedding = nn.EmbeddingBag(num_embeddings, embedding_dim, mode="mean", sparse=sparse)
        self.fc = nn.Linear(embedding_dim, num_classes, bias=False)  # fastText has no bias
        nn.init.uniform_(self.embedding.weight, -init_uniform_scale, init_uniform_scale)
        if zero_init_output:
            nn.init.zeros_(self.fc.weight)

    def forward(self, features: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """features: 1-D LongTensor of concatenated ids; offsets: start of each document."""
        hidden = self.embedding(features, offsets)   # [batch, dim] averaged bag of features
        return self.fc(hidden)                       # [batch, num_classes] logits

    @classmethod
    def from_config(cls, cfg: Config, vocab_size: int) -> "FastTextClassifier":
        return cls(
            num_embeddings=vocab_size + cfg.effective_bucket,
            embedding_dim=cfg.embedding_dim,
            num_classes=cfg.num_classes,
            init_uniform_scale=cfg.init_uniform_scale,
            zero_init_output=cfg.zero_init_output,
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
