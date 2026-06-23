"""Token embedding lookup."""

import torch
from torch import Tensor, nn


class Embedding(nn.Module):
    """Map token IDs to vectors stored in a trainable embedding matrix."""

    weight: nn.Parameter

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return embeddings with shape ``(*token_ids.shape, embedding_dim)``."""
        return self.weight[token_ids]
