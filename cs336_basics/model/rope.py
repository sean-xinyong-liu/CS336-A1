"""Rotary positional embeddings (RoPE)."""

import torch
from torch import Tensor, nn


class RotaryPositionalEmbedding(nn.Module):
    """Apply rotary position information to query or key vectors."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # TODO: precompute and register any non-persistent or persistent buffers.
        raise NotImplementedError("TODO: construct RoPE buffers")

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """Rotate ``(..., seq_len, d_k)`` using positions ``(..., seq_len)``."""
        raise NotImplementedError("TODO: apply RoPE")
