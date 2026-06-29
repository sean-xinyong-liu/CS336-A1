"""Rotary positional embeddings (RoPE)."""

import torch
from torch import Tensor, nn
from einops import einsum, rearrange


class RotaryPositionalEmbedding(nn.Module):
    """Apply rotary position information to query or key vectors."""

    def __init__(
        self,
        d_k: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even to apply RoPE.")

        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        dimension_indices = torch.arange(
            0,
            d_k,
            2,
            dtype=torch.float32,
            device=device,
        )
        inv_freq = theta ** (-dimension_indices / d_k)

        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
            device=device,
        )

        angles = einsum(positions, inv_freq, "p, d -> p d")

        cos_cached = torch.cos(angles)
        sin_cached = torch.sin(angles)

        self.register_buffer(
            "cos_cached",
            cos_cached,
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            sin_cached,
            persistent=False,
        )

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """Rotate ``(..., seq_len, d_k)`` using positions ``(..., seq_len)``."""
        cos = self.cos_cached[token_positions].to(dtype=x.dtype)
        sin = self.sin_cached[token_positions].to(dtype=x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        while cos.ndim < x_even.ndim:
            cos = rearrange(
                cos,
                "... seq half_d -> ... 1 seq half_d",
            )
            sin = rearrange(
                sin,
                "... seq half_d -> ... 1 seq half_d",
            )

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        rotated_pairs = torch.stack(
            [rotated_even, rotated_odd],
            dim=-1,
        )

        rotated_pairs = rearrange(
            rotated_pairs,
            "... half_d pair -> ... (half_d pair)",
        )
        return rotated_pairs
