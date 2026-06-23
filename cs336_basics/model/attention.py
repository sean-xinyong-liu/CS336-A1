"""Scaled dot-product attention and causal multi-head self-attention."""

import torch
from torch import Tensor, nn

from cs336_basics.model.linear import Linear
from cs336_basics.model.rope import RotaryPositionalEmbedding


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Attend over keys and values.

    Args:
        query: Tensor shaped ``(..., queries, d_k)``.
        key: Tensor shaped ``(..., keys, d_k)``.
        value: Tensor shaped ``(..., keys, d_v)``.
        mask: Optional boolean tensor broadcastable to ``(..., queries, keys)``;
            ``True`` entries participate in attention.
    """
    raise NotImplementedError("TODO: implement scaled dot-product attention")


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention, optionally using RoPE."""

    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    output_proj: Linear
    rope: RotaryPositionalEmbedding | None

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # TODO: validate divisibility, construct the four projections, and
        # construct RoPE when theta and max_seq_len are supplied.
        raise NotImplementedError("TODO: construct multi-head self-attention")

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """Process ``(..., seq_len, d_model)`` and preserve its shape."""
        raise NotImplementedError("TODO: implement causal multi-head self-attention")
