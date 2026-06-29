"""Scaled dot-product attention and causal multi-head self-attention."""

import torch
from torch import Tensor, nn
from einops import einsum, rearrange

from cs336_basics.model.linear import Linear
from cs336_basics.model.rope import RotaryPositionalEmbedding
from cs336_basics.model.softmax import softmax

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
    d_k = key.shape[-1]
    scores = einsum(query, key, "... queries d_k, ... keys d_k -> ... queries keys")
    scores = scores / d_k ** 0.5

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = softmax(scores, dim=-1)
    output = einsum(attn_weights, value, "... queries keys, ... keys d_v -> ... queries d_v")
    return output


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
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_k = d_model // num_heads

        self.q_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.output_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(self.d_k, theta, max_seq_len, device=device)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """Process ``(..., seq_len, d_model)`` and preserve its shape."""
        seq_len = x.shape[-2]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "... seq (head d_k) -> ... head seq d_k", head=self.num_heads)
        k = rearrange(k, "... seq (head d_k) -> ... head seq d_k", head=self.num_heads)
        v = rearrange(v, "... seq (head d_v) -> ... head seq d_v", head=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        causal_mask = torch.tril(
            torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool),
        )

        attn_output = scaled_dot_product_attention(q, k, v, causal_mask)
        attn_output = rearrange(
            attn_output,
            "... head seq d_v -> ... seq (head d_v)",
        )

        return self.output_proj(attn_output)
