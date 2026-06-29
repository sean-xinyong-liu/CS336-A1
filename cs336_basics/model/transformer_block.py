"""Pre-normalization Transformer block."""

import torch
from torch import Tensor, nn

from cs336_basics.model.attention import MultiHeadSelfAttention
from cs336_basics.model.rmsnorm import RMSNorm
from cs336_basics.model.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """A pre-norm attention block followed by a pre-norm SwiGLU block."""

    attn: MultiHeadSelfAttention
    ffn: SwiGLU
    ln1: RMSNorm
    ln2: RMSNorm

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.ln1 = RMSNorm(d_model, eps, device, dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len, device, dtype)
        self.ln2 = RMSNorm(d_model, eps, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """Process ``(..., seq_len, d_model)`` and preserve its shape."""
        attn_output = self.attn(self.ln1(x), token_positions)
        x = x + attn_output
        ffn_output = self.ffn(self.ln2(x))
        return x + ffn_output
