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
        # TODO: construct attn, ffn, ln1, and ln2.
        raise NotImplementedError("TODO: construct TransformerBlock submodules")

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """Process ``(..., seq_len, d_model)`` and preserve its shape."""
        raise NotImplementedError("TODO: implement the pre-norm residual block")
