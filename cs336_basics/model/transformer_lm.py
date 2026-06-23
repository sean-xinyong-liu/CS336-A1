"""Decoder-only Transformer language model."""

import torch
from torch import Tensor, nn

from cs336_basics.model.embedding import Embedding
from cs336_basics.model.linear import Linear
from cs336_basics.model.rmsnorm import RMSNorm


class TransformerLM(nn.Module):
    """A stack of causal Transformer blocks producing vocabulary logits."""

    token_embeddings: Embedding
    layers: nn.ModuleList
    ln_final: RMSNorm
    lm_head: Linear

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        # TODO: construct token_embeddings, layers, ln_final, and lm_head.
        raise NotImplementedError("TODO: construct TransformerLM submodules")

    def forward(self, token_ids: Tensor) -> Tensor:
        """Map ``(batch, seq_len)`` IDs to ``(batch, seq_len, vocab_size)`` logits."""
        raise NotImplementedError("TODO: implement the Transformer LM forward pass")
