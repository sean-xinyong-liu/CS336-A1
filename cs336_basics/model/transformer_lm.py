"""Decoder-only Transformer language model."""

import torch
from torch import Tensor, nn

from cs336_basics.model.embedding import Embedding
from cs336_basics.model.linear import Linear
from cs336_basics.model.rmsnorm import RMSNorm
from cs336_basics.model.transformer_block import TransformerBlock

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
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, eps, device, dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, eps, device, dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Map ``(batch, seq_len)`` IDs to ``(batch, seq_len, vocab_size)`` logits."""
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

