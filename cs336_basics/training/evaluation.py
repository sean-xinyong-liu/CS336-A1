"""Evaluation helpers for language-model training."""

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.training.batching import get_batch
from cs336_basics.training.losses import cross_entropy


@torch.no_grad()
def estimate_loss(
    model: TransformerLM,
    tokens: np.ndarray,
    batch_size: int,
    context_length: int,
    eval_iters: int,
    device: torch.device | str,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        inputs, targets = get_batch(tokens, batch_size, context_length, device)
        logits = model(inputs)
        losses.append(cross_entropy(logits, targets).item())
    model.train()
    return sum(losses) / len(losses)
