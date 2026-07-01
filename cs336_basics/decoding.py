"""Autoregressive decoding utilities for language models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters(), torch.empty(0)).device


def apply_top_p(probs: Tensor, top_p: float | None) -> Tensor:
    """Apply nucleus filtering to a probability distribution.

    ``top_p`` keeps the smallest set of highest-probability tokens whose
    cumulative mass reaches the threshold, then renormalizes.
    """
    if top_p is None or top_p >= 1:
        return probs

    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs - sorted_probs >= top_p

    sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0)
    filtered_probs = torch.zeros_like(probs).scatter(dim=-1, index=sorted_indices, src=sorted_probs)
    return filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)


def sample_next_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_p: float | None = None,
) -> Tensor:
    """Sample token IDs from next-token logits.

    Args:
        logits: Tensor shaped ``(..., vocab_size)``.
        temperature: Softmax temperature. ``0`` selects the greedy argmax.
        top_p: Optional nucleus sampling threshold.
    """
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if temperature == 0:
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    probs = apply_top_p(probs, top_p)
    return torch.multinomial(probs.reshape(-1, probs.shape[-1]), num_samples=1).reshape(probs.shape[:-1])


def generate_token_ids(
    model: nn.Module,
    prompt_token_ids: Sequence[int],
    max_new_tokens: int,
    end_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float | None = None,
) -> list[int]:
    """Generate token IDs autoregressively from ``model``.

    The model is called with only the most recent ``model.context_length`` tokens
    when that attribute exists.
    """
    device = _model_device(model)
    token_ids = list(prompt_token_ids)
    generated: list[int] = []
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_length = getattr(model, "context_length", None)
            model_token_ids = token_ids[-context_length:] if context_length is not None else token_ids
            model_input = torch.tensor([model_token_ids], device=device, dtype=torch.long)
            logits = model(model_input)[:, -1, :]
            next_token_id = int(sample_next_token(logits, temperature, top_p).item())

            token_ids.append(next_token_id)
            if end_token_id is not None and next_token_id == end_token_id:
                break
            generated.append(next_token_id)

    if was_training:
        model.train()

    return generated


def generate_completion(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float | None = None,
    end_token: str = "<|endoftext|>",
) -> str:
    """Generate text from a prompt until ``end_token`` or ``max_new_tokens``."""
    prompt_ids = tokenizer.encode(prompt)
    output_ids = generate_token_ids(
        model=model,
        prompt_token_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        end_token_id=tokenizer.encode(end_token)[0],
        temperature=temperature,
        top_p=top_p,
    )
    return tokenizer.decode(output_ids)


__all__ = [
    "apply_top_p",
    "generate_completion",
    "generate_token_ids",
    "sample_next_token",
]
