"""Gradient clipping helpers."""

from collections.abc import Iterable

import torch


EPS = 1e-6


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Clip gradients in place so their global L2 norm is at most ``max_l2_norm``.

    Parameters without gradients should be ignored.
    """
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads:
        return

    per_norms = torch.stack([torch.linalg.vector_norm(grad) for grad in grads])
    total_norm = torch.linalg.vector_norm(per_norms)
    clip_coef = max_l2_norm / (total_norm + EPS)

    if total_norm > max_l2_norm:
        for grad in grads:
            grad.mul_(clip_coef)
