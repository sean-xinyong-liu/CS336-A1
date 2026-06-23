"""Numerically stable softmax."""

from torch import Tensor


def softmax(x: Tensor, dim: int) -> Tensor:
    """Normalize ``x`` along ``dim`` while preserving its shape."""
    raise NotImplementedError("TODO: implement max-shifted softmax")
