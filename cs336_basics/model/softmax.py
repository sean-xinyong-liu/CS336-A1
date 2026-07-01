"""Numerically stable softmax."""
from torch import Tensor


def softmax(x: Tensor, dim: int) -> Tensor:
    """Normalize ``x`` along ``dim`` while preserving its shape."""
    x_max = x.amax(dim=dim, keepdim=True)
    exp_x = (x - x_max).exp()
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
