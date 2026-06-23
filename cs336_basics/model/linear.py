"""Bias-free linear projection."""

import torch
from einops import einsum
from torch import Tensor, nn


class Linear(nn.Module):
    """Apply a bias-free linear transformation to the final input dimension.

    The public ``weight`` parameter has shape ``(out_features, in_features)``.
    """

    weight: nn.Parameter

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)


    def forward(self, x: Tensor) -> Tensor:
        """Transform ``(..., in_features)`` into ``(..., out_features)``."""
        return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")
