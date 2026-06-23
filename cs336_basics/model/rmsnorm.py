"""Root-mean-square layer normalization."""

import math

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Normalize the final dimension and apply a learned scale."""

    weight: nn.Parameter

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones((d_model, ), device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize ``(..., d_model)`` and return a tensor of the same shape.

        Perform normalization in float32, then cast back to the input dtype.
        """
        in_dtype = x.dtype 
        x = x.to(torch.float32)
        scalar = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = x * scalar * self.weight

        return result.to(in_dtype)
