"""SwiGLU position-wise feed-forward network."""

import torch
from torch import Tensor, nn

from cs336_basics.model.linear import Linear


def silu(x: Tensor) -> Tensor:
    """Apply the SiLU activation elementwise."""
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network with projections named for test weights."""

    w1: Linear
    w2: Linear
    w3: Linear

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # TODO: if d_ff is None, choose approximately 8/3 * d_model rounded
        # to a multiple of 64; then construct w1, w2, and w3.
        raise NotImplementedError("TODO: construct SwiGLU projections")

    def forward(self, x: Tensor) -> Tensor:
        """Transform ``(..., d_model)`` into a tensor of the same shape."""
        raise NotImplementedError("TODO: implement SwiGLU")
