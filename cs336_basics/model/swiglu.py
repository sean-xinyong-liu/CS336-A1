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
        if d_ff is None:
            d_ff = int(8/3 * d_model)
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        

    def forward(self, x: Tensor) -> Tensor:
        """Transform ``(..., d_model)`` into a tensor of the same shape."""
        return self.w2(silu(self.w1(x)) * self.w3(x))
        

