"""Checkpoint save/load helpers."""

import os
from typing import IO, BinaryIO

import torch


MODEL_STATE_KEY = "model_state"
OPTIMIZER_STATE_KEY = "optimizer_state"
ITERATION_KEY = "iteration"


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Serialize model state, optimizer state, and iteration."""
    checkpoint = {
        MODEL_STATE_KEY: model.state_dict(),
        OPTIMIZER_STATE_KEY: optimizer.state_dict(),
        ITERATION_KEY: iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore model and optimizer state, returning the saved iteration."""
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint[MODEL_STATE_KEY])
    optimizer.load_state_dict(checkpoint[OPTIMIZER_STATE_KEY])
    return checkpoint[ITERATION_KEY]
