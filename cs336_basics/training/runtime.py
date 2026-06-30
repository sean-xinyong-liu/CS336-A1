"""Small runtime helpers for the training loop."""

from pathlib import Path

import torch

from cs336_basics.training.checkpointing import save_checkpoint


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def save_checkpoint_if_configured(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    checkpoint_path: Path | None,
) -> None:
    if checkpoint_path is None:
        return

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, optimizer, iteration, checkpoint_path)
