"""Data-loading helpers for language-model training."""

import numpy as np
import numpy.typing as npt
import torch



def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample language-model inputs and next-token targets from a token array."""
    max_start = len(dataset) - context_length
    starts = np.random.randint(low=0, high=max_start, size=batch_size)
    offsets = np.arange(context_length)

    input_idx = starts[:, None] + offsets[None, :]
    target_idx = input_idx + 1
    inputs = torch.as_tensor(dataset[input_idx], dtype=torch.long, device=device)
    targets = torch.as_tensor(dataset[target_idx], dtype=torch.long, device=device)

    return inputs, targets
