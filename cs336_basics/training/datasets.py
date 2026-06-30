"""Dataset loading and validation helpers for training."""

from pathlib import Path

import numpy as np

TOKEN_ID_CHECK_PREFIX = 10_000


def load_tokens(path: Path, dtype: str, data_format: str) -> np.ndarray:
    if data_format == "auto":
        data_format = "npy" if path.suffix == ".npy" else "raw"

    if data_format == "npy":
        return np.load(path, mmap_mode="r")

    return np.memmap(path, dtype=np.dtype(dtype), mode="r")


def validate_tokens(name: str, tokens: np.ndarray, context_length: int, vocab_size: int) -> None:
    if len(tokens) <= context_length:
        raise ValueError(f"{name} dataset must contain more than context_length tokens.")

    sample = tokens[: min(len(tokens), TOKEN_ID_CHECK_PREFIX)]
    if sample.min() < 0 or sample.max() >= vocab_size:
        raise ValueError(f"{name} dataset contains token IDs outside [0, vocab_size).")
