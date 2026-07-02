"""Command-line configuration for language-model training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

DEFAULT_DATA_DTYPE = "uint16"
DEFAULT_ADAM_BETA1 = 0.9
DEFAULT_ADAM_BETA2 = 0.999
DEFAULT_ADAM_EPS = 1e-8
DEFAULT_NORM_EPS = 1e-5
DEFAULT_ROPE_THETA = 10_000.0


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_dtype(name: str) -> torch.dtype:
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtypes[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")

    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path)
    parser.add_argument("--data-dtype", default=DEFAULT_DATA_DTYPE)
    parser.add_argument("--data-format", choices=("auto", "raw", "npy"), default="auto")

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--rope-theta", type=float, default=DEFAULT_ROPE_THETA)
    parser.add_argument("--norm-eps", type=float, default=DEFAULT_NORM_EPS)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=1_000)
    parser.add_argument("--cosine-cycle-iters", type=int, default=10_000)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=DEFAULT_ADAM_BETA1)
    parser.add_argument("--beta2", type=float, default=DEFAULT_ADAM_BETA2)
    parser.add_argument("--adam-eps", type=float, default=DEFAULT_ADAM_EPS)
    parser.add_argument("--max-grad-norm", type=float)

    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--resume-from", type=Path)

    parser.add_argument("--device", default=default_device())
    parser.add_argument("--model-dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--log-file", type=Path)

    return parser.parse_args()
