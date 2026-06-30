"""Console and optional Weights and Biases logging."""

from __future__ import annotations

import argparse
from typing import Any


def init_wandb(args: argparse.Namespace) -> Any | None:
    if args.wandb_project is None or args.wandb_mode == "disabled":
        return None

    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=vars(args),
    )


def log_metrics(run: Any | None, metrics: dict[str, float | int]) -> None:
    fields = []
    for key, value in metrics.items():
        fields.append(f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}")

    print(" ".join(fields), flush=True)
    if run is not None:
        run.log(metrics, step=int(metrics["iter"]))
