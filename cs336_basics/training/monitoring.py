"""Console and optional Weights and Biases logging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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


def log_metrics(run: Any | None, metrics: dict[str, float | int], log_file: Path | None = None) -> None:
    fields = []
    for key, value in metrics.items():
        fields.append(f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}")

    print(" ".join(fields), flush=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as file:
            json.dump(metrics, file)
            file.write("\n")
    if run is not None:
        run.log(metrics, step=int(metrics["iter"]))
