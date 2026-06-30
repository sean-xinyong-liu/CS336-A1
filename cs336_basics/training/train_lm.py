"""Command-line training loop for the Transformer language model."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.training import (
    AdamW,
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
)
from cs336_basics.training.config import parse_args, torch_dtype
from cs336_basics.training.datasets import load_tokens, validate_tokens
from cs336_basics.training.evaluation import estimate_loss
from cs336_basics.training.monitoring import init_wandb, log_metrics
from cs336_basics.training.runtime import save_checkpoint_if_configured, set_lr


def build_model(args: argparse.Namespace, device: torch.device) -> TransformerLM:
    return TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        eps=args.norm_eps,
        device=device,
        dtype=torch_dtype(args.model_dtype),
    )


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )


def current_lr(args: argparse.Namespace, iteration: int) -> float:
    return get_lr_cosine_schedule(
        it=iteration,
        max_learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.cosine_cycle_iters,
    )


def train_step(
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    tokens: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    inputs, targets = get_batch(tokens, args.batch_size, args.context_length, device)
    loss = cross_entropy(model(inputs), targets)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if args.max_grad_norm is not None:
        gradient_clipping(model.parameters(), args.max_grad_norm)
    optimizer.step()

    return loss.item()


def maybe_log(
    model: TransformerLM,
    val_tokens: np.ndarray | None,
    train_loss: float,
    lr: float,
    iteration: int,
    args: argparse.Namespace,
    device: torch.device,
    run: object | None,
) -> None:
    should_log = iteration == 1 or iteration % args.log_every == 0
    should_eval = val_tokens is not None and iteration % args.eval_every == 0
    if not should_log and not should_eval:
        return

    metrics: dict[str, float | int] = {
        "iter": iteration,
        "lr": lr,
        "train_loss": train_loss,
    }
    if should_eval and val_tokens is not None:
        metrics["val_loss"] = estimate_loss(
            model=model,
            tokens=val_tokens,
            batch_size=args.batch_size,
            context_length=args.context_length,
            eval_iters=args.eval_iters,
            device=device,
        )
    log_metrics(run, metrics)


def train(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_tokens = load_tokens(args.train_data, args.data_dtype, args.data_format)
    val_tokens = load_tokens(args.val_data, args.data_dtype, args.data_format) if args.val_data is not None else None
    validate_tokens("train", train_tokens, args.context_length, args.vocab_size)
    if val_tokens is not None:
        validate_tokens("val", val_tokens, args.context_length, args.vocab_size)

    device = torch.device(args.device)
    model = build_model(args, device)
    optimizer = build_optimizer(args, model)

    start_iter = load_checkpoint(args.resume_from, model, optimizer) if args.resume_from is not None else 0
    run = init_wandb(args)
    model.train()

    for it in range(start_iter, args.max_iters):
        lr = current_lr(args, it)
        set_lr(optimizer, lr)

        train_loss = train_step(model, optimizer, train_tokens, args, device)
        iteration = it + 1

        maybe_log(model, val_tokens, train_loss, lr, iteration, args, device, run)
        if args.checkpoint_path is not None and iteration % args.checkpoint_every == 0:
            save_checkpoint_if_configured(model, optimizer, iteration, args.checkpoint_path)

    save_checkpoint_if_configured(model, optimizer, args.max_iters, args.checkpoint_path)
    if run is not None:
        run.finish()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
