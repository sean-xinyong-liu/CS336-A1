# AGENTS.md

## Project Overview

This repository contains an implementation of CS336 Assignment 1: Basics.

The project uses Python and is managed with `uv`. Source code lives under
`cs336_basics/`, and the assignment test suite lives under `tests/`.

## Development Principles

Implementations should prioritize correctness and clarity before performance
optimization. Keep changes focused on the assignment component being worked on,
and avoid unrelated refactors.

The public interfaces expected by the assignment tests are defined in
`tests/adapters.py`. Implementation code should be connected through those
adapters rather than changing test expectations.

## Current Progress

As of 2026-07-02, the implementation work for the assignment is effectively
complete except for the final experiments section.

Completed components:
- BPE tokenizer training and tokenization.
- Transformer model modules, including attention, RoPE, RMSNorm, SwiGLU,
  Transformer blocks, and TransformerLM.
- Training utilities, including batching, cross-entropy, AdamW, learning-rate
  scheduling, gradient clipping, checkpoint save/load, evaluation helpers, and
  the training entrypoint.
- Experiment logging in the training entrypoint records step, train loss,
  optional validation loss, wall-clock seconds, tokens processed, throughput,
  and can write JSONL via `--log-file`.
- Decoding utilities in `cs336_basics/decoding.py`, wired through
  `tests/adapters.py` with `run_generate_token_ids` and
  `run_generate_completion`.

Recent verification:
- `uv run pytest` passed with 50 passed and 2 skipped.
- `uv run ruff check cs336_basics/decoding.py tests/test_decoding.py` passed.

Remaining work:
- Run and document the final experimental section.
