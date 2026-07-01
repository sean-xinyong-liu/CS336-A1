# Transformer LM Training

This package contains the training utilities for CS336 Assignment 1 and a
command-line script that puts the model, loss, optimizer, data loading,
scheduling, gradient clipping, checkpointing, and logging together.

## Module Layout

- `batching.py`
  - Samples random contiguous language-model batches from a token array.
  - Returns input IDs and next-token targets on the requested PyTorch device.
- `losses.py`
  - Implements numerically stable cross-entropy over the final vocabulary
    dimension.
- `optimizer.py`
  - Implements AdamW as a `torch.optim.Optimizer` subclass.
- `scheduler.py`
  - Implements linear warmup followed by cosine learning-rate decay.
- `clipping.py`
  - Clips all parameter gradients by their global L2 norm.
- `checkpointing.py`
  - Saves and loads model state, optimizer state, and training iteration.
- `config.py`
  - Defines command-line arguments, default hyperparameters, device selection,
    and model dtype mapping.
- `token_data.py`
  - Loads raw token arrays with `np.memmap` or `.npy` arrays with
    `np.load(..., mmap_mode="r")`.
  - Performs lightweight dataset sanity checks.
- `evaluation.py`
  - Estimates validation loss over sampled batches.
- `monitoring.py`
  - Handles console logging and optional Weights and Biases logging.
- `runtime.py`
  - Contains small runtime helpers such as learning-rate assignment and
    checkpoint-path handling.
- `train_lm.py`
  - Builds the model and optimizer, then runs the main training loop.

## Training Data

The training script expects tokenized datasets: one-dimensional integer arrays
of token IDs. For large datasets, prefer memory-mapped files so the whole array
does not need to be loaded into RAM.

The tokenizer experiment scripts write raw little-endian `uint16` token arrays:

```bash
uv run python -m cs336_basics.tokenizer.train_bpe_tinystories --stage all --num-processes 8
```

Those arrays can be passed directly to the training script:

```text
artifacts/tokenizer/tinystories_bpe_10k/train_tokens.bin
artifacts/tokenizer/tinystories_bpe_10k/validation_tokens.bin
```

For raw binary files, set `--data-dtype` to the dtype used when writing the
array. The default is `uint16`. For `.npy` files, the script uses
`np.load(..., mmap_mode="r")` and reads the dtype from the file.

The script checks a small prefix of each split and raises an error if it sees
token IDs outside `[0, vocab_size)`.

## Basic Usage

Train a small Transformer LM from raw token arrays:

```bash
uv run python -m cs336_basics.training.train_lm \
  --train-data artifacts/tokenizer/tinystories_bpe_10k/train_tokens.bin \
  --val-data artifacts/tokenizer/tinystories_bpe_10k/validation_tokens.bin \
  --data-dtype uint16 \
  --vocab-size 10000 \
  --context-length 256 \
  --d-model 512 \
  --num-layers 6 \
  --num-heads 8 \
  --d-ff 2048 \
  --batch-size 32 \
  --max-iters 10000 \
  --learning-rate 3e-4 \
  --min-learning-rate 3e-5 \
  --warmup-iters 1000 \
  --cosine-cycle-iters 10000 \
  --max-grad-norm 1.0 \
  --checkpoint-path checkpoints/tinystories_latest.pt
```

Use `--data-format npy` for `.npy` arrays. With the default
`--data-format auto`, paths ending in `.npy` are loaded as `.npy`; all other
paths are treated as raw binary arrays.

## Important Options

Model configuration:

- `--vocab-size`
- `--context-length`
- `--d-model`
- `--num-layers`
- `--num-heads`
- `--d-ff`
- `--rope-theta`
- `--norm-eps`

Optimizer and schedule:

- `--learning-rate`
- `--min-learning-rate`
- `--warmup-iters`
- `--cosine-cycle-iters`
- `--weight-decay`
- `--beta1`
- `--beta2`
- `--adam-eps`
- `--max-grad-norm`

Runtime:

- `--batch-size`
- `--max-iters`
- `--device`
- `--model-dtype`
- `--seed`

Logging and evaluation:

- `--log-every`
- `--eval-every`
- `--eval-iters`
- `--wandb-project`
- `--wandb-run-name`
- `--wandb-mode`

Checkpointing:

- `--checkpoint-path`
- `--checkpoint-every`
- `--resume-from`

## Training Loop

Each training iteration runs:

1. Compute the current learning rate with warmup plus cosine decay.
2. Sample a random batch from the memory-mapped training token array.
3. Run the Transformer LM forward pass to produce logits.
4. Compute average cross-entropy against next-token targets.
5. Backpropagate.
6. Clip gradients if `--max-grad-norm` is set.
7. Take one AdamW optimizer step.
8. Periodically log metrics, evaluate on validation data, and save a
   checkpoint.

Validation loss is estimated by averaging `--eval-iters` random validation
batches. This keeps evaluation cheap and avoids scanning the full validation
set every time.

## Checkpoints and Resuming

Checkpoints contain:

- `model.state_dict()`
- `optimizer.state_dict()`
- the completed training iteration

Save periodically with:

```bash
--checkpoint-path checkpoints/latest.pt --checkpoint-every 1000
```

Resume from a checkpoint:

```bash
uv run python -m cs336_basics.training.train_lm \
  --train-data artifacts/tokenizer/tinystories_bpe_10k/train_tokens.bin \
  --val-data artifacts/tokenizer/tinystories_bpe_10k/validation_tokens.bin \
  --vocab-size 10000 \
  --checkpoint-path checkpoints/latest.pt \
  --resume-from checkpoints/latest.pt
```

When resuming, model weights, optimizer state, and the saved iteration are
restored before the training loop continues.

## Weights and Biases

Console logging is always enabled. To also log to Weights and Biases:

```bash
--wandb-project cs336-assignment1 --wandb-run-name tinystories-small
```

For local/offline logging:

```bash
--wandb-project cs336-assignment1 --wandb-mode offline
```

Set `--wandb-mode disabled` or omit `--wandb-project` to avoid initializing
Weights and Biases.

## Quick Smoke Test

Create tiny raw token arrays and run two iterations on CPU:

```bash
uv run python -c "import numpy as np; (np.arange(256, dtype=np.uint16) % 16).tofile('/tmp/cs336_train.bin'); (np.arange(128, dtype=np.uint16) % 16).tofile('/tmp/cs336_val.bin')"

uv run python -m cs336_basics.training.train_lm \
  --train-data /tmp/cs336_train.bin \
  --val-data /tmp/cs336_val.bin \
  --data-dtype uint16 \
  --vocab-size 16 \
  --context-length 8 \
  --d-model 16 \
  --num-layers 1 \
  --num-heads 4 \
  --d-ff 32 \
  --batch-size 2 \
  --max-iters 2 \
  --learning-rate 1e-3 \
  --min-learning-rate 1e-4 \
  --warmup-iters 1 \
  --cosine-cycle-iters 2 \
  --eval-iters 1 \
  --log-every 1 \
  --eval-every 1 \
  --checkpoint-path /tmp/cs336_train_test.pt \
  --device cpu
```

Expected output includes `iter`, `lr`, `train_loss`, and `val_loss` lines.

## Tests

The training utilities are covered by the assignment tests:

```bash
uv run pytest -k "test_get_batch or test_cross_entropy or test_adamw or test_get_lr_cosine_schedule or test_gradient_clipping or test_checkpointing"
```
