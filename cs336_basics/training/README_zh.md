# Transformer LM 训练

本包包含 CS336 Assignment 1 的训练工具，以及一个命令行训练脚本。训练脚本把模型、loss、optimizer、数据加载、学习率调度、梯度裁剪、checkpoint 和日志串在一起。

## 模块结构

- `data.py`
  - 从 token 数组中随机采样连续的语言模型 batch。
  - 返回输入 token ID 和 next-token target，并放到指定 PyTorch device。
- `losses.py`
  - 在最后一维词表维度上实现数值稳定的 cross-entropy。
- `optimizer.py`
  - 以 `torch.optim.Optimizer` 子类形式实现 AdamW。
- `scheduler.py`
  - 实现 linear warmup 后接 cosine learning-rate decay。
- `clipping.py`
  - 按全局 L2 norm 裁剪所有参数梯度。
- `checkpointing.py`
  - 保存和加载模型状态、optimizer 状态和训练 iteration。
- `config.py`
  - 定义命令行参数、默认超参、device 选择和模型 dtype 映射。
- `datasets.py`
  - 使用 `np.memmap` 加载 raw token 数组，或使用
    `np.load(..., mmap_mode="r")` 加载 `.npy` 数组。
  - 执行轻量数据集 sanity check。
- `evaluation.py`
  - 基于采样 batch 估计验证 loss。
- `monitoring.py`
  - 处理 console logging 和可选 Weights and Biases logging。
- `runtime.py`
  - 放置学习率赋值、checkpoint 路径处理等小型运行时工具。
- `train_lm.py`
  - 构造模型和 optimizer，并运行主训练循环。

## 训练数据

训练脚本期望输入已经 tokenized 的数据集：一维整数数组，每个元素是一个 token ID。对于大数据集，优先使用 memory-mapped 文件，避免把整个数组读入内存。

Tokenizer 实验脚本会写出 raw little-endian `uint16` token 数组：

```bash
uv run python -m cs336_basics.tokenizer.train_bpe_tinystories --stage all --num-processes 8
```

这些数组可以直接传给训练脚本：

```text
artifacts/tokenizer/tinystories_bpe_10k/train_tokens.bin
artifacts/tokenizer/tinystories_bpe_10k/validation_tokens.bin
```

对于 raw binary 文件，使用 `--data-dtype` 指定写入数组时的 dtype，默认是 `uint16`。对于 `.npy` 文件，脚本使用 `np.load(..., mmap_mode="r")`，并从文件中读取 dtype。

脚本会检查每个 split 的一小段前缀；如果发现 token ID 不在 `[0, vocab_size)` 范围内，会直接报错。

## 基本用法

从 raw token 数组训练一个小型 Transformer LM：

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

对 `.npy` 数组使用 `--data-format npy`。默认 `--data-format auto` 会把 `.npy` 后缀按 `.npy` 加载，其余路径按 raw binary 数组加载。

## 重要参数

模型配置：

- `--vocab-size`
- `--context-length`
- `--d-model`
- `--num-layers`
- `--num-heads`
- `--d-ff`
- `--rope-theta`
- `--norm-eps`

Optimizer 和 scheduler：

- `--learning-rate`
- `--min-learning-rate`
- `--warmup-iters`
- `--cosine-cycle-iters`
- `--weight-decay`
- `--beta1`
- `--beta2`
- `--adam-eps`
- `--max-grad-norm`

运行配置：

- `--batch-size`
- `--max-iters`
- `--device`
- `--model-dtype`
- `--seed`

日志和验证：

- `--log-every`
- `--eval-every`
- `--eval-iters`
- `--wandb-project`
- `--wandb-run-name`
- `--wandb-mode`

Checkpoint：

- `--checkpoint-path`
- `--checkpoint-every`
- `--resume-from`

## 训练循环

每个训练 iteration 执行：

1. 使用 warmup + cosine decay 计算当前学习率。
2. 从 memory-mapped 训练 token 数组中随机采样 batch。
3. 执行 Transformer LM 前向传播，得到 logits。
4. 对 next-token targets 计算平均 cross-entropy。
5. 反向传播。
6. 如果设置了 `--max-grad-norm`，执行梯度裁剪。
7. 执行一次 AdamW optimizer step。
8. 周期性记录指标、在验证集上评估，并保存 checkpoint。

验证 loss 通过平均 `--eval-iters` 个随机验证 batch 得到。这样评估成本较低，不需要每次扫描完整验证集。

## Checkpoint 和恢复训练

Checkpoint 包含：

- `model.state_dict()`
- `optimizer.state_dict()`
- 已完成的训练 iteration

周期性保存：

```bash
--checkpoint-path checkpoints/latest.pt --checkpoint-every 1000
```

从 checkpoint 恢复：

```bash
uv run python -m cs336_basics.training.train_lm \
  --train-data artifacts/tokenizer/tinystories_bpe_10k/train_tokens.bin \
  --val-data artifacts/tokenizer/tinystories_bpe_10k/validation_tokens.bin \
  --vocab-size 10000 \
  --checkpoint-path checkpoints/latest.pt \
  --resume-from checkpoints/latest.pt
```

恢复时会先加载模型权重、optimizer 状态和保存的 iteration，然后训练循环从该 iteration 继续。

## Weights and Biases

Console logging 始终启用。如果还想记录到 Weights and Biases：

```bash
--wandb-project cs336-assignment1 --wandb-run-name tinystories-small
```

本地或离线记录：

```bash
--wandb-project cs336-assignment1 --wandb-mode offline
```

省略 `--wandb-project`，或设置 `--wandb-mode disabled`，即可避免初始化 Weights and Biases。

## 快速烟测

创建很小的 raw token 数组，并在 CPU 上跑两个 iteration：

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

预期输出包含 `iter`、`lr`、`train_loss` 和 `val_loss`。

## 测试

训练工具由以下作业测试覆盖：

```bash
uv run pytest -k "test_get_batch or test_cross_entropy or test_adamw or test_get_lr_cosine_schedule or test_gradient_clipping or test_checkpointing"
```
