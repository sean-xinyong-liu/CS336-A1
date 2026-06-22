from pathlib import Path

from cs336_basics.tokenizer.bpe_experiment import BPEExperimentConfig, run_cli

CONFIG = BPEExperimentConfig(
    dataset_name="OpenWebText",
    train_path=Path("data/owt_train.txt"),
    validation_path=Path("data/owt_valid.txt"),
    output_dir=Path("artifacts/tokenizer/owt_bpe_32k"),
    vocab_size=32_000,
    special_tokens=("<|endoftext|>",),
)


if __name__ == "__main__":
    run_cli(CONFIG)
