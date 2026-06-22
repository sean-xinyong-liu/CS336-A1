from pathlib import Path

from cs336_basics.tokenizer.bpe_experiment import BPEExperimentConfig, run_cli

CONFIG = BPEExperimentConfig(
    dataset_name="TinyStories",
    train_path=Path("data/TinyStoriesV2-GPT4-train.txt"),
    validation_path=Path("data/TinyStoriesV2-GPT4-valid.txt"),
    output_dir=Path("artifacts/tokenizer/tinystories_bpe_10k"),
    vocab_size=10_000,
    special_tokens=("<|endoftext|>",),
)


if __name__ == "__main__":
    run_cli(CONFIG)
