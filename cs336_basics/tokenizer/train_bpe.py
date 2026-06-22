import os

from cs336_basics.tokenizer.bpe import train_bpe_from_word_counts
from cs336_basics.tokenizer.pretokenization import pretokenize_file_parallel


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_processes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size must include all 256 byte tokens and special tokens")

    pretoken_counts = pretokenize_file_parallel(
        input_path,
        special_tokens,
        num_processes=num_processes,
    )
    return train_bpe_from_word_counts(pretoken_counts, vocab_size, special_tokens)
