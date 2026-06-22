import os
from collections import Counter
from collections.abc import Iterable

import regex as re

PRETOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def split_specials(chunk: bytes, special_token_bytes: list[bytes]) -> list[bytes]:
    if not special_token_bytes:
        return [chunk]

    special_token_bytes = sorted(special_token_bytes, key=len, reverse=True)
    pattern = b"|".join(re.escape(token) for token in special_token_bytes)
    return re.split(pattern, chunk)


def pretokenize_bytes(chunk: bytes, special_token_bytes: list[bytes]) -> Counter[str]:
    pretoken_counts: Counter[str] = Counter()

    for part in split_specials(chunk, special_token_bytes):
        if not part:
            continue

        text = part.decode("utf-8", errors="ignore")
        for match in PRETOKEN_PATTERN.finditer(text):
            pretoken_counts[match.group(0)] += 1

    return pretoken_counts


def merge_counters(counters: Iterable[Counter[str]]) -> Counter[str]:
    merged_counts: Counter[str] = Counter()
    for counts in counters:
        merged_counts.update(counts)
    return merged_counts


def find_chunk_boundaries(
    input_path: str | os.PathLike,
    desired_num_chunks: int,
    split_token: bytes,
) -> list[int]:
    assert desired_num_chunks > 0
    assert split_token

    with open(input_path, "rb") as file:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        if file_size == 0:
            return [0]

        chunk_size = file_size // desired_num_chunks
        boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        boundaries[-1] = file_size

        mini_chunk_size = 4096
        for boundary_index in range(1, len(boundaries) - 1):
            initial_position = boundaries[boundary_index]
            file.seek(initial_position)

            while True:
                mini_chunk = file.read(mini_chunk_size)
                if mini_chunk == b"":
                    boundaries[boundary_index] = file_size
                    break

                found_at = mini_chunk.find(split_token)
                if found_at != -1:
                    boundaries[boundary_index] = initial_position + found_at
                    break

                initial_position += mini_chunk_size

    return sorted(set(boundaries))


def _pretokenize_file_chunk(args: tuple[str | os.PathLike, int, int, list[bytes]]) -> Counter[str]:
    input_path, start, end, special_token_bytes = args
    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start)
    return pretokenize_bytes(chunk, special_token_bytes)


def pretokenize_file(
    input_path: str | os.PathLike,
    special_tokens: list[str],
) -> Counter[str]:
    special_token_bytes = [token.encode() for token in special_tokens]
    with open(input_path, "rb") as file:
        return pretokenize_bytes(file.read(), special_token_bytes)


def pretokenize_file_parallel(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> Counter[str]:
    special_token_bytes = [token.encode() for token in special_tokens]
    process_count = num_processes or min(os.cpu_count() or 1, 8)

    if process_count <= 1 or not special_token_bytes:
        return pretokenize_file(input_path, special_tokens)

    split_token = max(special_token_bytes, key=len)
    boundaries = find_chunk_boundaries(input_path, process_count, split_token)
    chunks = [(start, end) for start, end in zip(boundaries, boundaries[1:]) if end > start]

    if len(chunks) <= 1:
        return pretokenize_file(input_path, special_tokens)

    import multiprocessing

    tasks = [(input_path, start, end, special_token_bytes) for start, end in chunks]
    with multiprocessing.Pool(processes=min(process_count, len(tasks))) as pool:
        return merge_counters(pool.map(_pretokenize_file_chunk, tasks))
