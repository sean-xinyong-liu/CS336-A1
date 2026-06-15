import os
from collections import Counter
from collections.abc import Iterable

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def split_specials(chunk: bytes, specials: list[bytes]) -> list[bytes]:
    if not specials:
        return [chunk]

    specials = sorted(specials, key=len, reverse=True)
    pattern = b"|".join(re.escape(token) for token in specials)
    return re.split(pattern, chunk)


def pretokenize_bytes(chunk: bytes, specials: list[bytes]) -> Counter[str]:
    counter: Counter[str] = Counter()

    for part in split_specials(chunk, specials):
        if not part:
            continue

        text = part.decode("utf-8", errors="ignore")
        for match in re.finditer(PAT, text):
            counter[match.group(0)] += 1

    return counter


def merge_counters(counters: Iterable[Counter[str]]) -> Counter[str]:
    merged: Counter[str] = Counter()
    for counter in counters:
        merged.update(counter)
    return merged


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
    input_path, start, end, specials = args
    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start)
    return pretokenize_bytes(chunk, specials)


def pretokenize_file(
    input_path: str | os.PathLike,
    special_tokens: list[str],
) -> Counter[str]:
    specials = [token.encode("utf-8") for token in special_tokens]
    with open(input_path, "rb") as file:
        return pretokenize_bytes(file.read(), specials)


def pretokenize_file_parallel(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> Counter[str]:
    specials = [token.encode("utf-8") for token in special_tokens]
    process_count = num_processes or min(os.cpu_count() or 1, 8)

    if process_count <= 1 or not specials:
        return pretokenize_file(input_path, special_tokens)

    split_token = max(specials, key=len)
    boundaries = find_chunk_boundaries(input_path, process_count, split_token)
    chunks = [(start, end) for start, end in zip(boundaries, boundaries[1:]) if end > start]

    if len(chunks) <= 1:
        return pretokenize_file(input_path, special_tokens)

    import multiprocessing

    tasks = [(input_path, start, end, specials) for start, end in chunks]
    with multiprocessing.Pool(processes=min(process_count, len(tasks))) as pool:
        return merge_counters(pool.map(_pretokenize_file_chunk, tasks))
