import argparse
import cProfile
import io
import json
import os
import pickle
import pstats
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import psutil

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs336_basics.tokenizer.train_bpe import train_bpe  # noqa: E402

DEFAULT_INPUT_PATH = Path("data/TinyStoriesV2-GPT4-train.txt")
DEFAULT_OUTPUT_DIR = Path("artifacts/tokenizer/tinystories_bpe_10k")
DEFAULT_VOCAB_SIZE = 10_000
DEFAULT_SPECIAL_TOKEN = "<|endoftext|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer on TinyStories.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Path to the TinyStories train file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for serialized vocab, merges, profile, and summary files.",
    )
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE, help="Maximum vocabulary size.")
    parser.add_argument(
        "--special-token",
        action="append",
        default=None,
        help="Special token to add. Can be passed multiple times.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of pre-tokenization worker processes. Defaults to train_bpe's local CPU heuristic.",
    )
    parser.add_argument(
        "--profile-top-n",
        type=int,
        default=25,
        help="Number of cumulative-time profile entries to write to profile.txt and summary.json.",
    )
    return parser.parse_args()


def ensure_input_exists(input_path: Path) -> None:
    if input_path.exists():
        return

    raise FileNotFoundError(
        f"Missing TinyStories training file: {input_path}\n"
    )


def bytes_to_inspectable(token: bytes) -> dict[str, Any]:
    decoded = None

    try:
        decoded_text = token.decode("utf-8")
    except UnicodeDecodeError:
        decoded_text = None

    if decoded_text is not None:
        decoded = decoded_text

    return {
        "hex": token.hex(),
        "utf8": decoded,
        "length": len(token),
    }


def write_pickle(path: Path, value: Any) -> None:
    with open(path, "wb") as file:
        pickle.dump(value, file)


def write_vocab_json(path: Path, vocab: dict[int, bytes]) -> None:
    serializable_vocab = {str(token_id): bytes_to_inspectable(token) for token_id, token in vocab.items()}
    with open(path, "w", encoding="utf-8") as file:
        json.dump(serializable_vocab, file, indent=2, ensure_ascii=False)
        file.write("\n")


def get_profile_top_entries(stats: pstats.Stats, limit: int) -> list[dict[str, Any]]:
    entries = []
    for func, stat in sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:limit]:
        primitive_calls, total_calls, total_time, cumulative_time, _callers = stat
        filename, line_number, function_name = func
        entries.append(
            {
                "function": f"{filename}:{line_number}({function_name})",
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "total_time_seconds": total_time,
                "cumulative_time_seconds": cumulative_time,
            }
        )
    return entries


def write_profile_outputs(
    output_dir: Path,
    profiler: cProfile.Profile,
    profile_top_n: int,
) -> list[dict[str, Any]]:
    profile_path = output_dir / "profile.prof"
    text_profile_path = output_dir / "profile.txt"

    profiler.dump_stats(profile_path)

    stats_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream).strip_dirs().sort_stats("cumulative")
    stats.print_stats(profile_top_n)

    with open(text_profile_path, "w", encoding="utf-8") as file:
        file.write(stats_stream.getvalue())

    return get_profile_top_entries(stats, profile_top_n)


def describe_longest_token(vocab: dict[int, bytes]) -> dict[str, Any]:
    token_id, token = max(vocab.items(), key=lambda item: len(item[1]))
    description = bytes_to_inspectable(token)
    description["token_id"] = token_id
    return description


def train_with_measurements(args: argparse.Namespace) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]], dict[str, Any]]:
    special_tokens = args.special_token or [DEFAULT_SPECIAL_TOKEN]
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss

    profiler = cProfile.Profile()
    tracemalloc.start()
    start = time.perf_counter()

    profiler.enable()
    vocab, merges = train_bpe(
        args.input,
        args.vocab_size,
        special_tokens,
        num_processes=args.num_processes,
    )
    profiler.disable()

    elapsed_seconds = time.perf_counter() - start
    current_allocated, peak_allocated = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = process.memory_info().rss

    profile_top_entries = write_profile_outputs(args.output_dir, profiler, args.profile_top_n)

    measurements = {
        "elapsed_seconds": elapsed_seconds,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "rss_delta_bytes": rss_after - rss_before,
        "tracemalloc_current_bytes": current_allocated,
        "tracemalloc_peak_bytes": peak_allocated,
        "profile_top_entries": profile_top_entries,
    }
    return vocab, merges, measurements


def write_summary(
    path: Path,
    args: argparse.Namespace,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    measurements: dict[str, Any],
) -> dict[str, Any]:
    special_tokens = args.special_token or [DEFAULT_SPECIAL_TOKEN]
    summary = {
        "config": {
            "input": str(args.input),
            "output_dir": str(args.output_dir),
            "vocab_size": args.vocab_size,
            "special_tokens": special_tokens,
            "num_processes": args.num_processes,
        },
        "results": {
            "actual_vocab_size": len(vocab),
            "num_merges": len(merges),
            "longest_token": describe_longest_token(vocab),
        },
        "measurements": measurements,
        "outputs": {
            "vocab_pickle": str(args.output_dir / "vocab.pkl"),
            "merges_pickle": str(args.output_dir / "merges.pkl"),
            "vocab_json": str(args.output_dir / "vocab.json"),
            "profile_binary": str(args.output_dir / "profile.prof"),
            "profile_text": str(args.output_dir / "profile.txt"),
            "summary": str(path),
        },
    }

    with open(path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")

    return summary


def format_bytes(num_bytes: int) -> str:
    mib = num_bytes / (1024 * 1024)
    return f"{mib:.2f} MiB"


def print_report(summary: dict[str, Any]) -> None:
    measurements = summary["measurements"]
    longest_token = summary["results"]["longest_token"]
    top_profile_entry = measurements["profile_top_entries"][0] if measurements["profile_top_entries"] else None

    print("Training complete.")
    print(f"Time: {measurements['elapsed_seconds']:.2f}s")
    print(
        "Memory: "
        f"RSS delta {format_bytes(measurements['rss_delta_bytes'])}, "
        f"tracemalloc peak {format_bytes(measurements['tracemalloc_peak_bytes'])}"
    )
    print(f"Vocab: {summary['results']['actual_vocab_size']} tokens, {summary['results']['num_merges']} merges")
    print(f"Longest token: id={longest_token['token_id']}, length={longest_token['length']}")
    print(f"Longest token text: {longest_token['utf8']!r}")
    if top_profile_entry is not None:
        print(f"Slowest cumulative function: {top_profile_entry['function']}")
    print(f"Outputs written to: {summary['config']['output_dir']}")


def main() -> None:
    args = parse_args()
    ensure_input_exists(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vocab, merges, measurements = train_with_measurements(args)

    write_pickle(args.output_dir / "vocab.pkl", vocab)
    write_pickle(args.output_dir / "merges.pkl", merges)
    write_vocab_json(args.output_dir / "vocab.json", vocab)
    summary = write_summary(args.output_dir / "summary.json", args, vocab, merges, measurements)

    print_report(summary)


if __name__ == "__main__":
    main()
