import argparse
import json
import pickle
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from cs336_basics.tokenizer.tokenizer import BPETokenizer
from cs336_basics.tokenizer.train_bpe import train_bpe

UINT16_DTYPE = np.dtype("<u2")
DEFAULT_WRITE_BUFFER_SIZE = 65_536


@dataclass(frozen=True)
class BPEExperimentConfig:
    dataset_name: str
    train_path: Path
    validation_path: Path
    output_dir: Path
    vocab_size: int
    special_tokens: tuple[str, ...] = ()


def _write_pickle(path: Path, value: Any) -> None:
    with open(path, "wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)


def _write_json(path: Path, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")


def _inspect_token(token: bytes) -> dict[str, str | int | None]:
    try:
        text = token.decode()
    except UnicodeDecodeError:
        text = None
    return {"hex": token.hex(), "text": text, "length": len(token)}


def serialize_tokenizer(
    output_dir: Path,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_pickle(output_dir / "vocab.pkl", vocab)
    _write_pickle(output_dir / "merges.pkl", merges)
    _write_json(
        output_dir / "vocab.json",
        {str(token_id): _inspect_token(token) for token_id, token in vocab.items()},
    )


def load_tokenizer(output_dir: Path, special_tokens: tuple[str, ...]) -> BPETokenizer:
    return BPETokenizer.from_files(
        output_dir / "vocab.pkl",
        output_dir / "merges.pkl",
        special_tokens=list(special_tokens),
    )


def encode_dataset(
    tokenizer: BPETokenizer,
    input_path: Path,
    output_path: Path,
    write_buffer_size: int = DEFAULT_WRITE_BUFFER_SIZE,
) -> dict[str, Any]:
    if write_buffer_size <= 0:
        raise ValueError("write_buffer_size must be positive")

    max_token_id = max(tokenizer.id_to_bytes, default=-1)
    if max_token_id > np.iinfo(np.uint16).max:
        raise ValueError(f"token ID {max_token_id} does not fit in uint16")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    token_count = 0
    buffer: list[int] = []
    start = time.perf_counter()

    try:
        with open(input_path, encoding="utf-8") as input_file, open(temporary_path, "wb") as output_file:
            for token_id in tokenizer.encode_iterable(input_file):
                buffer.append(token_id)
                if len(buffer) >= write_buffer_size:
                    np.asarray(buffer, dtype=UINT16_DTYPE).tofile(output_file)
                    token_count += len(buffer)
                    buffer.clear()

            if buffer:
                np.asarray(buffer, dtype=UINT16_DTYPE).tofile(output_file)
                token_count += len(buffer)

        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "dtype": "uint16",
        "byte_order": "little-endian",
        "token_count": token_count,
        "elapsed_seconds": time.perf_counter() - start,
    }


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def _base_manifest(config: BPEExperimentConfig) -> dict[str, Any]:
    return {
        "dataset": config.dataset_name,
        "vocab_size": config.vocab_size,
        "special_tokens": list(config.special_tokens),
        "artifacts": {
            "vocab": str(config.output_dir / "vocab.pkl"),
            "merges": str(config.output_dir / "merges.pkl"),
            "inspectable_vocab": str(config.output_dir / "vocab.json"),
            "train_token_ids": str(config.output_dir / "train_tokens.bin"),
            "validation_token_ids": str(config.output_dir / "validation_tokens.bin"),
        },
    }


def _load_manifest(config: BPEExperimentConfig) -> dict[str, Any]:
    manifest_path = config.output_dir / "manifest.json"
    if not manifest_path.is_file():
        return _base_manifest(config)

    with open(manifest_path, encoding="utf-8") as file:
        existing = json.load(file)
    return {**existing, **_base_manifest(config)}


def run_experiment(
    config: BPEExperimentConfig,
    stage: str = "all",
    num_processes: int | None = None,
    write_buffer_size: int = DEFAULT_WRITE_BUFFER_SIZE,
) -> dict[str, Any]:
    if stage not in {"train", "encode", "all"}:
        raise ValueError("stage must be one of: train, encode, all")
    if config.vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError("vocab_size must be at most 65,536 when token IDs use uint16")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(config)
    tokenizer: BPETokenizer

    if stage in {"train", "all"}:
        _require_files([config.train_path])
        start = time.perf_counter()
        vocab, merges = train_bpe(
            config.train_path,
            config.vocab_size,
            list(config.special_tokens),
            num_processes=num_processes,
        )
        training_seconds = time.perf_counter() - start
        serialize_tokenizer(config.output_dir, vocab, merges)
        tokenizer = BPETokenizer(vocab, merges, special_tokens=list(config.special_tokens))
        manifest["training"] = {
            "input": str(config.train_path),
            "elapsed_seconds": training_seconds,
            "actual_vocab_size": len(vocab),
            "num_merges": len(merges),
        }
    else:
        _require_files([config.output_dir / "vocab.pkl", config.output_dir / "merges.pkl"])
        tokenizer = load_tokenizer(config.output_dir, config.special_tokens)

    if stage in {"encode", "all"}:
        _require_files([config.train_path, config.validation_path])
        manifest["encoding"] = {
            "train": encode_dataset(
                tokenizer,
                config.train_path,
                config.output_dir / "train_tokens.bin",
                write_buffer_size,
            ),
            "validation": encode_dataset(
                tokenizer,
                config.validation_path,
                config.output_dir / "validation_tokens.bin",
                write_buffer_size,
            ),
        }

    _write_json(config.output_dir / "manifest.json", manifest)
    return manifest


def _parse_args(config: BPEExperimentConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Train and apply the {config.dataset_name} byte-level BPE tokenizer.")
    parser.add_argument("--stage", choices=("train", "encode", "all"), default="all")
    parser.add_argument("--train-path", type=Path, default=config.train_path)
    parser.add_argument("--validation-path", type=Path, default=config.validation_path)
    parser.add_argument("--output-dir", type=Path, default=config.output_dir)
    parser.add_argument("--num-processes", type=int, default=None)
    parser.add_argument("--write-buffer-size", type=int, default=DEFAULT_WRITE_BUFFER_SIZE)
    return parser.parse_args()


def run_cli(default_config: BPEExperimentConfig) -> None:
    args = _parse_args(default_config)
    config = replace(
        default_config,
        train_path=args.train_path,
        validation_path=args.validation_path,
        output_dir=args.output_dir,
    )
    manifest = run_experiment(
        config,
        stage=args.stage,
        num_processes=args.num_processes,
        write_buffer_size=args.write_buffer_size,
    )

    print(f"Completed {config.dataset_name} stage={args.stage}")
    print(f"Artifacts: {config.output_dir}")
    if "training" in manifest:
        training = manifest["training"]
        print(
            f"Vocabulary: {training['actual_vocab_size']} tokens, "
            f"{training['num_merges']} merges, {training['elapsed_seconds']:.2f}s"
        )
    if "encoding" in manifest:
        for split, result in manifest["encoding"].items():
            print(f"{split}: {result['token_count']} tokens, {result['elapsed_seconds']:.2f}s")
