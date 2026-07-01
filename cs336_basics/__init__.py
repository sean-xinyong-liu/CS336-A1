import importlib.metadata

from cs336_basics.decoding import apply_top_p, generate_completion, generate_token_ids, sample_next_token

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass

__all__ = [
    "apply_top_p",
    "generate_completion",
    "generate_token_ids",
    "sample_next_token",
]
