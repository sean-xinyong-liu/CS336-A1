import pickle
from collections.abc import Iterable, Iterator
from functools import lru_cache
from os import PathLike
from typing import Self

import regex as re

from cs336_basics.tokenizer.bpe import merge_word
from cs336_basics.tokenizer.pretokenization import PRETOKEN_PATTERN


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.id_to_bytes = dict(vocab)
        self.bytes_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}

        self.merges = list(merges)
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

        self.special_tokens: list[str] = sorted(special_tokens or [], key=len, reverse=True)
        self.special_token_to_id: dict[str, int] = {
            token: self.bytes_to_id[token.encode()] for token in self.special_tokens
        }
        self._special_token_pattern = None
        if self.special_tokens:
            special_pattern = "|".join(re.escape(token) for token in self.special_tokens)
            self._special_token_pattern = re.compile(special_pattern)

        self._encode_pretoken_cached = lru_cache(maxsize=256)(self._encode_pretoken_uncached)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | PathLike[str],
        merges_filepath: str | PathLike[str],
        special_tokens: list[str] | None = None,
    ) -> Self:
        with open(vocab_filepath, "rb") as vocab_file:
            vocab = pickle.load(vocab_file)
        with open(merges_filepath, "rb") as merges_file:
            merges = pickle.load(merges_file)
        return cls(vocab, merges, special_tokens)

    def _split_on_special_tokens(self, text: str) -> Iterator[tuple[str, bool]]:
        if self._special_token_pattern is None:
            if text:
                yield text, False
            return

        start = 0
        for match in self._special_token_pattern.finditer(text):
            if match.start() > start:
                yield text[start:match.start()], False
            yield match.group(0), True
            start = match.end()

        if start < len(text):
            yield text[start:], False

    def _encode_pretoken_uncached(self, pretoken: str) -> tuple[int, ...]:
        word = tuple(bytes([byte]) for byte in pretoken.encode())
        while len(word) >= 2:
            merge = min(
                (pair for pair in zip(word, word[1:]) if pair in self.merge_ranks),
                key=self.merge_ranks.__getitem__,
                default=None,
            )
            if merge is None:
                break
            word = merge_word(word, merge)

        return tuple(self.bytes_to_id[token] for token in word)

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        return list(self._encode_pretoken_cached(pretoken))

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for part, is_special in self._split_on_special_tokens(text):
            if is_special:
                ids.append(self.special_token_to_id[part])
                continue
            for match in PRETOKEN_PATTERN.finditer(part):
                ids.extend(self._encode_pretoken_cached(match.group(0)))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.id_to_bytes[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> BPETokenizer:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return BPETokenizer(vocab, merges, special_tokens)
