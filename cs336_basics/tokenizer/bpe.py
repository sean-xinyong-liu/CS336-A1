from collections import Counter
from collections import defaultdict
from collections.abc import Iterable


def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    return vocab


def word_to_bytes(word: str) -> tuple[bytes, ...]:
    return tuple(bytes([byte]) for byte in word.encode("utf-8"))


def count_pairs(tokenized_word_counts: Counter[tuple[bytes, ...]]) -> Counter[tuple[bytes, bytes]]:
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()

    for word, count in tokenized_word_counts.items():
        for pair in zip(word, word[1:]):
            pair_counts[pair] += count

    return pair_counts


def iter_pairs(word: tuple[bytes, ...]) -> Iterable[tuple[bytes, bytes]]:
    return zip(word, word[1:])


def build_pair_index(
    tokenized_word_counts: Counter[tuple[bytes, ...]],
) -> tuple[Counter[tuple[bytes, bytes]], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_tokenized_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    for tokenized_word, count in tokenized_word_counts.items():
        for pair in iter_pairs(tokenized_word):
            pair_counts[pair] += count
            pair_to_tokenized_words[pair].add(tokenized_word)

    return pair_counts, pair_to_tokenized_words


def merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged: list[bytes] = []
    index = 0

    while index < len(word):
        if index < len(word) - 1 and word[index] == pair[0] and word[index + 1] == pair[1]:
            merged.append(pair[0] + pair[1])
            index += 2
        else:
            merged.append(word[index])
            index += 1

    return tuple(merged)


def merge_words(
    tokenized_word_counts: Counter[tuple[bytes, ...]],
    pair: tuple[bytes, bytes],
) -> Counter[tuple[bytes, ...]]:
    merged_tokenized_word_counts: Counter[tuple[bytes, ...]] = Counter()

    for word, count in tokenized_word_counts.items():
        merged_tokenized_word_counts[merge_word(word, pair)] += count

    return merged_tokenized_word_counts


def _decrement_pair_count(
    pair_counts: Counter[tuple[bytes, bytes]],
    pair: tuple[bytes, bytes],
    count: int,
) -> None:
    new_count = pair_counts[pair] - count
    if new_count > 0:
        pair_counts[pair] = new_count
    else:
        del pair_counts[pair]


def _replace_word(
    tokenized_word_counts: Counter[tuple[bytes, ...]],
    pair_counts: Counter[tuple[bytes, bytes]],
    pair_to_tokenized_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    old_word: tuple[bytes, ...],
    new_word: tuple[bytes, ...],
    count: int,
) -> None:
    del tokenized_word_counts[old_word]

    for pair, occurrences in Counter(iter_pairs(old_word)).items():
        _decrement_pair_count(pair_counts, pair, count * occurrences)
        tokenized_words = pair_to_tokenized_words.get(pair)
        if tokenized_words is None:
            continue
        tokenized_words.discard(old_word)
        if not tokenized_words:
            del pair_to_tokenized_words[pair]

    tokenized_word_counts[new_word] += count

    for pair, occurrences in Counter(iter_pairs(new_word)).items():
        pair_counts[pair] += count * occurrences
        pair_to_tokenized_words[pair].add(new_word)


def merge_words_with_index(
    tokenized_word_counts: Counter[tuple[bytes, ...]],
    pair_counts: Counter[tuple[bytes, bytes]],
    pair_to_tokenized_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    pair: tuple[bytes, bytes],
) -> None:
    affected_tokenized_words = list(pair_to_tokenized_words.get(pair, ()))

    for word in affected_tokenized_words:
        count = tokenized_word_counts.get(word)
        if not count:
            continue

        new_word = merge_word(word, pair)
        if new_word != word:
            _replace_word(tokenized_word_counts, pair_counts, pair_to_tokenized_words, word, new_word, count)


def build_tokenized_word_counts(word_counts: Counter[str]) -> Counter[tuple[bytes, ...]]:
    tokenized_word_counts: Counter[tuple[bytes, ...]] = Counter()

    for word, count in word_counts.items():
        tokenized_word_counts[word_to_bytes(word)] += count

    return tokenized_word_counts


def train_bpe_from_word_counts(
    word_counts: Counter[str],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = init_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []
    tokenized_word_counts = build_tokenized_word_counts(word_counts)
    pair_counts, pair_to_tokenized_words = build_pair_index(tokenized_word_counts)

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        merge_words_with_index(tokenized_word_counts, pair_counts, pair_to_tokenized_words, best_pair)

    return vocab, merges
