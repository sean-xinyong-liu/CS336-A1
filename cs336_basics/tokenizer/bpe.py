from collections import Counter
from collections import defaultdict
from collections.abc import Iterator

type TokenSequence = tuple[bytes, ...]
type Pair = tuple[bytes, bytes]
type TokenSequenceCounts = Counter[TokenSequence]
type PairCounts = Counter[Pair]
type PairIndex = dict[Pair, set[TokenSequence]]


def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    return vocab


def word_to_bytes(pretoken: str) -> TokenSequence:
    return tuple(bytes([byte]) for byte in pretoken.encode())


def count_pairs(token_sequence_counts: TokenSequenceCounts) -> PairCounts:
    pair_counts: PairCounts = Counter()

    for token_sequence, count in token_sequence_counts.items():
        for pair in iter_pairs(token_sequence):
            pair_counts[pair] += count

    return pair_counts


def iter_pairs(token_sequence: TokenSequence) -> Iterator[Pair]:
    return zip(token_sequence, token_sequence[1:])


def build_pair_index(
    token_sequence_counts: TokenSequenceCounts,
) -> tuple[PairCounts, PairIndex]:
    pair_counts: PairCounts = Counter()
    pair_index: PairIndex = defaultdict(set)

    for token_sequence, count in token_sequence_counts.items():
        for pair in iter_pairs(token_sequence):
            pair_counts[pair] += count
            pair_index[pair].add(token_sequence)

    return pair_counts, pair_index


def merge_word(token_sequence: TokenSequence, pair: Pair) -> TokenSequence:
    merged: list[bytes] = []
    index = 0

    while index < len(token_sequence):
        if (
            index < len(token_sequence) - 1
            and token_sequence[index] == pair[0]
            and token_sequence[index + 1] == pair[1]
        ):
            merged.append(pair[0] + pair[1])
            index += 2
        else:
            merged.append(token_sequence[index])
            index += 1

    return tuple(merged)


def merge_words(
    token_sequence_counts: TokenSequenceCounts,
    pair: Pair,
) -> TokenSequenceCounts:
    merged_counts: TokenSequenceCounts = Counter()

    for token_sequence, count in token_sequence_counts.items():
        merged_counts[merge_word(token_sequence, pair)] += count

    return merged_counts


def _decrement_pair_count(
    pair_counts: PairCounts,
    pair: Pair,
    count: int,
) -> None:
    new_count = pair_counts[pair] - count
    if new_count > 0:
        pair_counts[pair] = new_count
    else:
        del pair_counts[pair]


def _update_merged_sequence(
    token_sequence_counts: TokenSequenceCounts,
    pair_counts: PairCounts,
    pair_index: PairIndex,
    old_sequence: TokenSequence,
    new_sequence: TokenSequence,
    count: int,
) -> None:
    del token_sequence_counts[old_sequence]

    for pair, occurrences in Counter(iter_pairs(old_sequence)).items():
        _decrement_pair_count(pair_counts, pair, count * occurrences)
        indexed_sequences = pair_index.get(pair)
        if indexed_sequences is None:
            continue
        indexed_sequences.discard(old_sequence)
        if not indexed_sequences:
            del pair_index[pair]

    token_sequence_counts[new_sequence] += count

    for pair, occurrences in Counter(iter_pairs(new_sequence)).items():
        pair_counts[pair] += count * occurrences
        pair_index[pair].add(new_sequence)


def merge_words_with_index(
    token_sequence_counts: TokenSequenceCounts,
    pair_counts: PairCounts,
    pair_index: PairIndex,
    pair: Pair,
) -> None:
    affected_sequences = list(pair_index.get(pair, ()))

    for token_sequence in affected_sequences:
        count = token_sequence_counts.get(token_sequence)
        if not count:
            continue

        merged_sequence = merge_word(token_sequence, pair)
        if merged_sequence != token_sequence:
            _update_merged_sequence(
                token_sequence_counts,
                pair_counts,
                pair_index,
                token_sequence,
                merged_sequence,
                count,
            )


def build_tokenized_word_counts(pretoken_counts: Counter[str]) -> TokenSequenceCounts:
    token_sequence_counts: TokenSequenceCounts = Counter()

    for pretoken, count in pretoken_counts.items():
        token_sequence_counts[word_to_bytes(pretoken)] += count

    return token_sequence_counts


def train_bpe_from_word_counts(
    pretoken_counts: Counter[str],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = init_vocab(special_tokens)
    merges: list[Pair] = []
    token_sequence_counts = build_tokenized_word_counts(pretoken_counts)
    pair_counts, pair_index = build_pair_index(token_sequence_counts)

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        merge_words_with_index(token_sequence_counts, pair_counts, pair_index, best_pair)

    return vocab, merges
