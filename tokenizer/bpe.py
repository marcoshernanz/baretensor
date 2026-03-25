import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"
BYTE_VOCAB_SIZE = 256
DEFAULT_TARGET_VOCAB_SIZE = 266
DEFAULT_SPLIT_PATTERN = r"\s+\S+|\S+|\s+"

type TokenId = int
type TokenPair = tuple[TokenId, TokenId]
type TokenSequence = list[TokenId]
type Merge = tuple[TokenPair, TokenId]


@dataclass(frozen=True, slots=True)
class BPEModel:
    split_pattern: str
    vocab: dict[TokenId, bytes]
    merges: tuple[Merge, ...]
    merge_ranks: dict[TokenPair, int]
    merge_tokens: dict[TokenPair, TokenId]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def split_text(text: str, split_pattern: str = DEFAULT_SPLIT_PATTERN) -> list[bytes]:
    return [chunk.encode("utf-8") for chunk in re.findall(split_pattern, text)]


def count_pairs(sequences: Sequence[TokenSequence]) -> dict[TokenPair, int]:
    pair_counts: dict[TokenPair, int] = {}
    for sequence in sequences:
        for pair in zip(sequence, sequence[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return pair_counts


def select_best_pair(pair_counts: dict[TokenPair, int]) -> TokenPair | None:
    if not pair_counts:
        return None
    return min(pair_counts.items(), key=lambda item: (-item[1], item[0]))[0]


def select_best_mergeable_pair(
    sequence: Sequence[TokenId],
    merge_ranks: dict[TokenPair, int],
) -> TokenPair | None:
    best_pair: TokenPair | None = None
    best_rank: int | None = None
    for pair in zip(sequence, sequence[1:]):
        rank = merge_ranks.get(pair)
        if rank is None:
            continue
        if best_rank is None or rank < best_rank:
            best_pair = pair
            best_rank = rank
    return best_pair


def merge_sequence(
    sequence: Sequence[TokenId], pair: TokenPair, new_token_id: TokenId
) -> TokenSequence:
    merged: TokenSequence = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
            merged.append(new_token_id)
            index += 2
        else:
            merged.append(sequence[index])
            index += 1
    return merged


def build_vocab(merges: Sequence[Merge]) -> dict[TokenId, bytes]:
    vocab = {token_id: bytes([token_id]) for token_id in range(BYTE_VOCAB_SIZE)}
    for pair, token_id in merges:
        vocab[token_id] = vocab[pair[0]] + vocab[pair[1]]
    return vocab


def build_merge_ranks(merges: Sequence[Merge]) -> dict[TokenPair, int]:
    return {pair: rank for rank, (pair, _) in enumerate(merges)}


def build_merge_tokens(merges: Sequence[Merge]) -> dict[TokenPair, TokenId]:
    return {pair: token_id for pair, token_id in merges}


def train_bpe(
    text: str,
    target_vocab_size: int,
    *,
    split_pattern: str = DEFAULT_SPLIT_PATTERN,
) -> BPEModel:
    if target_vocab_size < BYTE_VOCAB_SIZE:
        raise ValueError(
            f"target_vocab_size must be at least {BYTE_VOCAB_SIZE} for byte-level BPE."
        )

    sequences = [list(chunk) for chunk in split_text(text, split_pattern)]
    merges: list[Merge] = []
    next_token_id = BYTE_VOCAB_SIZE

    while next_token_id < target_vocab_size:
        pair_counts = count_pairs(sequences)
        best_pair = select_best_pair(pair_counts)
        if best_pair is None:
            break
        sequences = [merge_sequence(sequence, best_pair, next_token_id) for sequence in sequences]
        merges.append((best_pair, next_token_id))
        next_token_id += 1

    frozen_merges = tuple(merges)
    return BPEModel(
        split_pattern=split_pattern,
        vocab=build_vocab(frozen_merges),
        merges=frozen_merges,
        merge_ranks=build_merge_ranks(frozen_merges),
        merge_tokens=build_merge_tokens(frozen_merges),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal byte-level BPE tokenizer.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Path to the UTF-8 training corpus.",
    )
    parser.add_argument(
        "--target-vocab-size",
        type=int,
        default=DEFAULT_TARGET_VOCAB_SIZE,
        help="Final vocabulary size, including the 256 byte tokens.",
    )
    parser.add_argument(
        "--text-limit",
        type=int,
        default=None,
        help="Optional character limit for quick experiments.",
    )
    parser.add_argument(
        "--split-pattern",
        default=DEFAULT_SPLIT_PATTERN,
        help="Regex used to split text into independently merged chunks.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    text = args.data_path.read_text(encoding="utf-8")
    if args.text_limit is not None:
        text = text[: args.text_limit]

    model = train_bpe(
        text,
        args.target_vocab_size,
        split_pattern=args.split_pattern,
    )
    print(f"trained {len(model.merges)} merges")
    print(f"vocab size: {model.vocab_size}")


if __name__ == "__main__":
    main()
