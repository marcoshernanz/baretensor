from dataclasses import dataclass
from pathlib import Path
import re

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"
BYTE_VOCAB_SIZE = 256
DEFAULT_TARGET_VOCAB_SIZE = 266
SPLIT_PATTERN = r"\s+\S+|\S+|\s+"


@dataclass(frozen=True)
class BPEModel:
    vocab: dict[int, bytes]
    merges: list[tuple[tuple[int, int], int]]
    merge_ranks: dict[tuple[int, int], int]
    split_pattern: str


def split_text(text: str, split_pattern: str = SPLIT_PATTERN) -> list[list[int]]:
    chunks = re.findall(split_pattern, text)
    return [list(chunk.encode("utf-8")) for chunk in chunks]


def most_frequent_pair(words: list[list[int]]) -> tuple[int, int]:
    freq: dict[tuple[int, int], int] = {}
    for word in words:
        for pair in zip(word, word[1:]):
            freq[pair] = freq.get(pair, 0) + 1

    return max(freq.items(), key=lambda item: item[1])[0]


def apply_merge(words: list[list[int]], pair: tuple[int, int], value: int) -> list[list[int]]:
    new_words: list[list[int]] = []
    for word in words:
        new_word: list[int] = []
        skip = False
        for i in range(len(word)):
            if skip:
                skip = False
                continue

            if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
                new_word.append(value)
                skip = True
            else:
                new_word.append(word[i])
        new_words.append(new_word)

    return new_words


def build_vocab(merges: list[tuple[tuple[int, int], int]]) -> dict[int, bytes]:
    vocab = {token_id: bytes([token_id]) for token_id in range(BYTE_VOCAB_SIZE)}
    for pair, token_id in merges:
        vocab[token_id] = vocab[pair[0]] + vocab[pair[1]]
    return vocab


def build_merge_ranks(
    merges: list[tuple[tuple[int, int], int]],
) -> dict[tuple[int, int], int]:
    return {pair: rank for rank, (pair, _) in enumerate(merges)}


def train_bpe(text: str, target_vocab_size: int) -> BPEModel:
    words = split_text(text)
    vocab_size = BYTE_VOCAB_SIZE
    merges: list[tuple[tuple[int, int], int]] = []

    while vocab_size < target_vocab_size and max(len(word) for word in words) > 1:
        pair = most_frequent_pair(words)
        words = apply_merge(words, pair, vocab_size)
        merges.append((pair, vocab_size))
        vocab_size += 1

    return BPEModel(
        vocab=build_vocab(merges),
        merges=merges,
        merge_ranks=build_merge_ranks(merges),
        split_pattern=SPLIT_PATTERN,
    )


def main() -> None:
    text = DATA_PATH.read_text(encoding="utf-8")[:100000]
    model = train_bpe(text, DEFAULT_TARGET_VOCAB_SIZE)
    print(f"trained {len(model.merges)} merges")
    print(f"vocab size: {len(model.vocab)}")


if __name__ == "__main__":
    main()
