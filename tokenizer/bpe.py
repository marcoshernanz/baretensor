from pathlib import Path
import re

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"
BYTE_VOCAB_SIZE = 256
DEFAULT_TARGET_VOCAB_SIZE = 266


def split_text(text: str) -> list[list[int]]:
    chunks = re.findall(r"\s+\S+|\S+|\s+", text)
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


def train_bpe(text: str, target_vocab_size: int) -> tuple[list[list[int]], list[tuple[tuple[int, int], int]]]:
    words = split_text(text)
    vocab_size = BYTE_VOCAB_SIZE
    merges: list[tuple[tuple[int, int], int]] = []

    while vocab_size < target_vocab_size and max(len(word) for word in words) > 1:
        pair = most_frequent_pair(words)
        words = apply_merge(words, pair, vocab_size)
        merges.append((pair, vocab_size))
        vocab_size += 1

    return words, merges


def main() -> None:
    text = DATA_PATH.read_text(encoding="utf-8")[:100000]
    _, merges = train_bpe(text, DEFAULT_TARGET_VOCAB_SIZE)
    print(f"trained {len(merges)} merges")


if __name__ == "__main__":
    main()
