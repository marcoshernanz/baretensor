#  %%
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"
text = DATA_PATH.read_text(encoding="utf-8")[:100000]
target_vocab_size = 266

words = [[c for c in f" {word}".encode()] for word in text.split(" ")]

# %%


def most_frequent_pair(words: list[list[int]]) -> tuple[int, int]:
    freq = dict()
    for word in words:
        for pair in zip(word, word[1:]):
            freq[pair] = freq[pair] + 1 if pair in freq else 1

    return max(freq.items(), key=lambda item: item[1])[0]


# %%


def apply_merge(words: list[list[int]], pair: tuple[int, int], value: int) -> list[list[int]]:
    new_words = []
    for word in words:
        new_word = []
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


# %%

vocab_size = 256
merges = []
while vocab_size < target_vocab_size and max([len(word) for word in words]) > 1:
    pair = most_frequent_pair(words)
    words = apply_merge(words, pair, vocab_size)
    merges.append((pair, vocab_size))
    vocab_size += 1
