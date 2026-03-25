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
