#  %%
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"
text = DATA_PATH.read_text(encoding="utf-8")[:100000]
target_vocab_size = 266

words = [[c for c in f" {word}".encode()] for word in text.split(" ")]
