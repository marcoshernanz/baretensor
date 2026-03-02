"""Milestone 001: smoothed character-level bigram language model."""

from __future__ import annotations

from pathlib import Path

import torch

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"
LAPLACE_SMOOTHING = 1.0


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Place tinyshakespeare.txt there before running this script."
        )
    text = path.read_text(encoding="utf-8")
    if len(text) < 2:
        raise ValueError("Dataset is too small. Need at least 2 characters.")
    return text


def main() -> None:
    tokens = load_text(DATA_PATH)

    chars = sorted(set(tokens))
    char_to_id = {char: idx for idx, char in enumerate(chars)}
    vocab_size = len(char_to_id)

    encoded = torch.tensor([char_to_id[ch] for ch in tokens], dtype=torch.long)

    bigram_counts = torch.ones((vocab_size, vocab_size), dtype=torch.float32) * LAPLACE_SMOOTHING
    for prev_id, next_id in zip(encoded, encoded[1:]):
        bigram_counts[prev_id, next_id] += 1.0

    probs = bigram_counts / bigram_counts.sum(1, keepdim=True)
    prev_tokens = encoded[:-1]
    next_tokens = encoded[1:]
    cross_entropy = -torch.log(probs[prev_tokens, next_tokens]).mean()
    perplexity = torch.exp(cross_entropy)

    print(f"vocab_size={vocab_size}")
    print(f"cross_entropy={cross_entropy.item():.6f}")
    print(f"perplexity={perplexity.item():.6f}")


if __name__ == "__main__":
    main()
