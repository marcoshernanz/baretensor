"""Milestone 001: smoothed character-level bigram language model."""

from __future__ import annotations

from pathlib import Path
import random

import torch
import bt
import numpy as np

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"
LAPLACE_SMOOTHING = 1.0
SEED = 1337
SAMPLE_LEN = 200


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)  # type: ignore


def build_bigram_probs(encoded: list[int], vocab_size: int) -> bt.Tensor:
    bigram_counts = [[LAPLACE_SMOOTHING for _ in range(vocab_size)] for _ in range(vocab_size)]
    for prev_id, next_id in zip(encoded, encoded[1:]):
        bigram_counts[prev_id][next_id] += 1.0
    counts = bt.tensor(np.array(bigram_counts))
    return counts / counts.sum(1, keepdim=True)


# def sample_text(probs: bt.Tensor, chars: list[str], sample_len: int) -> str:
#     sample_id = random.randrange(len(chars))
#     sample = [chars[sample_id]]
#     for _ in range(sample_len - 1):
#         sample_id = int(torch.multinomial(probs[sample_id], num_samples=1).item())
#         sample.append(chars[sample_id])
#     return "".join(sample)


def main() -> None:
    set_seed(SEED)
    tokens = load_text(DATA_PATH)

    chars = sorted(set(tokens))
    char_to_id = {char: idx for idx, char in enumerate(chars)}
    vocab_size = len(char_to_id)

    encoded = [char_to_id[ch] for ch in tokens]
    probs = build_bigram_probs(encoded, vocab_size)
    prev_tokens = encoded[:-1]
    next_tokens = encoded[1:]
    cross_entropy = -probs[prev_tokens, next_tokens].log().mean()
    perplexity = cross_entropy.exp()
    # sample = sample_text(probs, chars, SAMPLE_LEN)

    print(f"vocab_size={vocab_size}")
    print(f"seed={SEED}")
    print(f"cross_entropy={cross_entropy.item():.6f}")
    print(f"perplexity={perplexity.item():.6f}")
    # print(f'sample="""\n{sample}\n"""')


if __name__ == "__main__":
    main()
