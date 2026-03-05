"""Milestone 001: smoothed character-level bigram language model."""

from __future__ import annotations

from pathlib import Path
import math
import random

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
    np.random.seed(seed)


def build_bigram_probs(encoded: list[int], vocab_size: int) -> bt.Tensor:
    counts_np = np.full((vocab_size, vocab_size), LAPLACE_SMOOTHING, dtype=np.float32)
    prev_np = np.asarray(encoded[:-1], dtype=np.int64)
    next_np = np.asarray(encoded[1:], dtype=np.int64)
    np.add.at(counts_np, (prev_np, next_np), 1.0)
    counts = bt.tensor(counts_np)
    return counts / counts.sum(1, keepdim=True)


def sample_text(probs: bt.Tensor, chars: list[str], sample_len: int) -> str:
    probs_np = np.asarray(probs.numpy(), dtype=np.float32)
    sample_id = random.randrange(len(chars))
    sample_chars = [chars[sample_id]]

    for _ in range(sample_len - 1):
        weights = probs_np[sample_id].tolist()
        sample_id = int(random.choices(range(len(chars)), weights=weights, k=1)[0])
        sample_chars.append(chars[sample_id])

    return "".join(sample_chars)


def main() -> None:
    set_seed(SEED)
    tokens = load_text(DATA_PATH)

    chars = sorted(set(tokens))
    char_to_id = {char: idx for idx, char in enumerate(chars)}
    vocab_size = len(char_to_id)

    encoded = [char_to_id[ch] for ch in tokens]
    probs = build_bigram_probs(encoded, vocab_size)
    log_prob_sum = 0.0
    token_count = 0
    for prev_id, next_id in zip(encoded, encoded[1:]):
        log_prob_sum += probs[prev_id, next_id].log().item()
        token_count += 1
    cross_entropy = -log_prob_sum / token_count
    perplexity = math.exp(cross_entropy)
    sample = sample_text(probs, chars, SAMPLE_LEN)

    print(f"vocab_size={vocab_size}")
    print(f"seed={SEED}")
    print(f"cross_entropy={cross_entropy:.6f}")
    print(f"perplexity={perplexity:.6f}")
    print(f'sample="""\n{sample}\n"""')


if __name__ == "__main__":
    main()
