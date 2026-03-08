"""Milestone 001: smoothed character-level bigram language model."""

from __future__ import annotations

from pathlib import Path
import random

import torch

from experiment_artifacts import write_loss_artifacts

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


def build_bigram_probs(encoded: torch.Tensor, vocab_size: int) -> torch.Tensor:
    bigram_counts = torch.ones((vocab_size, vocab_size), dtype=torch.float32) * LAPLACE_SMOOTHING
    for prev_id, next_id in zip(encoded, encoded[1:]):
        bigram_counts[prev_id, next_id] += 1.0
    return bigram_counts / bigram_counts.sum(1, keepdim=True)


def sample_text(probs: torch.Tensor, chars: list[str], sample_len: int) -> str:
    sample_id = random.randrange(len(chars))
    sample = [chars[sample_id]]
    for _ in range(sample_len - 1):
        sample_id = int(torch.multinomial(probs[sample_id], num_samples=1).item())
        sample.append(chars[sample_id])
    return "".join(sample)


def main() -> None:
    set_seed(SEED)
    tokens = load_text(DATA_PATH)

    chars = sorted(set(tokens))
    char_to_id = {char: idx for idx, char in enumerate(chars)}
    vocab_size = len(char_to_id)

    encoded = torch.tensor([char_to_id[ch] for ch in tokens], dtype=torch.long)
    probs = build_bigram_probs(encoded, vocab_size)
    prev_tokens = encoded[:-1]
    next_tokens = encoded[1:]
    cross_entropy = -torch.log(probs[prev_tokens, next_tokens]).mean()
    sample = sample_text(probs, chars, SAMPLE_LEN)
    loss_value = float(cross_entropy.item())
    loss_history = [(0, loss_value, loss_value)]
    loss_history_csv, loss_curve_svg = write_loss_artifacts(Path(__file__), loss_history)

    print(f"cross_entropy={loss_value:.6f}")
    print(f"loss_history_csv={loss_history_csv}")
    print(f"loss_curve_svg={loss_curve_svg}")
    print(f'sample="""\n{sample}\n"""')


if __name__ == "__main__":
    main()
