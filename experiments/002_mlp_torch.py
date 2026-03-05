from __future__ import annotations

from pathlib import Path
import random

import torch
import torch.nn.functional as F

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"
SEED = 1337
EMBEDDING_LEN = 64
BATCH_LEN = 32
HIDDEN_LAYER_LEN = 16
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


def forward(
    inputs: torch.Tensor,
    embeddings: torch.Tensor,
    weights1: torch.Tensor,
    biases1: torch.Tensor,
    weights2: torch.Tensor,
    biases2: torch.Tensor,
) -> torch.Tensor:
    e = F.embedding(inputs, embeddings)
    h1 = (e @ weights1 + biases1).tanh()
    return h1 @ weights2 + biases2


def split_loss(
    encoded_split: torch.Tensor,
    embeddings: torch.Tensor,
    weights1: torch.Tensor,
    biases1: torch.Tensor,
    weights2: torch.Tensor,
    biases2: torch.Tensor,
) -> torch.Tensor:
    inputs = encoded_split[:-1]
    targets = encoded_split[1:]
    logits = forward(inputs, embeddings, weights1, biases1, weights2, biases2)
    return F.cross_entropy(logits, targets)


def sample_text(
    chars: list[str],
    sample_len: int,
    embeddings: torch.Tensor,
    weights1: torch.Tensor,
    biases1: torch.Tensor,
    weights2: torch.Tensor,
    biases2: torch.Tensor,
) -> str:
    sample_id = random.randrange(len(chars))
    sample = [chars[sample_id]]
    current = torch.tensor([sample_id], dtype=torch.long)

    for _ in range(sample_len - 1):
        logits = forward(current, embeddings, weights1, biases1, weights2, biases2)
        probs = F.softmax(logits[0], dim=0)
        sample_id = int(torch.multinomial(probs, num_samples=1).item())
        sample.append(chars[sample_id])
        current = torch.tensor([sample_id], dtype=torch.long)

    return "".join(sample)


def main() -> None:
    set_seed(SEED)
    tokens = load_text(DATA_PATH)

    chars = sorted(set(tokens))
    char_to_id = {char: idx for idx, char in enumerate(chars)}
    vocab_size = len(char_to_id)

    encoded = torch.tensor([char_to_id[ch] for ch in tokens], dtype=torch.long)
    num_tokens = len(encoded)
    encoded_train = encoded[: int(num_tokens * 0.8)]
    encoded_val = encoded[int(num_tokens * 0.8) :]

    embeddings = torch.randn((vocab_size, EMBEDDING_LEN))
    weights1 = torch.randn((EMBEDDING_LEN, HIDDEN_LAYER_LEN))
    biases1 = torch.randn((HIDDEN_LAYER_LEN,))
    weights2 = torch.randn((HIDDEN_LAYER_LEN, vocab_size))
    biases2 = torch.randn((vocab_size,))

    params = [embeddings, weights1, biases1, weights2, biases2]
    for p in params:
        p.requires_grad = True

    for step in range(10000):
        batch = torch.randint(0, len(encoded_train) - 1, (BATCH_LEN,))
        inputs = encoded_train[batch]
        targets = encoded_train[batch + 1]
        logits = forward(inputs, embeddings, weights1, biases1, weights2, biases2)
        loss = F.cross_entropy(logits, targets)

        for p in params:
            p.grad = None

        loss.backward()  # type: ignore

        with torch.no_grad():
            for p in params:
                assert p.grad is not None
                p -= 0.1 * p.grad

        if step % 100 == 0:
            print(f"step={step} loss={loss.item():.6f}")

    with torch.no_grad():
        train_loss = split_loss(encoded_train, embeddings, weights1, biases1, weights2, biases2)
        validation_loss = split_loss(encoded_val, embeddings, weights1, biases1, weights2, biases2)
        sample = sample_text(chars, SAMPLE_LEN, embeddings, weights1, biases1, weights2, biases2)

    print(f"train_loss={train_loss.item():.6f}")
    print(f"validation_loss={validation_loss.item():.6f}")
    print(f'sample="""\n{sample}\n"""')


if __name__ == "__main__":
    main()
