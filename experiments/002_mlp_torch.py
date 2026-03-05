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


def sample_text(logits: torch.Tensor, chars: list[str], sample_len: int) -> str:
    probs = logits.exp() / logits.exp().sum(1, keepdim=True)
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
        e = F.embedding(inputs, embeddings)
        h1 = (e @ weights1 + biases1).tanh()
        out = h1 @ weights2 + biases2
        loss = F.cross_entropy(out, targets)

        for p in params:
            p.grad = None

        loss.backward()

        with torch.no_grad():
            for p in params:
                p -= 0.1 * p.grad

        if step % 100 == 0:
            print(f"step={step} loss={loss.item():.6f}")

    batch = torch.tensor(range(len(encoded_val) - 1))
    inputs = encoded_val[batch]
    targets = encoded_val[batch + 1]
    e = F.embedding(inputs, embeddings)
    h1 = (e @ weights1 + biases1).tanh()
    out = h1 @ weights2 + biases2
    validation_loss = F.cross_entropy(out, targets)

    sample = sample_text(out, chars, SAMPLE_LEN)

    print(f"train_loss={loss.item():.6f}")
    print(f"validation_loss={validation_loss.item():.6f}")
    print(f'sample="""\n{sample}\n"""')


if __name__ == "__main__":
    main()
