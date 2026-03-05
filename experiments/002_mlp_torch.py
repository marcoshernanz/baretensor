from __future__ import annotations

import math
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
LEARNING_RATE = 0.1
TRAIN_STEPS = 10_000


Model = dict[str, torch.Tensor]


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


def model_params(model: Model) -> tuple[torch.Tensor, ...]:
    return (
        model["embeddings"],
        model["weights1"],
        model["biases1"],
        model["weights2"],
        model["biases2"],
    )


def init_model(vocab_size: int) -> Model:
    tanh_gain = 5.0 / 3.0
    model: Model = {
        "embeddings": torch.randn((vocab_size, EMBEDDING_LEN)) * 0.1,
        "weights1": torch.randn((EMBEDDING_LEN, HIDDEN_LAYER_LEN))
        * (tanh_gain / math.sqrt(EMBEDDING_LEN)),
        "biases1": torch.zeros((HIDDEN_LAYER_LEN,)),
        "weights2": torch.randn((HIDDEN_LAYER_LEN, vocab_size))
        * (1.0 / math.sqrt(HIDDEN_LAYER_LEN)),
        "biases2": torch.zeros((vocab_size,)),
    }
    for p in model_params(model):
        p.requires_grad = True
    return model


def forward(inputs: torch.Tensor, model: Model) -> torch.Tensor:
    e = F.embedding(inputs, model["embeddings"])
    h1 = (e @ model["weights1"] + model["biases1"]).tanh()
    return h1 @ model["weights2"] + model["biases2"]


def evaluate_split(encoded_split: torch.Tensor, model: Model) -> float:
    with torch.no_grad():
        inputs = encoded_split[:-1]
        targets = encoded_split[1:]
        logits = forward(inputs, model)
        loss = F.cross_entropy(logits, targets)
        return float(loss.item())


def sample_text(chars: list[str], sample_len: int, model: Model) -> str:
    with torch.no_grad():
        sample_id = random.randrange(len(chars))
        sample = [chars[sample_id]]
        current = torch.tensor([sample_id], dtype=torch.long)

        for _ in range(sample_len - 1):
            logits = forward(current, model)
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

    model = init_model(vocab_size)

    for step in range(TRAIN_STEPS):
        batch = torch.randint(0, len(encoded_train) - 1, (BATCH_LEN,))
        inputs = encoded_train[batch]
        targets = encoded_train[batch + 1]
        logits = forward(inputs, model)
        loss = F.cross_entropy(logits, targets)

        for p in model_params(model):
            p.grad = None

        loss.backward()  # pyright: ignore[reportUnknownMemberType]

        with torch.no_grad():
            for p in model_params(model):
                grad = p.grad
                assert grad is not None
                p -= LEARNING_RATE * grad

        if step % 100 == 0:
            print(f"step={step} loss={loss.item():.6f}")

    train_loss = evaluate_split(encoded_train, model)
    validation_loss = evaluate_split(encoded_val, model)
    sample = sample_text(chars, SAMPLE_LEN, model)

    print(f"train_loss={train_loss:.6f}")
    print(f"validation_loss={validation_loss:.6f}")
    print(f'sample="""\n{sample}\n"""')


if __name__ == "__main__":
    main()
