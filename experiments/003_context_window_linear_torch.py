from __future__ import annotations

import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"
SEED = 1337
EMBEDDING_DIM = 64
BATCH_SIZE = 32
SAMPLE_LENGTH = 200
LEARNING_RATE = 0.05
TRAIN_STEPS = 25_000
CONTEXT_LENGTH = 4


Model = dict[str, torch.Tensor]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)  # type: ignore


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


def model_params(model: Model) -> tuple[torch.Tensor, ...]:
    return (
        model["embedding_table"],
        model["output_weights"],
        model["output_bias"],
    )


def init_model(vocab_size: int) -> Model:
    input_dim = EMBEDDING_DIM * CONTEXT_LENGTH
    model: Model = {
        "embedding_table": torch.randn((vocab_size, EMBEDDING_DIM)) * 0.1,
        "output_weights": torch.randn((input_dim, vocab_size)) * (1.0 / math.sqrt(input_dim)),
        "output_bias": torch.zeros((vocab_size,)),
    }
    for param in model_params(model):
        param.requires_grad = True
    return model


def build_examples(
    token_ids: torch.Tensor,
    start_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.arange(CONTEXT_LENGTH)
    input_ids = token_ids[start_positions[:, None] + offsets]
    target_ids = token_ids[start_positions + CONTEXT_LENGTH]
    return input_ids, target_ids


def forward(input_ids: torch.Tensor, model: Model) -> torch.Tensor:
    embedded = F.embedding(input_ids, model["embedding_table"]).flatten(1)
    return embedded @ model["output_weights"] + model["output_bias"]


def evaluate_split(token_ids: torch.Tensor, model: Model) -> float:
    with torch.no_grad():
        start_positions = torch.arange(len(token_ids) - CONTEXT_LENGTH)
        input_ids, target_ids = build_examples(token_ids, start_positions)
        logits = forward(input_ids, model)
        loss = F.cross_entropy(logits, target_ids)
        return float(loss.item())


def sample_text(
    vocab_chars: list[str],
    sample_length: int,
    model: Model,
    seed_token_ids: torch.Tensor,
) -> str:
    with torch.no_grad():
        seed_start = random.randrange(len(seed_token_ids) - CONTEXT_LENGTH + 1)
        context = seed_token_ids[seed_start : seed_start + CONTEXT_LENGTH].clone()
        sample = [vocab_chars[int(token_id)] for token_id in context[:sample_length]]

        for _ in range(max(sample_length - len(sample), 0)):
            logits = forward(context.unsqueeze(0), model)
            probs = F.softmax(logits[0], dim=0)
            next_token_id = int(torch.multinomial(probs, num_samples=1).item())
            sample.append(vocab_chars[next_token_id])
            context = torch.cat([context[1:], context.new_tensor([next_token_id])])

    return "".join(sample)


def main() -> None:
    set_seed(SEED)
    text = load_text(DATA_PATH)

    vocab_chars = sorted(set(text))
    char_to_index = {char: idx for idx, char in enumerate(vocab_chars)}
    vocab_size = len(char_to_index)

    token_ids = torch.tensor([char_to_index[ch] for ch in text], dtype=torch.long)
    num_tokens = len(token_ids)
    train_token_ids = token_ids[: int(num_tokens * 0.8)]
    val_token_ids = token_ids[int(num_tokens * 0.8) :]
    if len(train_token_ids) <= CONTEXT_LENGTH or len(val_token_ids) <= CONTEXT_LENGTH:
        raise ValueError(
            f"Dataset splits are too small for context length {CONTEXT_LENGTH}. "
            "Need at least one full context window plus one target token in each split."
        )

    model = init_model(vocab_size)

    for step in range(TRAIN_STEPS):
        start_positions = torch.randint(0, len(train_token_ids) - CONTEXT_LENGTH, (BATCH_SIZE,))
        input_ids, target_ids = build_examples(train_token_ids, start_positions)
        logits = forward(input_ids, model)
        loss = F.cross_entropy(logits, target_ids)

        for param in model_params(model):
            param.grad = None

        loss.backward()  # type: ignore

        with torch.no_grad():
            for param in model_params(model):
                grad = param.grad
                assert grad is not None
                param -= LEARNING_RATE * grad

        if step % 1000 == 0:
            print(f"step={step} loss={loss.item():.6f}")

    train_loss = evaluate_split(train_token_ids, model)
    validation_loss = evaluate_split(val_token_ids, model)
    sample = sample_text(vocab_chars, SAMPLE_LENGTH, model, train_token_ids)

    print(f"train_loss={train_loss:.6f}")
    print(f"validation_loss={validation_loss:.6f}")
    print(f'sample="""\n{sample}\n"""')


if __name__ == "__main__":
    main()
