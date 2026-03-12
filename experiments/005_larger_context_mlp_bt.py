from __future__ import annotations

import math
from pathlib import Path
import random
from time import perf_counter
from typing import cast

import bt
import bt.nn.functional as F
import numpy as np

from experiment_artifacts import write_loss_artifacts

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"
SEED = 1337
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
BATCH_SIZE = 32
SAMPLE_LENGTH = 200
LEARNING_RATE = 0.05
TRAIN_STEPS = 50_000
CONTEXT_LENGTH = 16
LOSS_EMA_DECAY = 0.95
LOG_INTERVAL = 1000


Model = dict[str, bt.Tensor]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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


def model_params(model: Model) -> tuple[bt.Tensor, ...]:
    return (
        model["embedding_table"],
        model["hidden_weights"],
        model["hidden_bias"],
        model["output_weights"],
        model["output_bias"],
    )


def init_model(vocab_size: int) -> Model:
    tanh_gain = 5.0 / 3.0
    input_dim = EMBEDDING_DIM * CONTEXT_LENGTH
    model: Model = {
        "embedding_table": bt.tensor(np.random.randn(vocab_size, EMBEDDING_DIM).astype(np.float32))
        * 0.1,
        "hidden_weights": bt.tensor(np.random.randn(input_dim, HIDDEN_DIM).astype(np.float32))
        * (tanh_gain / math.sqrt(input_dim)),
        "hidden_bias": bt.zeros((HIDDEN_DIM,)),
        "output_weights": bt.tensor(np.random.randn(HIDDEN_DIM, vocab_size).astype(np.float32))
        * (1.0 / math.sqrt(HIDDEN_DIM)),
        "output_bias": bt.zeros((vocab_size,)),
    }
    for param in model_params(model):
        param.requires_grad = True
    return model


def build_examples(
    token_ids: np.ndarray,
    start_positions: np.ndarray,
) -> tuple[bt.Tensor, bt.Tensor]:
    offsets = np.arange(CONTEXT_LENGTH, dtype=np.int64)
    input_ids = bt.tensor(token_ids[start_positions[:, None] + offsets])
    target_ids = bt.tensor(token_ids[start_positions + CONTEXT_LENGTH])
    return input_ids, target_ids


def forward(input_ids: bt.Tensor, model: Model) -> bt.Tensor:
    embedded = F.embedding(input_ids, model["embedding_table"]).flatten(1)
    hidden = (embedded @ model["hidden_weights"] + model["hidden_bias"]).tanh()
    return hidden @ model["output_weights"] + model["output_bias"]


def evaluate_split(token_ids: np.ndarray, model: Model) -> float:
    with bt.no_grad():
        start_positions = np.arange(len(token_ids) - CONTEXT_LENGTH)
        input_ids, target_ids = build_examples(token_ids, start_positions)
        logits = forward(input_ids, model)
        loss = F.cross_entropy(logits, target_ids)
        return cast(float, loss.item())


def sample_text(
    vocab_chars: list[str],
    sample_length: int,
    model: Model,
    seed_token_ids: np.ndarray,
) -> str:
    with bt.no_grad():
        seed_start = random.randrange(len(seed_token_ids) - CONTEXT_LENGTH + 1)
        seed_context = seed_token_ids[seed_start : seed_start + CONTEXT_LENGTH]
        context = bt.tensor(seed_context)
        sample = [vocab_chars[int(token_id)] for token_id in seed_context[:sample_length]]

        for _ in range(max(sample_length - len(sample), 0)):
            logits = forward(context.reshape((1, CONTEXT_LENGTH)), model)
            probs = logits[0].softmax(0)
            weights = np.asarray(probs.numpy(), dtype=np.float32).tolist()
            next_token_id = int(random.choices(range(len(vocab_chars)), weights=weights, k=1)[0])
            sample.append(vocab_chars[next_token_id])
            context = bt.cat([context[1:], bt.tensor([next_token_id])], dim=0)

    return "".join(sample)


def main() -> None:
    total_start = perf_counter()
    set_seed(SEED)
    text = load_text(DATA_PATH)

    vocab_chars = sorted(set(text))
    char_to_index = {char: idx for idx, char in enumerate(vocab_chars)}
    vocab_size = len(char_to_index)

    token_ids = np.array([char_to_index[ch] for ch in text], dtype=np.int64)
    num_tokens = len(token_ids)
    train_token_ids = token_ids[: int(num_tokens * 0.8)]
    val_token_ids = token_ids[int(num_tokens * 0.8) :]
    if len(train_token_ids) <= CONTEXT_LENGTH or len(val_token_ids) <= CONTEXT_LENGTH:
        raise ValueError(
            f"Dataset splits are too small for context length {CONTEXT_LENGTH}. "
            "Need at least one full context window plus one target token in each split."
        )

    model = init_model(vocab_size)
    loss_history: list[tuple[int, float, float]] = []
    ema_loss: float | None = None
    train_start = perf_counter()

    for step in range(TRAIN_STEPS):
        start_positions = np.random.randint(0, len(train_token_ids) - CONTEXT_LENGTH, (BATCH_SIZE,))
        input_ids, target_ids = build_examples(train_token_ids, start_positions)
        logits = forward(input_ids, model)
        loss = F.cross_entropy(logits, target_ids)

        for param in model_params(model):
            param.zero_grad()

        loss.backward()  # type: ignore

        with bt.no_grad():
            for param in model_params(model):
                grad = param.grad
                assert grad is not None
                param -= LEARNING_RATE * grad

        raw_loss = cast(float, loss.item())
        ema_loss = (
            raw_loss
            if ema_loss is None
            else LOSS_EMA_DECAY * ema_loss + (1.0 - LOSS_EMA_DECAY) * raw_loss
        )
        loss_history.append((step, raw_loss, ema_loss))

        if step % LOG_INTERVAL == 0:
            print(f"step={step} loss={raw_loss:.6f} ema_loss={ema_loss:.6f}")

    train_seconds = perf_counter() - train_start
    train_loss = evaluate_split(train_token_ids, model)
    validation_loss = evaluate_split(val_token_ids, model)
    sample = sample_text(vocab_chars, SAMPLE_LENGTH, model, train_token_ids)
    loss_history_csv, loss_curve_svg = write_loss_artifacts(Path(__file__), loss_history)
    total_seconds = perf_counter() - total_start

    print(f"train_loss={train_loss:.6f}")
    print(f"validation_loss={validation_loss:.6f}")
    print(f"loss_history_csv={loss_history_csv}")
    print(f"loss_curve_svg={loss_curve_svg}")
    print(f"train_seconds={train_seconds:.3f}")
    print(f"steps_per_second={TRAIN_STEPS / train_seconds:.3f}")
    print(f"total_seconds={total_seconds:.3f}")
    print(f'sample="""\n{sample}\n"""')


if __name__ == "__main__":
    main()
