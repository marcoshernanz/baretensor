from __future__ import annotations

import math
from pathlib import Path
import random
from time import perf_counter

import torch
import torch.nn.functional as F

from experiment_artifacts import write_loss_artifacts

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"
SEED = 1337
EMBEDDING_DIM = 64
BATCH_SIZE = 32
HIDDEN_DIM = 16
SAMPLE_LENGTH = 200
LEARNING_RATE = 0.05
TRAIN_STEPS = 25_000
LOSS_EMA_DECAY = 0.95
LOG_INTERVAL = 1000


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
        model["embedding_table"],
        model["hidden_weights"],
        model["hidden_bias"],
        model["output_weights"],
        model["output_bias"],
    )


def init_model(vocab_size: int) -> Model:
    tanh_gain = 5.0 / 3.0
    model: Model = {
        "embedding_table": torch.randn((vocab_size, EMBEDDING_DIM)) * 0.1,
        "hidden_weights": torch.randn((EMBEDDING_DIM, HIDDEN_DIM))
        * (tanh_gain / math.sqrt(EMBEDDING_DIM)),
        "hidden_bias": torch.zeros((HIDDEN_DIM,)),
        "output_weights": torch.randn((HIDDEN_DIM, vocab_size)) * (1.0 / math.sqrt(HIDDEN_DIM)),
        "output_bias": torch.zeros((vocab_size,)),
    }
    for param in model_params(model):
        param.requires_grad = True
    return model


def forward(input_ids: torch.Tensor, model: Model) -> torch.Tensor:
    embedded = F.embedding(input_ids, model["embedding_table"])
    hidden = (embedded @ model["hidden_weights"] + model["hidden_bias"]).tanh()
    return hidden @ model["output_weights"] + model["output_bias"]


def evaluate_split(token_ids: torch.Tensor, model: Model) -> float:
    with torch.no_grad():
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]
        logits = forward(input_ids, model)
        loss = F.cross_entropy(logits, target_ids)
        return float(loss.item())


def sample_text(vocab_chars: list[str], sample_length: int, model: Model) -> str:
    with torch.no_grad():
        token_id = random.randrange(len(vocab_chars))
        sample = [vocab_chars[token_id]]
        current_token = torch.tensor([token_id], dtype=torch.long)

        for _ in range(sample_length - 1):
            logits = forward(current_token, model)
            probs = F.softmax(logits[0], dim=0)
            token_id = int(torch.multinomial(probs, num_samples=1).item())
            sample.append(vocab_chars[token_id])
            current_token = torch.tensor([token_id], dtype=torch.long)

    return "".join(sample)


def main() -> None:
    total_start = perf_counter()
    set_seed(SEED)
    text = load_text(DATA_PATH)

    vocab_chars = sorted(set(text))
    char_to_index = {char: idx for idx, char in enumerate(vocab_chars)}
    vocab_size = len(char_to_index)

    token_ids = torch.tensor([char_to_index[ch] for ch in text], dtype=torch.long)
    num_tokens = len(token_ids)
    train_token_ids = token_ids[: int(num_tokens * 0.8)]
    val_token_ids = token_ids[int(num_tokens * 0.8) :]

    model = init_model(vocab_size)
    loss_history: list[tuple[int, float, float]] = []
    ema_loss: float | None = None
    train_start = perf_counter()

    for step in range(TRAIN_STEPS):
        batch_indices = torch.randint(0, len(train_token_ids) - 1, (BATCH_SIZE,))
        input_ids = train_token_ids[batch_indices]
        target_ids = train_token_ids[batch_indices + 1]
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

        raw_loss = float(loss.item())
        ema_loss = raw_loss if ema_loss is None else LOSS_EMA_DECAY * ema_loss + (1.0 - LOSS_EMA_DECAY) * raw_loss
        loss_history.append((step, raw_loss, ema_loss))

        if step % LOG_INTERVAL == 0:
            print(f"step={step} loss={raw_loss:.6f} ema_loss={ema_loss:.6f}")

    train_seconds = perf_counter() - train_start
    train_loss = evaluate_split(train_token_ids, model)
    validation_loss = evaluate_split(val_token_ids, model)
    sample = sample_text(vocab_chars, SAMPLE_LENGTH, model)
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
