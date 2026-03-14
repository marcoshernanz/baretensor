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
HIDDEN_DIM = 64
SEQUENCE_LENGTH = 16
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256
SAMPLE_LENGTH = 200
LEARNING_RATE = 0.05
TRAIN_STEPS = 50_000
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
        model["input_weights"],
        model["recurrent_weights"],
        model["hidden_bias"],
        model["output_weights"],
        model["output_bias"],
    )


def init_model(vocab_size: int) -> Model:
    tanh_gain = 5.0 / 3.0
    model: Model = {
        "embedding_table": torch.randn((vocab_size, EMBEDDING_DIM)) * 0.1,
        "input_weights": torch.randn((EMBEDDING_DIM, HIDDEN_DIM))
        * (tanh_gain / math.sqrt(EMBEDDING_DIM)),
        "recurrent_weights": torch.randn((HIDDEN_DIM, HIDDEN_DIM))
        * (tanh_gain / math.sqrt(HIDDEN_DIM)),
        "hidden_bias": torch.zeros((HIDDEN_DIM,)),
        "output_weights": torch.randn((HIDDEN_DIM, vocab_size)) * (1.0 / math.sqrt(HIDDEN_DIM)),
        "output_bias": torch.zeros((vocab_size,)),
    }
    for param in model_params(model):
        param.requires_grad = True
    return model


def init_hidden_state(batch_size: int, model: Model) -> torch.Tensor:
    return model["hidden_bias"].new_zeros((batch_size, HIDDEN_DIM))


def build_sequences(
    token_ids: torch.Tensor,
    start_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.arange(SEQUENCE_LENGTH + 1, device=start_positions.device)
    sequence_token_ids = token_ids[start_positions[:, None] + offsets]
    return sequence_token_ids[:, :-1], sequence_token_ids[:, 1:]


def rnn_step(
    input_token_ids: torch.Tensor,
    previous_hidden_state: torch.Tensor,
    model: Model,
) -> tuple[torch.Tensor, torch.Tensor]:
    embedded_tokens = F.embedding(input_token_ids, model["embedding_table"])
    hidden_state = (
        embedded_tokens @ model["input_weights"]
        + previous_hidden_state @ model["recurrent_weights"]
        + model["hidden_bias"]
    ).tanh()
    logits = hidden_state @ model["output_weights"] + model["output_bias"]
    return logits, hidden_state


def forward_sequence(
    input_token_ids: torch.Tensor,
    model: Model,
    initial_hidden_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length = input_token_ids.shape
    hidden_state = (
        init_hidden_state(batch_size, model)
        if initial_hidden_state is None
        else initial_hidden_state
    )
    logits_by_step: list[torch.Tensor] = []

    for time_step in range(sequence_length):
        step_input_token_ids = input_token_ids[:, time_step]
        step_logits, hidden_state = rnn_step(step_input_token_ids, hidden_state, model)
        logits_by_step.append(step_logits)

    return torch.stack(logits_by_step, dim=1), hidden_state


def sequence_loss(logits_by_step: torch.Tensor, target_token_ids: torch.Tensor) -> torch.Tensor:
    vocab_size = logits_by_step.shape[-1]
    return F.cross_entropy(
        logits_by_step.reshape(-1, vocab_size),
        target_token_ids.reshape(-1),
    )


def evaluate_split(token_ids: torch.Tensor, model: Model) -> float:
    with torch.no_grad():
        start_positions = torch.arange(
            0,
            len(token_ids) - SEQUENCE_LENGTH,
            SEQUENCE_LENGTH,
            device=token_ids.device,
        )
        total_loss = 0.0
        total_sequences = 0

        for batch_start in range(0, len(start_positions), EVAL_BATCH_SIZE):
            batch_positions = start_positions[batch_start : batch_start + EVAL_BATCH_SIZE]
            input_token_ids, target_token_ids = build_sequences(token_ids, batch_positions)
            logits_by_step, _ = forward_sequence(input_token_ids, model)
            batch_loss = sequence_loss(logits_by_step, target_token_ids)
            batch_sequence_count = int(batch_positions.numel())
            total_loss += float(batch_loss.item()) * batch_sequence_count
            total_sequences += batch_sequence_count

        return total_loss / total_sequences


def sample_text(
    vocab_chars: list[str],
    sample_length: int,
    model: Model,
    seed_token_ids: torch.Tensor,
) -> str:
    if sample_length <= 0:
        return ""

    with torch.no_grad():
        seed_token_id = int(seed_token_ids[random.randrange(len(seed_token_ids))].item())
        sample = [vocab_chars[seed_token_id]]
        current_token_ids = seed_token_ids.new_tensor([seed_token_id])
        hidden_state = init_hidden_state(1, model)

        for _ in range(sample_length - 1):
            logits, hidden_state = rnn_step(current_token_ids, hidden_state, model)
            probabilities = F.softmax(logits[0], dim=0)
            next_token_id = int(torch.multinomial(probabilities, num_samples=1).item())
            sample.append(vocab_chars[next_token_id])
            current_token_ids = seed_token_ids.new_tensor([next_token_id])

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
    if len(train_token_ids) <= SEQUENCE_LENGTH or len(val_token_ids) <= SEQUENCE_LENGTH:
        raise ValueError(
            f"Dataset splits are too small for sequence length {SEQUENCE_LENGTH}. "
            "Need at least one full input sequence plus one target token in each split."
        )

    model = init_model(vocab_size)
    loss_history: list[tuple[int, float, float]] = []
    ema_loss: float | None = None
    train_start = perf_counter()

    for step in range(TRAIN_STEPS):
        start_positions = torch.randint(
            0,
            len(train_token_ids) - SEQUENCE_LENGTH,
            (BATCH_SIZE,),
            device=train_token_ids.device,
        )
        input_token_ids, target_token_ids = build_sequences(train_token_ids, start_positions)
        logits_by_step, _ = forward_sequence(input_token_ids, model)
        loss = sequence_loss(logits_by_step, target_token_ids)

        for param in model_params(model):
            param.grad = None

        loss.backward()  # type: ignore

        with torch.no_grad():
            for param in model_params(model):
                grad = param.grad
                assert grad is not None
                param -= LEARNING_RATE * grad

        raw_loss = float(loss.item())
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
