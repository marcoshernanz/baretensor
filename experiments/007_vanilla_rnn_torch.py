from __future__ import annotations

import math
from pathlib import Path
import random
from time import perf_counter

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from experiment_artifacts import write_loss_artifacts

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"
SEED = 1337
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
SEQUENCE_LENGTH = 64
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 256
SAMPLE_LENGTH = 200
LEARNING_RATE = 0.02
TRAIN_STEPS = 50_000
GRAD_CLIP_NORM = 1.0
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
    torch.manual_seed(seed)


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


def build_streams(token_ids: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    usable_token_count = ((len(token_ids) - 1) // batch_size) * batch_size
    if usable_token_count == 0:
        raise ValueError(
            f"Dataset split with {len(token_ids)} tokens is too small for batch size {batch_size}."
        )

    input_streams = token_ids[:usable_token_count].reshape(batch_size, -1)
    target_streams = token_ids[1 : usable_token_count + 1].reshape(batch_size, -1)
    if input_streams.shape[1] < SEQUENCE_LENGTH:
        raise ValueError(
            f"Stream length {input_streams.shape[1]} is too short for sequence length "
            f"{SEQUENCE_LENGTH} at batch size {batch_size}."
        )
    return input_streams, target_streams


def get_sequence_chunk(
    input_streams: torch.Tensor,
    target_streams: torch.Tensor,
    chunk_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunk_end = chunk_start + SEQUENCE_LENGTH
    return (
        input_streams[:, chunk_start:chunk_end],
        target_streams[:, chunk_start:chunk_end],
    )


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
        input_streams, target_streams = build_streams(token_ids, EVAL_BATCH_SIZE)
        stream_length = input_streams.shape[1]
        hidden_state: torch.Tensor | None = None
        total_loss = 0.0
        total_tokens = 0

        for chunk_start in range(0, stream_length - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH):
            input_chunk, target_chunk = get_sequence_chunk(
                input_streams,
                target_streams,
                chunk_start,
            )
            logits_by_step, hidden_state = forward_sequence(input_chunk, model, hidden_state)
            batch_loss = sequence_loss(logits_by_step, target_chunk)
            batch_token_count = int(input_chunk.numel())
            total_loss += float(batch_loss.item()) * batch_token_count
            total_tokens += batch_token_count

        return total_loss / total_tokens


def sample_text(
    vocab_chars: list[str],
    sample_length: int,
    model: Model,
    seed_token_ids: torch.Tensor,
) -> str:
    if sample_length <= 0:
        return ""

    with torch.no_grad():
        primer_start = random.randrange(len(seed_token_ids) - SEQUENCE_LENGTH + 1)
        primer_token_ids = seed_token_ids[primer_start : primer_start + SEQUENCE_LENGTH].clone()
        sample = [vocab_chars[int(token_id)] for token_id in primer_token_ids[:sample_length]]
        hidden_state = init_hidden_state(1, model)

        for primer_token_id in primer_token_ids[:-1]:
            _, hidden_state = rnn_step(primer_token_id.view(1), hidden_state, model)

        current_token_ids = primer_token_ids[-1:].clone()
        for _ in range(max(sample_length - len(sample), 0)):
            logits, hidden_state = rnn_step(current_token_ids, hidden_state, model)
            probabilities = F.softmax(logits[0], dim=0)
            next_token_id = int(torch.multinomial(probabilities, num_samples=1).item())
            sample.append(vocab_chars[next_token_id])
            current_token_ids = primer_token_ids.new_tensor([next_token_id])

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

    train_input_streams, train_target_streams = build_streams(train_token_ids, BATCH_SIZE)
    train_stream_length = train_input_streams.shape[1]
    model = init_model(vocab_size)
    loss_history: list[tuple[int, float, float]] = []
    ema_loss: float | None = None
    chunk_start = 0
    hidden_state: torch.Tensor | None = None
    train_start = perf_counter()

    for step in range(TRAIN_STEPS):
        if chunk_start + SEQUENCE_LENGTH > train_stream_length:
            chunk_start = 0
            hidden_state = None
        elif hidden_state is not None:
            hidden_state = hidden_state.detach()

        input_chunk, target_chunk = get_sequence_chunk(
            train_input_streams,
            train_target_streams,
            chunk_start,
        )
        logits_by_step, hidden_state = forward_sequence(input_chunk, model, hidden_state)
        loss = sequence_loss(logits_by_step, target_chunk)

        for param in model_params(model):
            param.grad = None

        loss.backward()
        total_grad_norm = float(clip_grad_norm_(model_params(model), GRAD_CLIP_NORM))

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
        hidden_norm = float(hidden_state.norm(dim=1).mean().item())
        loss_history.append((step, raw_loss, ema_loss))

        if step % LOG_INTERVAL == 0:
            print(
                "step="
                f"{step} "
                f"loss={raw_loss:.6f} "
                f"ema_loss={ema_loss:.6f} "
                f"grad_norm={total_grad_norm:.6f} "
                f"hidden_norm={hidden_norm:.6f}"
            )

        chunk_start += SEQUENCE_LENGTH

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
