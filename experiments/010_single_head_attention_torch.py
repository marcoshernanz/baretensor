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
EMBEDDING_DIM = 128
ATTENTION_DIM = 64
CONTEXT_LENGTH = 64
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
SAMPLE_LENGTH = 200
LEARNING_RATE = 0.02
TRAIN_STEPS = 100_000
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
        model["token_embedding_table"],
        model["position_embedding_table"],
        model["query_weights"],
        model["key_weights"],
        model["value_weights"],
        model["attention_output_weights"],
        model["logit_weights"],
        model["logit_bias"],
    )


def init_model(vocab_size: int) -> Model:
    model: Model = {
        "token_embedding_table": torch.randn((vocab_size, EMBEDDING_DIM)) * 0.1,
        "position_embedding_table": torch.randn((CONTEXT_LENGTH, EMBEDDING_DIM)) * 0.1,
        "query_weights": torch.randn((EMBEDDING_DIM, ATTENTION_DIM))
        * (1.0 / math.sqrt(EMBEDDING_DIM)),
        "key_weights": torch.randn((EMBEDDING_DIM, ATTENTION_DIM))
        * (1.0 / math.sqrt(EMBEDDING_DIM)),
        "value_weights": torch.randn((EMBEDDING_DIM, ATTENTION_DIM))
        * (1.0 / math.sqrt(EMBEDDING_DIM)),
        "attention_output_weights": torch.randn((ATTENTION_DIM, EMBEDDING_DIM))
        * (1.0 / math.sqrt(ATTENTION_DIM)),
        "logit_weights": torch.randn((EMBEDDING_DIM, vocab_size))
        * (1.0 / math.sqrt(EMBEDDING_DIM)),
        "logit_bias": torch.zeros((vocab_size,)),
    }
    for param in model_params(model):
        param.requires_grad = True
    return model


def build_examples(
    token_ids: torch.Tensor,
    start_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.arange(CONTEXT_LENGTH, device=start_positions.device)
    input_ids = token_ids[start_positions[:, None] + offsets]
    target_ids = token_ids[start_positions[:, None] + offsets + 1]
    return input_ids, target_ids


def forward(input_ids: torch.Tensor, model: Model) -> torch.Tensor:
    token_embeddings = F.embedding(input_ids, model["token_embedding_table"])
    position_embeddings = model["position_embedding_table"][
        torch.arange(CONTEXT_LENGTH, device=input_ids.device)
    ]
    input_embeddings = token_embeddings + position_embeddings

    queries = input_embeddings @ model["query_weights"]
    keys = input_embeddings @ model["key_weights"]
    values = input_embeddings @ model["value_weights"]

    scores = (queries @ keys.transpose(-1, -2)) / math.sqrt(ATTENTION_DIM)
    causal_mask = torch.triu(
        torch.ones((CONTEXT_LENGTH, CONTEXT_LENGTH), dtype=torch.bool, device=input_ids.device),
        diagonal=1,
    )
    masked_scores = scores.masked_fill(causal_mask, float("-inf"))
    attention_weights = F.softmax(masked_scores, dim=-1)
    attention_output = attention_weights @ values
    output = attention_output @ model["attention_output_weights"]
    return output @ model["logit_weights"] + model["logit_bias"]


def loss_fn(model: Model, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    logits = forward(input_ids, model)
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
    )


def evaluate_batch_loss(
    model: Model, input_ids: torch.Tensor, target_ids: torch.Tensor
) -> torch.Tensor:
    return loss_fn(model, input_ids, target_ids)


def evaluate_split(token_ids: torch.Tensor, model: Model) -> float:
    max_start = len(token_ids) - CONTEXT_LENGTH
    if max_start <= 0:
        raise ValueError(
            f"Dataset split is too small for context length {CONTEXT_LENGTH}. "
            "Need at least one full context window plus one target token."
        )

    with torch.no_grad():
        total_loss = 0.0
        total_examples = 0

        for batch_start in range(0, max_start, EVAL_BATCH_SIZE):
            batch_end = min(batch_start + EVAL_BATCH_SIZE, max_start)
            start_positions = torch.arange(batch_start, batch_end, dtype=torch.long)
            input_ids, target_ids = build_examples(token_ids, start_positions)
            batch_loss = evaluate_batch_loss(model, input_ids, target_ids)
            batch_size = int(len(start_positions))
            total_loss += float(batch_loss.item()) * batch_size
            total_examples += batch_size

    return total_loss / total_examples


def sample_text(
    vocab_chars: list[str],
    sample_length: int,
    model: Model,
    seed_token_ids: torch.Tensor,
) -> str:
    if sample_length <= 0:
        return ""

    with torch.no_grad():
        seed_start = random.randrange(len(seed_token_ids) - CONTEXT_LENGTH)
        context = seed_token_ids[seed_start : seed_start + CONTEXT_LENGTH].clone()
        sample = [vocab_chars[int(token_id)] for token_id in context[:sample_length]]

        for _ in range(max(sample_length - len(sample), 0)):
            logits = forward(context.unsqueeze(0), model)
            probabilities = F.softmax(logits[0, -1], dim=0)
            next_token_id = int(torch.multinomial(probabilities, num_samples=1).item())
            sample.append(vocab_chars[next_token_id])
            context = torch.cat([context[1:], context.new_tensor([next_token_id])])

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
        start_positions = torch.randint(0, len(train_token_ids) - CONTEXT_LENGTH, (BATCH_SIZE,))
        input_ids, target_ids = build_examples(train_token_ids, start_positions)
        loss = loss_fn(model, input_ids, target_ids)

        for param in model_params(model):
            param.grad = None

        loss.backward()

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
