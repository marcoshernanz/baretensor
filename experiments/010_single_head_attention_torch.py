from __future__ import annotations

import math
from pathlib import Path
import random
from time import perf_counter

import torch
import torch.nn as nn
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


def build_examples(
    token_ids: torch.Tensor,
    start_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.arange(CONTEXT_LENGTH, device=start_positions.device)
    input_ids = token_ids[start_positions[:, None] + offsets]
    target_ids = token_ids[start_positions[:, None] + offsets + 1]
    return input_ids, target_ids


class SingleHeadAttentionLanguageModel(nn.Module):
    token_embedding: nn.Embedding
    position_embedding: nn.Embedding
    query: nn.Linear
    key: nn.Linear
    value: nn.Linear
    output: nn.Linear
    lm_head: nn.Linear
    causal_mask: torch.Tensor

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.position_embedding = nn.Embedding(CONTEXT_LENGTH, EMBEDDING_DIM)
        self.query = nn.Linear(EMBEDDING_DIM, ATTENTION_DIM, bias=False)
        self.key = nn.Linear(EMBEDDING_DIM, ATTENTION_DIM, bias=False)
        self.value = nn.Linear(EMBEDDING_DIM, ATTENTION_DIM, bias=False)
        self.output = nn.Linear(ATTENTION_DIM, EMBEDDING_DIM, bias=False)
        self.lm_head = nn.Linear(EMBEDDING_DIM, vocab_size)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones((CONTEXT_LENGTH, CONTEXT_LENGTH), dtype=torch.bool), diagonal=1),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.1)
        nn.init.normal_(self.position_embedding.weight, std=0.1)
        nn.init.normal_(self.query.weight, std=1.0 / math.sqrt(EMBEDDING_DIM))
        nn.init.normal_(self.key.weight, std=1.0 / math.sqrt(EMBEDDING_DIM))
        nn.init.normal_(self.value.weight, std=1.0 / math.sqrt(EMBEDDING_DIM))
        nn.init.normal_(self.output.weight, std=1.0 / math.sqrt(ATTENTION_DIM))
        nn.init.normal_(self.lm_head.weight, std=1.0 / math.sqrt(EMBEDDING_DIM))
        nn.init.zeros_(self.lm_head.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, sequence_length = input_ids.shape
        if sequence_length != CONTEXT_LENGTH:
            raise ValueError(
                f"Input sequence length {sequence_length} does not match "
                f"context length {CONTEXT_LENGTH}."
            )

        positions = torch.arange(CONTEXT_LENGTH, device=input_ids.device)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions).unsqueeze(0)
        input_embeddings = token_embeddings + position_embeddings

        queries = self.query(input_embeddings)
        keys = self.key(input_embeddings)
        values = self.value(input_embeddings)

        scores = (queries @ keys.transpose(-1, -2)) / math.sqrt(ATTENTION_DIM)
        causal_mask = self.get_buffer("causal_mask")
        masked_scores = scores.masked_fill(causal_mask, float("-inf"))
        attention_weights = F.softmax(masked_scores, dim=-1)
        attention_output = attention_weights @ values
        output = self.output(attention_output)
        return self.lm_head(output)


def loss_fn(
    model: SingleHeadAttentionLanguageModel, input_ids: torch.Tensor, target_ids: torch.Tensor
) -> torch.Tensor:
    logits = model(input_ids)
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
    )


def evaluate_split(token_ids: torch.Tensor, model: SingleHeadAttentionLanguageModel) -> float:
    with torch.no_grad():
        max_start = len(token_ids) - CONTEXT_LENGTH
        if max_start <= 0:
            raise ValueError(
                f"Dataset split is too small for context length {CONTEXT_LENGTH}. "
                "Need at least one full context window plus one target token."
            )

        total_loss = 0.0
        total_examples = 0

        for batch_start in range(0, max_start, EVAL_BATCH_SIZE):
            batch_end = min(batch_start + EVAL_BATCH_SIZE, max_start)
            start_positions = torch.arange(batch_start, batch_end)
            input_ids, target_ids = build_examples(token_ids, start_positions)
            batch_loss = loss_fn(model, input_ids, target_ids)
            batch_size = len(start_positions)
            total_loss += float(batch_loss.item()) * batch_size
            total_examples += batch_size

        return total_loss / total_examples


def sample_text(
    vocab_chars: list[str],
    sample_length: int,
    model: SingleHeadAttentionLanguageModel,
    seed_token_ids: torch.Tensor,
) -> str:
    if sample_length <= 0:
        return ""

    with torch.no_grad():
        seed_start = random.randrange(len(seed_token_ids) - CONTEXT_LENGTH)
        context = seed_token_ids[seed_start : seed_start + CONTEXT_LENGTH].clone()
        sample = [vocab_chars[int(token_id)] for token_id in context[:sample_length]]

        for _ in range(max(sample_length - len(sample), 0)):
            logits = model(context.unsqueeze(0))
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

    model = SingleHeadAttentionLanguageModel(vocab_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_history: list[tuple[int, float, float]] = []
    ema_loss: float | None = None
    train_start = perf_counter()

    for step in range(TRAIN_STEPS):
        start_positions = torch.randint(0, len(train_token_ids) - CONTEXT_LENGTH, (BATCH_SIZE,))
        input_ids, target_ids = build_examples(train_token_ids, start_positions)
        loss = loss_fn(model, input_ids, target_ids)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

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
