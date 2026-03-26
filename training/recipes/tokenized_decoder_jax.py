from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from flax import nnx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import optax  # pyright: ignore

from tokenizer.bpe import BPEModel
from training.config import TrainingConfig


TRAIN_SPLIT_RATIO = 0.8
LAYER_NORM_EPS = 1e-5


@dataclass(frozen=True, slots=True)
class DatasetStats:
    vocab_size: int
    train_chars: int
    validation_chars: int
    train_tokens: int
    validation_tokens: int
    train_chars_per_token: float
    validation_chars_per_token: float


class LayerNorm(nnx.Module):
    scale: nnx.Param[jax.Array]
    shift: nnx.Param[jax.Array]

    def __init__(self, features: int):
        self.scale = nnx.Param(jnp.ones((features,)))
        self.shift = nnx.Param(jnp.zeros((features,)))

    def __call__(self, x: jax.Array, *, eps: float) -> jax.Array:
        mean = x.mean(axis=-1, keepdims=True)
        variance = x.var(axis=-1, keepdims=True)
        normalized = (x - mean) / jnp.sqrt(variance + eps)
        return self.scale * normalized + self.shift


class Embedding(nnx.Module):
    weight: nnx.Param[jax.Array]

    def __init__(self, num_embeddings: int, embedding_dim: int, *, rngs: nnx.Rngs):
        self.weight = nnx.Param(rngs.params.normal((num_embeddings, embedding_dim)) * 0.1)

    def __call__(self, indices: jax.Array) -> jax.Array:
        return self.weight[indices]


class Linear(nnx.Module):
    weight: nnx.Param[jax.Array]
    bias: nnx.Param[jax.Array] | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
        bias: bool = True,
    ):
        scale = 1.0 / math.sqrt(in_features)
        self.weight = nnx.Param(rngs.params.normal((in_features, out_features)) * scale)
        self.bias = nnx.Param(jnp.zeros((out_features,))) if bias else None

    def __call__(self, x: jax.Array) -> jax.Array:
        output = x @ self.weight
        if self.bias is not None:
            output = output + self.bias
        return output


class CausalSelfAttention(nnx.Module):
    query: Linear
    key: Linear
    value: Linear
    output: Linear
    num_heads: int
    head_dim: int

    def __init__(self, embedding_dim: int, num_heads: int, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.query = Linear(embedding_dim, embedding_dim, rngs=rngs, bias=False)
        self.key = Linear(embedding_dim, embedding_dim, rngs=rngs, bias=False)
        self.value = Linear(embedding_dim, embedding_dim, rngs=rngs, bias=False)
        self.output = Linear(embedding_dim, embedding_dim, rngs=rngs, bias=False)

    def split_heads(self, x: jax.Array) -> jax.Array:
        batch_size, sequence_length, _ = x.shape
        head_shape = (batch_size, sequence_length, self.num_heads, self.head_dim)
        return x.reshape(head_shape).swapaxes(1, 2)

    def combine_heads(self, x: jax.Array) -> jax.Array:
        batch_size, _, sequence_length, _ = x.shape
        combined_shape = (batch_size, sequence_length, self.num_heads * self.head_dim)
        return x.swapaxes(1, 2).reshape(combined_shape)

    def __call__(self, x: jax.Array) -> jax.Array:
        sequence_length = x.shape[1]
        queries = self.split_heads(self.query(x))
        keys = self.split_heads(self.key(x))
        values = self.split_heads(self.value(x))

        attention_scores = (queries @ keys.mT) / math.sqrt(self.head_dim)
        causal_mask = jnp.triu(jnp.ones((sequence_length, sequence_length), dtype=bool), k=1)
        masked_attention_scores = jnp.where(causal_mask, -jnp.inf, attention_scores)
        attention_weights = jnn.softmax(masked_attention_scores, axis=-1)
        attended_values = attention_weights @ values
        combined_heads = self.combine_heads(attended_values)
        return self.output(combined_heads)


class FeedForward(nnx.Module):
    hidden: Linear
    output: Linear

    def __init__(self, embedding_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        self.hidden = Linear(embedding_dim, hidden_dim, rngs=rngs)
        self.output = Linear(hidden_dim, embedding_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        hidden_activation = jnp.tanh(self.hidden(x))
        return self.output(hidden_activation)


class DecoderBlock(nnx.Module):
    attention: CausalSelfAttention
    attention_norm: LayerNorm
    feed_forward: FeedForward
    feed_forward_norm: LayerNorm

    def __init__(self, embedding_dim: int, hidden_dim: int, num_heads: int, *, rngs: nnx.Rngs):
        self.attention = CausalSelfAttention(embedding_dim, num_heads, rngs=rngs)
        self.attention_norm = LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, hidden_dim, rngs=rngs)
        self.feed_forward_norm = LayerNorm(embedding_dim)

    def __call__(self, x: jax.Array, *, eps: float) -> jax.Array:
        attention_residual = x + self.attention(x)
        attention_block_output = self.attention_norm(attention_residual, eps=eps)

        feed_forward_residual = attention_block_output + self.feed_forward(attention_block_output)
        return self.feed_forward_norm(feed_forward_residual, eps=eps)


class Decoder(nnx.Module):
    blocks: nnx.List[DecoderBlock]

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.blocks = nnx.List(
            [
                DecoderBlock(embedding_dim, hidden_dim, num_heads, rngs=rngs)
                for _ in range(num_blocks)
            ]
        )

    def __call__(self, x: jax.Array, *, eps: float) -> jax.Array:
        for block in self.blocks:
            x = block(x, eps=eps)
        return x


class LanguageModel(nnx.Module):
    token_embedding: Embedding
    position_embedding: Embedding
    decoder: Decoder
    lm_head: Linear

    def __init__(self, config: TrainingConfig, vocab_size: int, *, rngs: nnx.Rngs):
        self.token_embedding = Embedding(vocab_size, config.model.embedding_dim, rngs=rngs)
        self.position_embedding = Embedding(
            config.data.context_tokens,
            config.model.embedding_dim,
            rngs=rngs,
        )
        self.decoder = Decoder(
            config.model.embedding_dim,
            config.model.hidden_dim,
            config.model.num_heads,
            config.model.num_decoder_blocks,
            rngs=rngs,
        )
        self.lm_head = Linear(config.model.embedding_dim, vocab_size, rngs=rngs)

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        positions = jnp.arange(input_ids.shape[-1], dtype=jnp.int32)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        decoder_input = token_embeddings + position_embeddings
        decoder_output = self.decoder(decoder_input, eps=LAYER_NORM_EPS)
        return self.lm_head(decoder_output)


@dataclass(slots=True)
class TokenizedDecoderJaxRecipe:
    config: TrainingConfig
    model: LanguageModel
    optimizer: nnx.Optimizer[LanguageModel]
    tokenizer: BPEModel
    train_token_ids: jax.Array
    validation_token_ids: jax.Array
    train_text: str
    validation_text: str
    stats: DatasetStats

    @classmethod
    def create(cls, config: TrainingConfig) -> TokenizedDecoderJaxRecipe:
        text = _load_text(config.data.dataset_path, config.data.text_limit)
        tokenizer = _load_tokenizer(config.data.tokenizer_path)
        train_token_ids, validation_token_ids, train_text, validation_text = _build_token_splits(
            text, tokenizer, train_split_ratio=TRAIN_SPLIT_RATIO
        )
        if (
            train_token_ids.shape[0] <= config.data.context_tokens
            or validation_token_ids.shape[0] <= config.data.context_tokens
        ):
            raise ValueError(
                f"Dataset splits are too small for context length {config.data.context_tokens}. "
                "Need at least one full context window plus one target token in each split."
            )

        rngs = nnx.Rngs(config.run.seed)
        model = LanguageModel(config, tokenizer.vocab_size, rngs=rngs)
        optimizer = nnx.Optimizer(model, optax.sgd(config.optimizer.learning_rate), wrt=nnx.Param)
        stats = DatasetStats(
            vocab_size=tokenizer.vocab_size,
            train_chars=len(train_text),
            validation_chars=len(validation_text),
            train_tokens=int(train_token_ids.shape[0]),
            validation_tokens=int(validation_token_ids.shape[0]),
            train_chars_per_token=len(train_text) / int(train_token_ids.shape[0]),
            validation_chars_per_token=len(validation_text) / int(validation_token_ids.shape[0]),
        )
        return cls(
            config=config,
            model=model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            train_token_ids=train_token_ids,
            validation_token_ids=validation_token_ids,
            train_text=train_text,
            validation_text=validation_text,
            stats=stats,
        )

    def train_batch(self, rng: jax.Array) -> tuple[jax.Array, float]:
        rng, batch_rng = jax.random.split(rng)
        start_positions = jax.random.randint(
            batch_rng,
            shape=(self.config.train.batch_size,),
            minval=0,
            maxval=self.train_token_ids.shape[0] - self.config.data.context_tokens,
        )
        input_ids, target_ids = _build_examples(
            self.train_token_ids,
            start_positions,
            self.config.data.context_tokens,
        )
        loss = _train_step(self.model, self.optimizer, input_ids, target_ids)
        return rng, float(loss)

    def evaluate_train_loss(self) -> float:
        return _evaluate_split(
            self.train_token_ids,
            self.model,
            context_tokens=self.config.data.context_tokens,
            eval_batch_size=self.config.train.eval_batch_size,
        )

    def evaluate_validation_loss(self) -> float:
        return _evaluate_split(
            self.validation_token_ids,
            self.model,
            context_tokens=self.config.data.context_tokens,
            eval_batch_size=self.config.train.eval_batch_size,
        )

    def generate_sample(self, rng: jax.Array) -> tuple[jax.Array, str]:
        rng, seed_rng = jax.random.split(rng)
        context_tokens = self.config.data.context_tokens
        seed_start = int(
            jax.random.randint(
                seed_rng,
                shape=(),
                minval=0,
                maxval=self.train_token_ids.shape[0] - context_tokens,
            )
        )
        context = self.train_token_ids[seed_start : seed_start + context_tokens]
        generated_token_ids = context[: self.config.train.sample_tokens].tolist()

        for _ in range(max(self.config.train.sample_tokens - len(generated_token_ids), 0)):
            logits = self.model(context[None, :])
            rng, token_rng = jax.random.split(rng)
            next_token_id = int(jax.random.categorical(token_rng, logits[0, -1]))
            generated_token_ids.append(next_token_id)
            context = jnp.concatenate((context[1:], jnp.asarray([next_token_id], dtype=jnp.int32)))

        return rng, _decode_token_ids_for_sample(self.tokenizer, generated_token_ids)


def _load_text(path: Path, text_limit: int | None) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place tinyshakespeare.txt there before running this recipe."
        )
    text = path.read_text(encoding="utf-8")
    if text_limit is not None:
        text = text[:text_limit]
    if len(text) < 2:
        raise ValueError("Dataset is too small. Need at least 2 characters.")
    return text


def _load_tokenizer(path: Path) -> BPEModel:
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenizer artifact not found at {path}. Train and freeze the tokenizer first."
        )
    return BPEModel.load(path)


def _encode_text(tokenizer: BPEModel, text: str) -> jax.Array:
    return jnp.asarray(tokenizer.encode(text), dtype=jnp.int32)


def _build_token_splits(
    text: str,
    tokenizer: BPEModel,
    *,
    train_split_ratio: float,
) -> tuple[jax.Array, jax.Array, str, str]:
    split_index = int(len(text) * train_split_ratio)
    train_text = text[:split_index]
    validation_text = text[split_index:]
    train_token_ids = _encode_text(tokenizer, train_text)
    validation_token_ids = _encode_text(tokenizer, validation_text)
    return train_token_ids, validation_token_ids, train_text, validation_text


def _build_examples(
    token_ids: jax.Array,
    start_positions: jax.Array,
    context_tokens: int,
) -> tuple[jax.Array, jax.Array]:
    offsets = jnp.arange(context_tokens, dtype=start_positions.dtype)
    input_ids = token_ids[start_positions[:, None] + offsets]
    target_ids = token_ids[start_positions[:, None] + offsets + 1]
    return input_ids, target_ids


def _loss_fn(model: LanguageModel, input_ids: jax.Array, target_ids: jax.Array) -> jax.Array:
    logits = model(input_ids)
    log_probs = jnn.log_softmax(logits, axis=-1)
    loss_per_token = -jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1).squeeze(-1)
    return loss_per_token.mean()


@nnx.jit
def _train_step(
    model: LanguageModel,
    optimizer: nnx.Optimizer[LanguageModel],
    input_ids: jax.Array,
    target_ids: jax.Array,
) -> jax.Array:
    loss, grads = nnx.value_and_grad(_loss_fn)(model, input_ids, target_ids)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def _evaluate_batch_loss(
    model: LanguageModel,
    input_ids: jax.Array,
    target_ids: jax.Array,
) -> jax.Array:
    return _loss_fn(model, input_ids, target_ids)


def _evaluate_split(
    token_ids: jax.Array,
    model: LanguageModel,
    *,
    context_tokens: int,
    eval_batch_size: int,
) -> float:
    max_start = token_ids.shape[0] - context_tokens
    if max_start <= 0:
        raise ValueError(
            f"Dataset split is too small for context length {context_tokens}. "
            "Need at least one full context window plus one target token."
        )

    total_loss = 0.0
    total_examples = 0

    for batch_start in range(0, max_start, eval_batch_size):
        batch_end = min(batch_start + eval_batch_size, max_start)
        start_positions = jnp.arange(batch_start, batch_end, dtype=jnp.int32)
        input_ids, target_ids = _build_examples(token_ids, start_positions, context_tokens)
        batch_loss = _evaluate_batch_loss(model, input_ids, target_ids)
        batch_size = int(start_positions.shape[0])
        total_loss += float(batch_loss) * batch_size
        total_examples += batch_size

    return total_loss / total_examples


def _decode_token_ids_for_sample(tokenizer: BPEModel, token_ids: list[int]) -> str:
    decoded = b"".join(tokenizer.vocab[token_id] for token_id in token_ids)
    return decoded.decode("utf-8", errors="replace")
