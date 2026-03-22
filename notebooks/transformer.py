from pathlib import Path
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as jnn

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"
SEED = 1337
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
CONTEXT_WINDOW = 512
ATTENTION_DIM = 32
LEARNING_RATE = 0.05
TRAIN_STEPS = 5_000
LAYER_NORM_EPS = 1e-5


class LayerNorm(eqx.Module):
    scale: jax.Array
    shift: jax.Array

    def __init__(self):
        self.scale = jnp.ones((EMBEDDING_DIM,))
        self.shift = jnp.zeros((EMBEDDING_DIM,))

    def __call__(self, x: jax.Array) -> jax.Array:
        mean = x.mean(axis=-1, keepdims=True)
        variance = x.var(axis=-1, keepdims=True)
        normalized = (x - mean) / jnp.sqrt(variance + LAYER_NORM_EPS)
        return self.scale * normalized + self.shift


class CausalSelfAttention(eqx.Module):
    query_weights: jax.Array
    key_weights: jax.Array
    value_weights: jax.Array
    output_weights: jax.Array

    def __init__(self, rng: jax.Array):
        query_rng, key_rng, value_rng, output_rng = jax.random.split(rng, 4)
        self.query_weights = jax.random.normal(query_rng, (EMBEDDING_DIM, ATTENTION_DIM))
        self.key_weights = jax.random.normal(key_rng, (EMBEDDING_DIM, ATTENTION_DIM))
        self.value_weights = jax.random.normal(value_rng, (EMBEDDING_DIM, ATTENTION_DIM))
        self.output_weights = jax.random.normal(output_rng, (ATTENTION_DIM, EMBEDDING_DIM))

    def __call__(self, x: jax.Array) -> jax.Array:
        queries = x @ self.query_weights
        keys = x @ self.key_weights
        values = x @ self.value_weights

        scores = (queries @ keys.mT) / jnp.sqrt(ATTENTION_DIM)
        causal_mask = jnp.triu(jnp.ones((x.shape[-2], x.shape[-2]), dtype=bool), k=1)
        masked_scores = jnp.where(causal_mask, -jnp.inf, scores)
        attention_weights = jnn.softmax(masked_scores, axis=-1)
        mixed_values = attention_weights @ values
        return mixed_values @ self.output_weights


class FeedForward(eqx.Module):
    hidden_weights: jax.Array
    hidden_bias: jax.Array
    output_weights: jax.Array
    output_bias: jax.Array

    def __init__(self, rng: jax.Array):
        hidden_rng, output_rng = jax.random.split(rng, 2)
        self.hidden_weights = jax.random.normal(hidden_rng, (EMBEDDING_DIM, HIDDEN_DIM))
        self.hidden_bias = jnp.zeros((HIDDEN_DIM,))
        self.output_weights = jax.random.normal(output_rng, (HIDDEN_DIM, EMBEDDING_DIM))
        self.output_bias = jnp.zeros((EMBEDDING_DIM,))

    def __call__(self, x: jax.Array) -> jax.Array:
        hidden = jnp.tanh(x @ self.hidden_weights + self.hidden_bias)
        return hidden @ self.output_weights + self.output_bias


class DecoderBlock(eqx.Module):
    attention: CausalSelfAttention
    attention_norm: LayerNorm
    feed_forward: FeedForward
    feed_forward_norm: LayerNorm

    def __init__(self, rng: jax.Array):
        attention_rng, feed_forward_rng = jax.random.split(rng, 2)
        self.attention = CausalSelfAttention(attention_rng)
        self.attention_norm = LayerNorm()
        self.feed_forward = FeedForward(feed_forward_rng)
        self.feed_forward_norm = LayerNorm()

    def __call__(self, x: jax.Array) -> jax.Array:
        attention_block_output = self.attention_norm(x + self.attention(x))
        return self.feed_forward_norm(
            attention_block_output + self.feed_forward(attention_block_output)
        )


class LanguageModel(eqx.Module):
    token_embeddings: jax.Array
    position_embeddings: jax.Array
    decoder_block: DecoderBlock
    logit_weights: jax.Array
    logit_bias: jax.Array

    def __init__(self, rng: jax.Array, vocab_size: int):
        embedding_rng, position_rng, transformer_rng, logits_rng = jax.random.split(rng, 4)

        self.token_embeddings = jax.random.normal(embedding_rng, (vocab_size, EMBEDDING_DIM))
        self.position_embeddings = jax.random.normal(position_rng, (CONTEXT_WINDOW, EMBEDDING_DIM))
        self.decoder_block = DecoderBlock(transformer_rng)
        self.logit_weights = jax.random.normal(logits_rng, (EMBEDDING_DIM, vocab_size)) * (
            1.0 / math.sqrt(EMBEDDING_DIM)
        )
        self.logit_bias = jnp.zeros((vocab_size,))

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        positions = jnp.arange(input_ids.shape[-1], dtype=jnp.int32)
        token_embeddings = self.token_embeddings[input_ids]
        position_embeddings = self.position_embeddings[positions]
        decoder_input = token_embeddings + position_embeddings
        decoder_output = self.decoder_block(decoder_input)
        return decoder_output @ self.logit_weights + self.logit_bias


@eqx.filter_value_and_grad
def loss_fn(model: LanguageModel, input_ids: jax.Array, target_ids: jax.Array) -> jax.Array:
    logits = model(input_ids)
    log_probs = jnn.log_softmax(logits, axis=-1)
    loss_per_token = -jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1).squeeze(-1)
    return loss_per_token.mean()


@eqx.filter_jit
def train_step(
    model: LanguageModel, input_ids: jax.Array, target_ids: jax.Array
) -> tuple[LanguageModel, jax.Array]:
    loss, grads = loss_fn(model, input_ids, target_ids)
    updates = jax.tree_util.tree_map(lambda grad: -LEARNING_RATE * grad, grads)
    model = eqx.apply_updates(model, updates)
    return model, loss


def sample_batch(batch_key: jax.Array, token_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
    max_start = token_ids.shape[0] - CONTEXT_WINDOW
    start_positions = jax.random.randint(batch_key, (BATCH_SIZE,), 0, max_start)
    input_positions = start_positions[:, None] + jnp.arange(CONTEXT_WINDOW)
    input_ids = token_ids[input_positions]
    target_ids = token_ids[input_positions + 1]
    return input_ids, target_ids


key = jax.random.key(SEED)
corpus = DATA_PATH.read_text(encoding="utf-8")
vocab_chars = sorted(set(corpus))
char_to_id = {char: idx for idx, char in enumerate(vocab_chars)}
token_ids = jnp.asarray([char_to_id[ch] for ch in corpus], dtype=jnp.int32)

key, model_rng = jax.random.split(key)
model = LanguageModel(model_rng, len(vocab_chars))

for step in range(TRAIN_STEPS):
    key, batch_key = jax.random.split(key)
    input_ids, target_ids = sample_batch(batch_key, token_ids)
    model, loss = train_step(model, input_ids, target_ids)

    if step % 100 == 0:
        print(f"step={step} loss={loss.item():.4f}")
