from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import jax.nn as jnn
import optax

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


class CausalSelfAttention(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.query = nnx.Linear(EMBEDDING_DIM, ATTENTION_DIM, use_bias=False, rngs=rngs)
        self.key = nnx.Linear(EMBEDDING_DIM, ATTENTION_DIM, use_bias=False, rngs=rngs)
        self.value = nnx.Linear(EMBEDDING_DIM, ATTENTION_DIM, use_bias=False, rngs=rngs)
        self.output = nnx.Linear(ATTENTION_DIM, EMBEDDING_DIM, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = (queries @ jnp.swapaxes(keys, -1, -2)) / jnp.sqrt(ATTENTION_DIM)
        causal_mask = jnp.triu(jnp.ones((x.shape[-2], x.shape[-2]), dtype=bool), k=1)
        masked_scores = jnp.where(causal_mask, -jnp.inf, scores)
        attention_weights = jnn.softmax(masked_scores, axis=-1)
        mixed_values = attention_weights @ values
        return self.output(mixed_values)


class FeedForward(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.hidden = nnx.Linear(EMBEDDING_DIM, HIDDEN_DIM, rngs=rngs)
        self.output = nnx.Linear(HIDDEN_DIM, EMBEDDING_DIM, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        hidden = jnp.tanh(self.hidden(x))
        return self.output(hidden)


class LanguageModel(nnx.Module):
    def __init__(self, vocab_size: int, *, rngs: nnx.Rngs):
        self.token_embeddings = nnx.Embed(vocab_size, EMBEDDING_DIM, rngs=rngs)
        self.position_embeddings = nnx.Embed(CONTEXT_WINDOW, EMBEDDING_DIM, rngs=rngs)
        self.attention = CausalSelfAttention(rngs=rngs)
        self.attention_norm = nnx.LayerNorm(EMBEDDING_DIM, epsilon=LAYER_NORM_EPS, rngs=rngs)
        self.feed_forward = FeedForward(rngs=rngs)
        self.feed_forward_norm = nnx.LayerNorm(EMBEDDING_DIM, epsilon=LAYER_NORM_EPS, rngs=rngs)
        self.logits = nnx.Linear(EMBEDDING_DIM, vocab_size, rngs=rngs)

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        positions = jnp.arange(input_ids.shape[1], dtype=jnp.int32)
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(positions)
        embeddings = token_embeddings + position_embeddings

        attention = self.attention_norm(embeddings + self.attention(embeddings))
        transformer = self.feed_forward_norm(attention + self.feed_forward(attention))
        return self.logits(transformer)


def loss_fn(model: LanguageModel, input_ids: jax.Array, target_ids: jax.Array) -> jax.Array:
    logits = model(input_ids)
    log_probs = jnn.log_softmax(logits, axis=-1)
    loss_per_token = -jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1).squeeze(-1)
    return loss_per_token.mean()


@nnx.jit
def train_step(
    model: LanguageModel, optimizer: nnx.Optimizer, input_ids: jax.Array, target_ids: jax.Array
) -> jax.Array:
    loss, grads = nnx.value_and_grad(loss_fn)(model, input_ids, target_ids)
    optimizer.update(model, grads)
    return loss


def sample_batch(batch_key: jax.Array, token_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
    start_positions = jax.random.randint(
        batch_key, (BATCH_SIZE,), 0, token_ids.shape[0] - CONTEXT_WINDOW
    )
    input_positions = start_positions[:, None] + jnp.arange(CONTEXT_WINDOW)
    input_ids = token_ids[input_positions]
    target_ids = token_ids[input_positions + 1]
    return input_ids, target_ids


key = jax.random.key(SEED)
corpus = DATA_PATH.read_text(encoding="utf-8")
vocab_chars = sorted(set(corpus))
char_to_id = {char: idx for idx, char in enumerate(vocab_chars)}
token_ids = jnp.asarray([char_to_id[ch] for ch in corpus], dtype=jnp.int32)

model = LanguageModel(len(vocab_chars), rngs=nnx.Rngs(SEED))
optimizer = nnx.Optimizer(model, optax.sgd(LEARNING_RATE), wrt=nnx.Param)

for step in range(TRAIN_STEPS):
    key, batch_key = jax.random.split(key)
    input_ids, target_ids = sample_batch(batch_key, token_ids)
    loss = train_step(model, optimizer, input_ids, target_ids)

    if step % 100 == 0:
        print(f"step={step} loss={loss.item():.4f}")
