from pathlib import Path

from flax import linen as nn
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


class CausalSelfAttention(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        queries = nn.Dense(ATTENTION_DIM, use_bias=False, name="query")(x)
        keys = nn.Dense(ATTENTION_DIM, use_bias=False, name="key")(x)
        values = nn.Dense(ATTENTION_DIM, use_bias=False, name="value")(x)

        scores = (queries @ jnp.swapaxes(keys, -1, -2)) / jnp.sqrt(ATTENTION_DIM)
        causal_mask = jnp.triu(jnp.ones((x.shape[-2], x.shape[-2]), dtype=bool), k=1)
        masked_scores = jnp.where(causal_mask, -jnp.inf, scores)
        attention_weights = jnn.softmax(masked_scores, axis=-1)
        mixed_values = attention_weights @ values
        return nn.Dense(EMBEDDING_DIM, use_bias=False, name="output")(mixed_values)


class FeedForward(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        hidden = nn.Dense(HIDDEN_DIM, name="hidden")(x)
        hidden = jnp.tanh(hidden)
        return nn.Dense(EMBEDDING_DIM, name="output")(hidden)


class LanguageModel(nn.Module):
    vocab_size: int

    @nn.compact
    def __call__(self, input_ids: jax.Array) -> jax.Array:
        positions = jnp.arange(input_ids.shape[1], dtype=jnp.int32)
        token_embeddings = nn.Embed(self.vocab_size, EMBEDDING_DIM, name="token_embeddings")(input_ids)
        position_embeddings = nn.Embed(CONTEXT_WINDOW, EMBEDDING_DIM, name="position_embeddings")(
            positions
        )
        embeddings = token_embeddings + position_embeddings

        attention = nn.LayerNorm(epsilon=LAYER_NORM_EPS, name="attention_norm")(
            embeddings + CausalSelfAttention(name="attention")(embeddings)
        )
        transformer = nn.LayerNorm(epsilon=LAYER_NORM_EPS, name="feed_forward_norm")(
            attention + FeedForward(name="feed_forward")(attention)
        )

        return nn.Dense(self.vocab_size, name="logits")(transformer)


def loss_fn(
    params: dict[str, jax.Array], model: LanguageModel, input_ids: jax.Array, target_ids: jax.Array
) -> jax.Array:
    logits = model.apply({"params": params}, input_ids)
    log_probs = jnn.log_softmax(logits, axis=-1)
    loss_per_token = -jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1).squeeze(-1)
    return loss_per_token.mean()


@jax.jit
def train_step(
    params: dict[str, jax.Array], model: LanguageModel, input_ids: jax.Array, target_ids: jax.Array
) -> tuple[dict[str, jax.Array], jax.Array]:
    loss, grads = jax.value_and_grad(loss_fn)(params, model, input_ids, target_ids)
    params = jax.tree_util.tree_map(lambda param, grad: param - LEARNING_RATE * grad, params, grads)
    return params, loss


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

model = LanguageModel(vocab_size=len(vocab_chars))
key, model_rng = jax.random.split(key)
params = model.init(model_rng, jnp.zeros((BATCH_SIZE, CONTEXT_WINDOW), dtype=jnp.int32))["params"]

for step in range(TRAIN_STEPS):
    key, batch_key = jax.random.split(key)
    input_ids, target_ids = sample_batch(batch_key, token_ids)
    params, loss = train_step(params, model, input_ids, target_ids)

    if step % 100 == 0:
        print(f"step={step} loss={loss.item():.4f}")
