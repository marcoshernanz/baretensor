from pathlib import Path
import math

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


def layer_norm(x: jax.Array, scale: jax.Array, shift: jax.Array) -> jax.Array:
    mean = x.mean(axis=-1, keepdims=True)
    variance = x.var(axis=-1, keepdims=True)
    normalized = (x - mean) / jnp.sqrt(variance + LAYER_NORM_EPS)
    return scale * normalized + shift


def init_params(rng: jax.Array, vocab_size: int) -> dict[str, jax.Array]:
    (
        embedding_rng,
        position_rng,
        query_rng,
        key_rng,
        value_rng,
        output_rng,
        attention_norm_rng,
        ffn_hidden_rng,
        ffn_output_rng,
        ffn_norm_rng,
        logit_rng,
    ) = jax.random.split(rng, 11)

    del attention_norm_rng, ffn_norm_rng

    return {
        "token_embeddings": jax.random.normal(embedding_rng, (vocab_size, EMBEDDING_DIM)),
        "position_embeddings": jax.random.normal(position_rng, (CONTEXT_WINDOW, EMBEDDING_DIM)),
        "query_weights": jax.random.normal(query_rng, (EMBEDDING_DIM, ATTENTION_DIM)),
        "key_weights": jax.random.normal(key_rng, (EMBEDDING_DIM, ATTENTION_DIM)),
        "value_weights": jax.random.normal(value_rng, (EMBEDDING_DIM, ATTENTION_DIM)),
        "attention_output_weights": jax.random.normal(
            output_rng, (ATTENTION_DIM, EMBEDDING_DIM)
        ),
        "attention_norm_scale": jnp.ones((EMBEDDING_DIM,)),
        "attention_norm_shift": jnp.zeros((EMBEDDING_DIM,)),
        "ffn_hidden_weights": jax.random.normal(ffn_hidden_rng, (EMBEDDING_DIM, HIDDEN_DIM)),
        "ffn_hidden_bias": jnp.zeros((HIDDEN_DIM,)),
        "ffn_output_weights": jax.random.normal(ffn_output_rng, (HIDDEN_DIM, EMBEDDING_DIM)),
        "ffn_output_bias": jnp.zeros((EMBEDDING_DIM,)),
        "ffn_norm_scale": jnp.ones((EMBEDDING_DIM,)),
        "ffn_norm_shift": jnp.zeros((EMBEDDING_DIM,)),
        "logit_weights": jax.random.normal(logit_rng, (EMBEDDING_DIM, vocab_size))
        * (1.0 / math.sqrt(EMBEDDING_DIM)),
        "logit_bias": jnp.zeros((vocab_size,)),
    }


def forward(params: dict[str, jax.Array], input_ids: jax.Array) -> jax.Array:
    positions = jnp.arange(input_ids.shape[1], dtype=jnp.int32)
    token_embeddings = params["token_embeddings"][input_ids]
    position_embeddings = params["position_embeddings"][positions]
    embeddings = token_embeddings + position_embeddings

    queries = embeddings @ params["query_weights"]
    keys = embeddings @ params["key_weights"]
    values = embeddings @ params["value_weights"]

    scores = (queries @ keys.mT) / jnp.sqrt(ATTENTION_DIM)
    causal_mask = jnp.triu(jnp.ones((input_ids.shape[1], input_ids.shape[1]), dtype=bool), k=1)
    masked_scores = jnp.where(causal_mask, -jnp.inf, scores)
    attention_weights = jnn.softmax(masked_scores, axis=-1)
    mixed_values = attention_weights @ values
    attention_output = mixed_values @ params["attention_output_weights"]
    attention_residual = embeddings + attention_output
    attention_block = layer_norm(
        attention_residual,
        params["attention_norm_scale"],
        params["attention_norm_shift"],
    )

    ffn_hidden = jnp.tanh(
        attention_block @ params["ffn_hidden_weights"] + params["ffn_hidden_bias"]
    )
    ffn_output = ffn_hidden @ params["ffn_output_weights"] + params["ffn_output_bias"]
    ffn_residual = attention_block + ffn_output
    block_output = layer_norm(ffn_residual, params["ffn_norm_scale"], params["ffn_norm_shift"])

    return block_output @ params["logit_weights"] + params["logit_bias"]


def loss_fn(params: dict[str, jax.Array], input_ids: jax.Array, target_ids: jax.Array) -> jax.Array:
    logits = forward(params, input_ids)
    log_probs = jnn.log_softmax(logits, axis=-1)
    loss_per_token = -jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1).squeeze(-1)
    return loss_per_token.mean()


@jax.jit
def train_step(
    params: dict[str, jax.Array], input_ids: jax.Array, target_ids: jax.Array
) -> tuple[dict[str, jax.Array], jax.Array]:
    loss, grads = jax.value_and_grad(loss_fn)(params, input_ids, target_ids)
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

key, model_rng = jax.random.split(key)
params = init_params(model_rng, len(vocab_chars))

for step in range(TRAIN_STEPS):
    key, batch_key = jax.random.split(key)
    input_ids, target_ids = sample_batch(batch_key, token_ids)
    params, loss = train_step(params, input_ids, target_ids)

    if step % 100 == 0:
        print(f"step={step} loss={loss.item():.4f}")
