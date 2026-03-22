# %%
from pathlib import Path
from typing import TypeAlias

import jax
import jax.numpy as jnp
import jax.nn as jnn

# %%

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

Array: TypeAlias = jax.Array
Params: TypeAlias = dict[str, Array]

# %%

key = jax.random.key(SEED)
corpus = DATA_PATH.read_text(encoding="utf-8")

vocab_chars = sorted(set(corpus))
char_to_id = {char: idx for idx, char in enumerate(vocab_chars)}
vocab_size = len(char_to_id)

token_ids = jnp.array([char_to_id[ch] for ch in corpus], dtype=jnp.int32)
num_tokens = token_ids.shape[0]

# %%

(
    key,
    token_embedding_key,
    position_embedding_key,
    Wq_key,
    Wk_key,
    Wv_key,
    Wo_key,
    W_key,
    B_key,
    W1_key,
    B1_key,
    W2_key,
    B2_key,
) = jax.random.split(key, 13)

token_embedding_table = jax.random.normal(token_embedding_key, (vocab_size, EMBEDDING_DIM))
position_embedding_table = jax.random.normal(
    position_embedding_key, (CONTEXT_WINDOW, EMBEDDING_DIM)
)
Wq = jax.random.normal(Wq_key, (EMBEDDING_DIM, ATTENTION_DIM))
Wk = jax.random.normal(Wk_key, (EMBEDDING_DIM, ATTENTION_DIM))
Wv = jax.random.normal(Wv_key, (EMBEDDING_DIM, ATTENTION_DIM))
Wo = jax.random.normal(Wo_key, (ATTENTION_DIM, EMBEDDING_DIM))
W = jax.random.normal(W_key, (EMBEDDING_DIM, vocab_size))
B = jax.random.normal(B_key, (vocab_size,))
layer_norm_scale = jnp.ones((EMBEDDING_DIM,))
layer_norm_shift = jnp.zeros((EMBEDDING_DIM,))
W1 = jax.random.normal(W1_key, (EMBEDDING_DIM, HIDDEN_DIM))
B1 = jax.random.normal(B1_key, (HIDDEN_DIM,))
W2 = jax.random.normal(W2_key, (HIDDEN_DIM, EMBEDDING_DIM))
B2 = jax.random.normal(B2_key, (EMBEDDING_DIM,))
layer_norm_scale2 = jnp.ones((EMBEDDING_DIM,))
layer_norm_shift2 = jnp.zeros((EMBEDDING_DIM,))

params: Params = {
    "token_embedding_table": token_embedding_table,
    "position_embedding_table": position_embedding_table,
    "Wq": Wq,
    "Wk": Wk,
    "Wv": Wv,
    "Wo": Wo,
    "layer_norm_scale": layer_norm_scale,
    "layer_norm_shift": layer_norm_shift,
    "W1": W1,
    "B1": B1,
    "W2": W2,
    "B2": B2,
    "layer_norm_scale2": layer_norm_scale2,
    "layer_norm_shift2": layer_norm_shift2,
    "W": W,
    "B": B,
}

# %%


def sample_batch(batch_key: Array) -> tuple[Array, Array]:
    start_positions = jax.random.randint(batch_key, (BATCH_SIZE,), 0, num_tokens - CONTEXT_WINDOW)
    input_positions = start_positions[:, None] + jnp.arange(CONTEXT_WINDOW)
    input_ids = token_ids[input_positions]
    target_ids = token_ids[input_positions + 1]
    return input_ids, target_ids


def loss_fn(params: Params, input_ids: Array, target_ids: Array) -> Array:
    positions = jnp.arange(CONTEXT_WINDOW)
    token_embeddings = params["token_embedding_table"][input_ids]
    position_embeddings = params["position_embedding_table"][positions]
    input_embeddings = token_embeddings + position_embeddings

    queries = input_embeddings @ params["Wq"]
    keys = input_embeddings @ params["Wk"]
    values = input_embeddings @ params["Wv"]

    scores = (queries @ keys.mT) / jnp.sqrt(ATTENTION_DIM)
    causal_mask = jnp.triu(jnp.ones((CONTEXT_WINDOW, CONTEXT_WINDOW), dtype=bool), k=1)
    masked_scores = jnp.where(causal_mask, -jnp.inf, scores)
    attention_weights = jnn.softmax(masked_scores, axis=-1)
    attention_output = attention_weights @ values
    projected_attention = attention_output @ params["Wo"]
    residual_output = input_embeddings + projected_attention

    normalized_residual = (
        residual_output - residual_output.mean(axis=-1, keepdims=True)
    ) / jnp.sqrt(residual_output.var(axis=-1, keepdims=True) + LAYER_NORM_EPS)
    layer_norm_output = (
        params["layer_norm_scale"] * normalized_residual + params["layer_norm_shift"]
    )

    h1 = jnp.tanh(layer_norm_output @ params["W1"] + params["B1"])
    h2 = h1 @ params["W2"] + params["B2"]
    output = h2 + layer_norm_output
    normalized_output = (output - output.mean(axis=-1, keepdims=True)) / jnp.sqrt(
        output.var(axis=-1, keepdims=True) + LAYER_NORM_EPS
    )
    layer_norm_output2 = (
        params["layer_norm_scale2"] * normalized_output + params["layer_norm_shift2"]
    )

    logits = layer_norm_output2 @ params["W"] + params["B"]
    log_probs = -jnn.log_softmax(logits, axis=-1)
    loss_per_token = jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1).squeeze(-1)
    return loss_per_token.mean()


@jax.jit
def train_step(params: Params, input_ids: Array, target_ids: Array) -> tuple[Params, Array]:
    loss, grads = jax.value_and_grad(loss_fn)(params, input_ids, target_ids)
    updated_params = jax.tree_util.tree_map(
        lambda param, grad: param - LEARNING_RATE * grad,
        params,
        grads,
    )
    return updated_params, loss


# %%

for step in range(TRAIN_STEPS):
    key, batch_key = jax.random.split(key, 2)
    input_ids, target_ids = sample_batch(batch_key)
    params, loss = train_step(params, input_ids, target_ids)

    if step % 100 == 0:
        print(f"step={step} loss={loss.item():.4f}")
