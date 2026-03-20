# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.nn as jnn

# %%

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"
SEED = 1337
BATCH_SIZE = 64
EMBEDDING_DIM = 128
CONTEXT_WINDOW = 512
ATTENTION_DIM = 32
LEARNING_RATE = 0.05
TRAIN_STEPS = 5_000

# %%

key = jax.random.key(SEED)
corpus = DATA_PATH.read_text(encoding="utf-8")

vocab_chars = sorted(set(corpus))
char_to_id = {char: idx for idx, char in enumerate(vocab_chars)}
vocab_size = len(char_to_id)

token_ids = jnp.array([char_to_id[ch] for ch in corpus], dtype=jnp.int32)
num_tokens = token_ids.shape[0]

# %%

key, token_embedding_key, position_embedding_key, Wq_key, Wk_key, Wv_key, Wo_key, W_key, B_key = (
    jax.random.split(key, 9)
)

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

# %%

# for i in range(TRAIN_STEPS):
for i in range(1):
    key, batch_key = jax.random.split(key, 2)
    start_positions = jax.random.randint(batch_key, (BATCH_SIZE,), 0, num_tokens - CONTEXT_WINDOW)
    input_positions = start_positions[:, None] + jnp.arange(CONTEXT_WINDOW)
    input_ids = token_ids[input_positions]
    target_ids = token_ids[input_positions + 1]

    positions = jnp.arange(CONTEXT_WINDOW)
    token_embeddings = token_embedding_table[input_ids]
    position_embeddings = position_embedding_table[positions]
    input_embeddings = token_embeddings + position_embeddings

    queries = input_embeddings @ Wq
    keys = input_embeddings @ Wk
    values = input_embeddings @ Wv

    scores = (queries @ keys.mT) / jnp.sqrt(ATTENTION_DIM)
    causal_mask = jnp.triu(jnp.ones((CONTEXT_WINDOW, CONTEXT_WINDOW), dtype=bool), k=1)
    masked_scores = jnp.where(causal_mask, -jnp.inf, scores)
    attention_weights = jnn.softmax(masked_scores, axis=-1)
    attention_output = attention_weights @ values
    output = attention_output @ Wo
    print(output.shape)


# %%

# positions = jnp.arange(SEQUENCE_LEN)
# token_embeddings = token_embedding_table[tokens]
# position_embeddings = position_embedding_table[positions]
# input_embeddings = token_embeddings + position_embeddings

# queries = input_embeddings @ Wq
# keys = input_embeddings @ Wk
# values = input_embeddings @ Wv

# scores = (queries @ keys.T) / jnp.sqrt(ATTENTION_DIM)
# causal_mask = jnp.triu(jnp.ones((SEQUENCE_LEN, SEQUENCE_LEN), dtype=bool), k=1)
# masked_scores = jnp.where(causal_mask, -jnp.inf, scores)
# attention_weights = jnn.softmax(masked_scores, axis=-1)
# attention_output = attention_weights @ values
# output = attention_output @ Wo
