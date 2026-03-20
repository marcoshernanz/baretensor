# %%
import jax
import jax.numpy as jnp
import jax.nn as jnn

sequence = "This is a test sequence to try out self-attention"

SEQUENCE_LEN = len(sequence)
VOCAB_SIZE = len(set(sequence))
EMBEDDING_DIM = 128
ATTENTION_DIM = 32

# %%

char_to_id = {c: i for i, c in enumerate(set(sequence))}
tokens = jnp.array([char_to_id[c] for c in sequence])

# %%

key = jax.random.key(1337)
key, token_embedding_key, position_embedding_key, Wq_key, Wk_key, Wv_key, Wo_key = jax.random.split(key, 7)

token_embedding_table = jax.random.normal(token_embedding_key, (VOCAB_SIZE, EMBEDDING_DIM))
position_embedding_table = jax.random.normal(position_embedding_key, (SEQUENCE_LEN, EMBEDDING_DIM))
Wq = jax.random.normal(Wq_key, (EMBEDDING_DIM, ATTENTION_DIM))
Wk = jax.random.normal(Wk_key, (EMBEDDING_DIM, ATTENTION_DIM))
Wv = jax.random.normal(Wv_key, (EMBEDDING_DIM, ATTENTION_DIM))
Wo = jax.random.normal(Wo_key, (ATTENTION_DIM, EMBEDDING_DIM))


# %%

positions = jnp.arange(SEQUENCE_LEN)
token_embeddings = token_embedding_table[tokens]
position_embeddings = position_embedding_table[positions]
input_embeddings = token_embeddings + position_embeddings

queries = input_embeddings @ Wq
keys = input_embeddings @ Wk
values = input_embeddings @ Wv

scores = (queries @ keys.T) / jnp.sqrt(ATTENTION_DIM)
causal_mask = jnp.triu(jnp.ones((SEQUENCE_LEN, SEQUENCE_LEN), dtype=bool), k=1)
masked_scores = jnp.where(causal_mask, -jnp.inf, scores)
attention_weights = jnn.softmax(masked_scores, axis=-1)
attention_output = attention_weights @ values
output = attention_output @ Wo
