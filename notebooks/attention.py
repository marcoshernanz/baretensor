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
key, embedding_key, position_key, Wq_key, Wk_key, Wv_key, Wo_key = jax.random.split(key, 7)

value_embeddings = jax.random.normal(embedding_key, (VOCAB_SIZE, EMBEDDING_DIM))
position_embeddings = jax.random.normal(position_key, (SEQUENCE_LEN, EMBEDDING_DIM))
Wq = jax.random.normal(Wq_key, (EMBEDDING_DIM, ATTENTION_DIM))
Wk = jax.random.normal(Wk_key, (EMBEDDING_DIM, ATTENTION_DIM))
Wv = jax.random.normal(Wv_key, (EMBEDDING_DIM, ATTENTION_DIM))
Wo = jax.random.normal(Wo_key, (ATTENTION_DIM, EMBEDDING_DIM))


# %%

token_vectors = value_embeddings[tokens]
position_vectors = position_embeddings[jnp.arange(SEQUENCE_LEN)]
E = token_vectors + position_vectors
Q = E @ Wq
K = E @ Wk
V = E @ Wv

x = (Q @ K.T) / jnp.sqrt(ATTENTION_DIM)
mask = jnp.triu(jnp.ones((SEQUENCE_LEN, SEQUENCE_LEN), dtype=bool), k=1)
masked = jnp.where(mask, -jnp.inf, x)
normalized = jnn.softmax(masked, -1)
attention = normalized @ V
output = attention @ Wo
