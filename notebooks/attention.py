# %%
import jax.numpy as jnp
import jax

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
key, embedding_key, Wq_key, Wk_key, Wv_key = jax.random.split(key, 5)

embeddings = jax.random.normal(embedding_key, (VOCAB_SIZE, EMBEDDING_DIM))
Wq = jax.random.normal(Wq_key, (EMBEDDING_DIM, ATTENTION_DIM))
Wk = jax.random.normal(Wk_key, (EMBEDDING_DIM, ATTENTION_DIM))

E = embeddings[tokens]
Q = E @ Wq
K = E @ Wk

x = (Q @ K.T) / jnp.sqrt(SEQUENCE_LEN)
