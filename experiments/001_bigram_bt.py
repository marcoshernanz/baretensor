# %%
from pathlib import Path

import torch

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"
tokens = DATA_PATH.read_text(encoding="utf-8")

chars = sorted(set(tokens))

ctoi = {c: i for i, c in enumerate(chars)}
itoc = {i: c for i, c in enumerate(chars)}
num_tokens = len(ctoi)

encoded = torch.tensor([ctoi[c] for c in tokens], dtype=torch.long)

# %%
bigram = torch.ones([num_tokens, num_tokens])

for t1, t2 in zip(encoded, encoded[1:]):
    bigram[t1, t2] += 1

# %%
row_sums = bigram.sum(1, keepdim=True)
probs = bigram / row_sums

t1 = encoded[:-1]
t2 = encoded[1:]
loss = -torch.log(probs[t1, t2]).mean()

print(loss.item())
