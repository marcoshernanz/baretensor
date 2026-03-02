# %%
import torch

tokens = open("../datasets/tinyshakespeare.txt", "r").read()

ttoi = {c: i for i, c in enumerate(set(tokens))}
itok = {i: c for i, c in enumerate(set(tokens))}
num_tokens = len(ttoi)

encoded = torch.tensor([ttoi[c] for c in tokens], dtype=torch.long)

# %%
bigram = torch.zeros([num_tokens, num_tokens])

for c1, c2 in zip(encoded, encoded[1:]):
    bigram[c1, c2] += 1

bigram /= bigram.sum(1, keepdim=True)

# %%
sum = 0
total = 0

for t1, t2 in zip(encoded, encoded[1:]):
    logits = bigram[t1]
    ce = torch.nn.functional.cross_entropy(logits, t2)
    sum += ce
    total += 1

loss = sum / total

print(loss)
