# Experiments (Learning-First)

Each milestone is a standalone script.

- `001_bigram_bt.py`: first language model baseline.
- `002_mlp_1hidden_bt.py`: first nonlinear upgrade over bigram.

## Run
1. Build native extension:
   - `make build`
2. Make sure dataset exists:
   - `datasets/tinyshakespeare.txt`
3. Run scripts:
   - `/Users/marcoshernanz/dev/baretensor/.venv/bin/python experiments/001_bigram_bt.py`
   - `/Users/marcoshernanz/dev/baretensor/.venv/bin/python experiments/002_mlp_1hidden_bt.py`

## Example output (001_bigram_bt.py)
```text
vocab_size=65
cross_entropy=2.454943
perplexity=11.645772
```

## What is perplexity?
Perplexity is `exp(cross_entropy)`.

Interpretation: average branching factor (effective number of next-token choices).
- Lower is better.
- `1.0` is perfect prediction.
- `11.65` means the model is, on average, as uncertain as choosing among about 11.65 plausible next tokens.
