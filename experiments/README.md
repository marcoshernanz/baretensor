# Experiments (Learning-First)

Each milestone is a standalone script.

- `001_bigram_bt.py`: first language model baseline.
- `002_mlp_1hidden_bt.py`: first nonlinear upgrade over bigram.

## Run
1. Build native extension:
   - `make build`
2. Run scripts:
   - `PYTHONPATH=src uv run python experiments/001_bigram_bt.py`
   - `PYTHONPATH=src uv run python experiments/002_mlp_1hidden_bt.py`

Both scripts auto-download Tiny Shakespeare to `.cache/tinyshakespeare/input.txt` if missing.
