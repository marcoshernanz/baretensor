# Learning-First LLM Roadmap

## Objective
Learn by building small language models end-to-end, then progressively move ideas into the BareTensor library only when the pain is real.

## Fixed Baseline Rules
- Keep dataset fixed while comparing milestones.
- Keep loss fixed to next-token cross-entropy.
- Log every run in `docs/learning_log.md`.
- Change one variable per milestone.

## Work Streams
- `experiments/`: fast learning scripts, one milestone per file.
- `src/bt/` and `native/`: reusable API and backend work, introduced later.

## Stage Plan

### Stage 0: Script-Only Learning (now)
Target milestones:
1. Bigram.
2. 1-hidden-layer MLP.
3. Better sampling and diagnostics.
4. Checkpoint save/load.
5. First tiny attention model.

Where to change code:
- Only `experiments/*.py` and `docs/learning_log.md`.

Do not do yet:
- No `train.py` orchestration framework.
- No config system.
- No tokenizer training.
- No CUDA kernels.

Why:
- Maximum learning velocity with minimal abstraction overhead.

### Stage 1: Introduce Tokenizer (after first attention model is stable)
Trigger:
- Char-level model clearly plateaus and you can run stable train/val loops.

What to add:
- Start with a simple BPE tokenizer script and frozen artifacts.
- Keep the same loss and eval process.

Where to change code:
- `scripts/` for tokenizer build script.
- `artifacts/tokenizer/...` for vocab/merges/tokenizer files.
- Minimal updates in `experiments/` to use token IDs.

Do not do yet:
- No library-wide `nn` refactor.
- No CUDA kernel work.

### Stage 2: Add `torch.nn`-like Layer API (when repetition hurts)
Trigger:
- You have copied linear/embedding/layernorm/optimizer logic across 3+ experiment files.

What to implement first:
- Minimal module abstraction in Python:
  - `Module`
  - `Linear`
  - `Embedding`
  - `LayerNorm`
- Keep it tiny and close to PyTorch naming.

Where to change code:
- `src/bt/nn/` (new module/layer files).
- `tests/` for layer forward/backward parity tests.
- `experiments/` updated to use these layers.

Notes:
- This is API ergonomics work, not backend optimization work.

### Stage 3: Move to Robust Runner (`train.py` + config)
Trigger:
- You run many long experiments and need reproducibility across machines.
- You keep editing the same boilerplate loop in every script.

What to add:
- One `train.py` entrypoint.
- One config format only (TOML).
- Profiles for `dev-cpu`, `local-2060`, `cloud`.

Where to change code:
- `scripts/train.py`
- `configs/*.toml`
- Reuse model code from `src/bt/nn/` and keep `experiments/` for prototypes.

Rule:
- Do this only after Stage 2 trigger is reached.

### Stage 4: Custom CUDA Kernels (only after model semantics are stable)
Trigger:
- Stable architecture and training stack already exist.
- Profiling shows clear bottlenecks on RTX 2060.

What to implement first:
1. CUDA matmul baseline.
2. CUDA softmax/log_softmax.
3. CUDA layer norm.
4. Fused cross-entropy.

Where to change code:
- `native/src/` and `native/include/`.
- Python bindings in `native/python/bindings.cpp` if new ops are exposed.
- Add parity tests under `tests/` against CPU/reference.

Rule:
- Every kernel change must include correctness tests before performance claims.

## Decision Checklist (quick)
- Need to learn model ideas quickly? -> stay in `experiments/`.
- Repeating layer code everywhere? -> Stage 2 (`src/bt/nn`).
- Repeating training boilerplate everywhere? -> Stage 3 (`train.py` + config).
- Training is correct but too slow on GPU? -> Stage 4 (CUDA kernels).
- Char-level quality bottleneck? -> Stage 1 (tokenizer).

## Practical Sequence For You
1. Finish bigram and MLP results in `docs/learning_log.md`.
2. Build first tiny attention script in `experiments/`.
3. Add tokenizer only after attention baseline is stable.
4. Add minimal `bt.nn` layers once copy-paste pain is real.
5. Add `train.py` and TOML configs after `bt.nn` is in place.
6. Start CUDA kernels last, after profiling identifies bottlenecks.

## Start Commands
- Build extension: `make build`
- Run bigram: `/Users/marcoshernanz/dev/baretensor/.venv/bin/python experiments/001_bigram_bt.py`
- Run MLP: `/Users/marcoshernanz/dev/baretensor/.venv/bin/python experiments/002_mlp_1hidden_bt.py`
