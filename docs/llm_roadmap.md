# BareTensor LLM Roadmap (End-to-End Stack Mastery)

## Primary goal
Build and understand the full AI stack in this repo, from tensor internals to modern architectures.

This roadmap is not about using external frameworks long-term. It is about owning:
- C++ tensor core,
- autograd,
- Python API,
- NN layer abstractions,
- tokenizer pipeline,
- CUDA kernels,
- training/evaluation systems,
- LLM architecture choices.

## Hard constraints
- Keep external dependencies minimal (near-zero target).
- `nanobind` is the required bridge for bindings.
- BareTensor-native implementations are the default path.
- Temporary placeholders are allowed only to unblock learning, then must be replaced.

## Project structure intent
- `native/`: C++ tensor and kernel backend.
- `src/bt/`: Python API surface.
- `src/bt/nn/`: high-level NN abstractions.
- `experiments/`: milestone scripts and quick prototypes.
- `docs/learning_log.md`: run history and metric tracking.

## Stage-by-stage plan

### Stage A: Bare minimum LM loop (current)
Goal:
- run simple language-model baselines and understand loss/gradients.

Milestones:
1. Bigram LM.
2. 1-hidden-layer MLP LM.
3. Train/val split tracking + sample generation.

Where to edit:
- `experiments/` first.

Exit criteria:
- You can explain every tensor shape and gradient path in your scripts.
- Baselines produce stable cross-entropy values and samples.

### Stage B: Move baseline logic into BareTensor-first code
Goal:
- remove placeholder framework usage from core milestone path.

Milestones:
1. Port bigram and MLP logic to BareTensor-native training path.
2. Keep loss fixed: next-token cross-entropy.
3. Keep dataset fixed while comparing improvements.

Where to edit:
- `experiments/` and `src/bt/` usage paths.

Exit criteria:
- Baselines run with BareTensor as the core tensor/autograd runtime.

### Stage C: Introduce tokenizer
When to do it:
- after the first attention-capable baseline is stable and char-level bottlenecks are visible.

What to implement:
1. Basic BPE tokenizer training script.
2. Frozen tokenizer artifacts (vocab/merges/config).
3. Deterministic encode/decode tests.

Where to edit:
- `scripts/` for tokenizer tooling.
- `artifacts/tokenizer/` for frozen tokenizer files.
- `experiments/` for tokenized training data path.

### Stage D: Build `bt.nn`-style layer API (PyTorch-like ergonomics)
When to do it:
- once repeated layer code appears across multiple experiments.

What to implement first:
1. `Module` base class.
2. `Linear`.
3. `Embedding`.
4. `LayerNorm`.
5. Minimal optimizer abstractions.

Where to edit:
- `src/bt/nn/`.
- `tests/` for correctness/parity.

Why here:
- this is where prototyping turns into reusable architecture code.

### Stage E: Introduce robust runner (`train.py` + single config format)
When to do it:
- after `bt.nn` exists and repeated training boilerplate slows iteration.

What to implement:
1. One `scripts/train.py` entrypoint.
2. One config format only (recommended TOML).
3. Reproducible logging, checkpointing, and profile presets.

Where to edit:
- `scripts/train.py`
- `configs/*.toml`

Rule:
- Do not add this before Stage D; otherwise infra complexity outpaces learning.

### Stage F: Architecture climb (Tiny GPT -> modern blocks)
Goal:
- progress from simple models to modern Transformer patterns.

Milestones:
1. Single-head causal self-attention.
2. Multi-head attention + residual + LayerNorm.
3. Stacked decoder-only Transformer.
4. Positional strategies and improved FFN variants.

Where to edit:
- primarily `src/bt/nn/` and experiment/train entrypoints.

### Stage G: Custom CUDA kernels
When to start:
- only after model/training semantics are stable and profiler shows bottlenecks.

Kernel order:
1. Matmul baseline.
2. Softmax/log_softmax.
3. LayerNorm.
4. Fused cross-entropy.
5. Attention-oriented kernels.

Where to edit:
- `native/src/`, `native/include/`, `native/python/bindings.cpp`.
- add CPU/CUDA parity tests in `tests/`.

Rule:
- correctness parity first, performance claims second.

## Transition checklist
Use these to decide what to do next.

- Tokenizer now?
  - Yes only if char-level has clearly plateaued and you already have a stable attention baseline.

- `bt.nn` now?
  - Yes when copy-paste layer code is frequent and slowing progress.

- `train.py` + config now?
  - Yes when experiments are numerous and reproducibility overhead is hurting momentum.

- Custom CUDA kernels now?
  - Yes only with stable model semantics + profiling evidence of hotspots.

## Practical next steps (immediate)
1. Keep logging baseline runs in `docs/learning_log.md`.
2. Continue milestone scripts in `experiments/` while porting logic toward BareTensor-native paths.
3. Add a first tiny attention milestone.
4. Start Stage D (`bt.nn`) only when repetition is concrete, not hypothetical.
