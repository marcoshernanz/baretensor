# BareTensor LLM Roadmap

This roadmap is intentionally small-step.
The purpose is to reduce leap size between milestones and make it obvious:
- what model comes next,
- what BareTensor work is required for that model,
- when to pause model work for library work,
- when to start CS229 and CS224N.

## Current Position
As of 2026-03-07, you are between `002` and `003`.

That means:
- `001` bigram is done.
- `002` 1-hidden-layer MLP is done or nearly done.
- You should start CS229 now.
- You should not implement `bt.nn` yet.
- The next move should be the first attention prototype without `bt.nn`.

## Principles
- Learning first: optimize for understanding and correctness.
- BareTensor-first: do not hide milestone logic behind abstractions too early.
- Milestones should be runnable scripts.
- Infrastructure breaks should happen only after the need is concrete.
- Do not start tokenizer or CUDA work before the model path earns it.

## Milestone 001: Bigram LM
Model:
- Simple character-level bigram model.

BareTensor work:
- Tensor construction.
- Basic indexing.
- `sum`.
- `log`.
- Sampling/readback via `numpy()`.

Exit criteria:
- Script runs end to end.
- Cross-entropy is stable.
- You can explain every tensor shape.

Course checkpoint:
- No course dependency required.

## Milestone 002: 1-Hidden-Layer MLP LM
Model:
- Character LM with one hidden layer and `tanh`.

BareTensor work:
- `matmul`.
- Elementwise autograd.
- `tanh`.
- `embedding`.
- `cross_entropy`.
- `no_grad`.
- In-place optimizer step under `no_grad`.

Exit criteria:
- Train/val/sample flow runs stably.
- You can explain the forward path and the backward path.
- It feels like a real BareTensor model, not a placeholder.

Course checkpoint:
- Start CS229 here.
- Recommended first pass:
  - Lecture 1: overview and course framing.
  - Lecture 2: linear regression and gradient descent.
  - Lecture 3: logistic regression and Newton's method.

## Milestone 003: MLP + 1 Causal Self-Attention Head
Model:
- Keep the `002` MLP path.
- Add one causal self-attention head.
- Keep it character-level.
- Do this without `bt.nn`.

Why this milestone exists:
- You want to feel the raw model mechanics before designing the reusable layer API.

BareTensor work:
- Attention score computation.
- Causal masking.
- Careful `transpose` / `permute` usage.
- Real-model `softmax` usage.

Exit criteria:
- A 1-head causal attention model runs end to end.
- You understand Q/K/V shapes, mask shape, and logits path.
- You can point to the code repetition that justifies `bt.nn`.

Course checkpoint:
- Continue CS229 if useful, especially later:
  - Lecture 10: introduction to neural networks.
  - Lecture 11: backprop and improving neural networks.
  - Lecture 12: debugging ML models and error analysis.
- Start CS224N here, because this is the first real attention decision point.
- Recommended first pass:
  - Lecture 1: intro and word vectors.
  - Lecture 2: word vectors and language models.
  - Lecture 3: backpropagation and neural networks.

## Break A: Implement `bt.nn`
Do this immediately after `003`.
At this point the repetition is no longer hypothetical.

What to implement first:
- `Module`.
- `Linear`.
- `Embedding`.
- `LayerNorm`.
- Minimal parameter traversal.

What not to build yet:
- No full training runner.
- No config system.
- No tokenizer.
- No optimizer hierarchy unless the current scripts really need it.

## Milestone 004: Rebuild 003 Using `bt.nn`
Model:
- Re-implement the same model from `003`.
- Same architecture goal.
- Cleaner code path through `bt.nn`.

Why this milestone exists:
- This is the proof that `bt.nn` is actually helping, not just extra infrastructure.

BareTensor work:
- Module composition.
- Parameter organization.
- Cleaner forward definitions.

Exit criteria:
- The `004` script is meaningfully cleaner than `003`.
- The model behavior is still understandable.
- You can now separate "model idea" from "framework plumbing".

Course checkpoint:
- Continue CS224N:
  - Lecture 7: attention and LLM intro.

## Milestone 005: Add Residuals and LayerNorm Around Attention
Model:
- Still a small character model.
- Add residual structure and normalization around the attention path.
- Do not build a full decoder block yet.

Why this milestone exists:
- It is a smaller step than jumping directly from `004` to a full Transformer block.

BareTensor work:
- Residual wiring.
- LayerNorm in a real architecture.
- Cleaner shape discipline around attention outputs.

Exit criteria:
- Residual + normalization path is stable.
- You can explain exactly where normalization sits and why.

Course checkpoint:
- Continue CS224N:
  - Lecture 8: self-attention and Transformers.

## Milestone 006: First Single-Block Decoder-Only Transformer
Model:
- Attention + residual + normalization + feedforward.
- One decoder block only.
- Still character-level.

BareTensor work:
- Feedforward block structure.
- Better module composition.
- Cleaner parameter reuse.

Exit criteria:
- A single decoder block trains and samples.
- The architecture now looks recognizably Transformer-like.

Course checkpoint:
- Continue CS224N:
  - Revisit Lecture 8 as needed.
  - Lecture 9: pretraining.

## Break B: Implement Tokenizer
Do this after the first stable decoder-style baseline.
Do not do it before `006`.

What to implement:
- Basic BPE training script.
- Frozen tokenizer artifacts.
- Deterministic encode/decode tests.

## Milestone 007: Tokenized Single-Block Decoder Baseline
Model:
- Same decoder idea as `006`.
- Move from characters to tokens.

BareTensor work:
- Tokenized data path.
- Tokenizer integration.
- Stable train/val/sample path with tokens.

Exit criteria:
- Tokenized baseline runs end to end.
- Samples and losses are interpretable.

## Milestone 008: Small Multi-Layer Decoder
Model:
- Stack multiple decoder blocks.
- Keep scale modest.

BareTensor work:
- Cleaner module composition.
- Better parameter traversal.
- Better experiment hygiene.

Exit criteria:
- Multi-layer decoder is stable.
- Experiment boilerplate is now the main friction.

## Break C: Training Runner and Config
Only do this after `008`.

What to build:
- One `train.py` entrypoint.
- One config format only.
- Logging, checkpoints, and reproducible run metadata.

## Later: Performance and CUDA
Only start this once decoder semantics are stable.

Kernel priority:
- `matmul`.
- `softmax` / `log_softmax`.
- `layer_norm`.
- Fused cross-entropy.
- Attention-specific kernels.

Rule:
- correctness parity first,
- profiler evidence second,
- performance claims third.

## Decision Summary
Use this if you are unsure what to do next.

- If `002` is not stable:
  stay on `002`.

- If `002` is stable:
  start CS229 and move to `003`.

- After `003`:
  stop and build `bt.nn`.

- After `bt.nn`:
  rebuild the same model as `004` before increasing architecture complexity.

- Do not start tokenizer work before `006`.

- Do not start training-runner/config work before `008`.

- Do not start CUDA work before a stable decoder baseline exists.
