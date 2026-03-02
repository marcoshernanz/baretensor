# LLM Roadmap (BareTensor)

## Goal
Build an end-to-end GPT-style language model stack in this repository, starting with a deliberately weak baseline and improving it over time while keeping the benchmark contract fixed.

This roadmap is optimized for:
- Learning first (understand every layer of the stack).
- TinyGPT scope first (ship a complete small model before scaling).
- PyTorch-like semantics where practical.
- Fair comparisons between milestones.

## Fixed Benchmark Contract (Do Not Change)

### 1. Task and model family
- Task: next-token prediction.
- Model family: decoder-only causal Transformer (GPT-style).
- Output target: class index of the next token.

### 2. Dataset
- Primary dataset: `roneneldan/TinyStories`.
- Freeze one deterministic split manifest (train/val/test) and commit it.
- Do not change corpus once leaderboard tracking starts.

### 3. Tokenizer
- Pick one tokenizer spec and freeze it (recommended: BPE vocab size `8k`).
- Keep tokenizer files/version fixed for all milestones used in leaderboard comparisons.
- You can reimplement the tokenizer internals later, but resulting token IDs must match the frozen spec.

### 4. Loss
- Loss: token-level cross-entropy on shifted sequence targets.
- Formula:
  - `logits = model(x[:, :-1])`
  - `targets = x[:, 1:]`
  - `loss = CrossEntropy(logits, targets, reduction="mean")`
- Use `ignore_index` only for explicit padding tokens.

### 5. Metrics
- Primary: validation cross-entropy loss.
- Secondary: perplexity `exp(val_loss)`.
- Systems metrics: tokens/sec, step time, peak memory.

### 6. Fairness protocol
- Compare model-quality ideas at fixed token budgets (for example: `20M`, `100M`, `300M`, `1B`).
- Compare systems ideas at fixed model/data and report time-to-target-loss.
- For claims, run 3 seeds and report mean/std.
- Keep benchmark device class fixed per leaderboard track:
  - local GPU track: RTX 2060 SUPER
  - cloud track: one pinned GPU SKU per experiment family

## Milestone Ladder

Each milestone should end with: code, tests, benchmark logs, and a short note explaining what changed and why.

### Phase 0: Reproducible training harness
1. Add a single `train.py` entrypoint with config object.
2. Add deterministic seeding for Python/NumPy/BareTensor.
3. Add run directory structure (`runs/<timestamp>-<name>`).
4. Log config, git SHA, seed, dataset manifest hash.
5. Add periodic validation loop.
6. Add checkpoint save/load.
7. Add simple text generation script for qualitative inspection.

Acceptance criteria:
- Same config + same seed reproduces very similar curves.
- Resume-from-checkpoint continues with no metric jumps.

### Phase 1: Pre-Transformer baselines (learning track)
8. Implement unigram or n-gram baseline for sanity.
9. Implement tiny MLP language model.
10. Add token embedding layer.
11. Add positional embeddings.
12. Add SGD optimizer.
13. Add Adam optimizer.
14. Add AdamW optimizer and verify parity against reference equations.
15. Add gradient clipping.
16. Add LR warmup + cosine decay scheduler.
17. Add dropout.
18. Add LayerNorm-based MLP block.
19. Implement RNN LM baseline.
20. Implement GRU/LSTM baseline.

Acceptance criteria:
- Each step has an ablation entry showing impact on validation loss.

### Phase 2: First GPT
21. Implement causal self-attention (single head).
22. Add multi-head attention.
23. Add residual connections.
24. Switch to Pre-LN Transformer block.
25. Stack multiple Transformer blocks.
26. Add tied input embedding/output projection.
27. Add standard GPT parameter initialization.
28. Add configurable context length.
29. Add attention masking tests.
30. Add full GPT model tests (shape, forward, backward).

Acceptance criteria:
- Small GPT beats MLP/RNN baselines on same token budget.

### Phase 3: Training stability and quality
31. Add gradient norm logging and alert thresholds.
32. Add loss-scaling path for mixed precision.
33. Add gradient accumulation for memory-limited training.
34. Add activation checkpointing.
35. Add label smoothing as optional branch (off by default for benchmark).
36. Add improved init/scaling for deep residual stacks.
37. Compare GELU vs SwiGLU FFN variants.
38. Add RoPE experiment branch (separate from locked leaderboard config).
39. Add RMSNorm experiment branch.

Acceptance criteria:
- Stable long runs with no NaN/Inf at target depth/sequence.

### Phase 4: CPU performance engineering
40. Add profiler hooks for operation-level timing.
41. Optimize dataloader throughput (prefetch, pinned memory where relevant).
42. Reduce Python overhead in training step.
43. Optimize CPU matmul path (cache-friendly tiling).
44. Fuse numerically stable softmax + cross-entropy path on CPU.
45. Add regression benchmarks in CI for critical kernels.

Acceptance criteria:
- Clear tokens/sec improvement on fixed model + batch.

### Phase 5: CUDA backend (owning the stack)
46. Introduce CUDA tensor storage/device plumbing.
47. Add CUDA elementwise kernels.
48. Add CUDA reduction kernels.
49. Add CUDA matmul naive kernel.
50. Add tiled/shared-memory CUDA matmul.
51. Integrate cuBLAS as correctness/perf reference path.
52. Add CUDA LayerNorm kernel.
53. Add CUDA softmax/log_softmax kernels.
54. Add fused CUDA cross-entropy kernel.
55. Add CUDA attention kernel.
56. Add FlashAttention-style implementation (advanced).
57. Add fused optimizer kernel (optional advanced).
58. Add CUDA profiling scripts (occupancy, memory bandwidth).

Acceptance criteria:
- CUDA path is numerically validated against CPU reference.
- Throughput improvement is measured and logged per kernel milestone.

### Phase 6: Scaling and cloud
59. Add multi-GPU data parallel training.
60. Add optimizer/state sharding if needed.
61. Add distributed checkpointing.
62. Add cluster reproducibility script (single command bring-up).
63. Run fixed-budget scaling experiments.
64. Publish internal leaderboard table for all milestones.

Acceptance criteria:
- Same experiment can run local or cloud with minimal config change.

### Phase 7: Paper-driven improvements (one at a time)
65. Select one paper feature.
66. Implement minimally.
67. Add unit tests and gradient checks.
68. Run controlled ablation at fixed token budget.
69. Keep feature only if it improves target metric or system efficiency.

Acceptance criteria:
- No feature merges without measured evidence.

## Compute Plan (Local First, Then Cloud)

## Hardware assumptions
- Primary coding machine: MacBook Air M4, 16GB RAM.
- Local GPU: RTX 2060 SUPER (8GB VRAM).
- Expect small contexts, small batch sizes, and heavy use of gradient accumulation initially.

## Device roles (recommended)
- MacBook Air M4:
  - Default machine for coding, unit tests, docs, data preprocessing, tokenizer tooling, and short CPU debug runs.
  - Fast feedback loop for non-CUDA milestones (Phases 0-4, except GPU perf validation).
- RTX 2060 SUPER desktop:
  - Canonical local benchmark/training device for all GPU-relevant milestones.
  - Use this device for any run you want to compare historically in the local leaderboard.
- Cloud GPU:
  - Use for runs blocked by local VRAM/runtime and for scaling milestones.
  - Treat cloud as a separate leaderboard track unless hardware is exactly matched.

## Suggested training allocation
1. MacBook + local desktop bootstrap:
   - MacBook for daily development and correctness iteration.
   - RTX 2060 SUPER for benchmarked training.
   - Total RTX 2060 SUPER budget: `~150-300 GPU hours`.
   - Phases 0-3 and early Phase 4.
   - Many short runs to validate correctness and training behavior.
2. Cloud burst budget (`$100-$300`):
   - Phase 5+ where CUDA and scale experiments benefit from stronger GPUs.
   - Reserve cloud spend for experiments that are blocked by local VRAM or runtime.

## Credit-aware strategy
- Use GitHub Student pack credits first (for example DigitalOcean/Azure) for burst runs.
- Use local machine for daily iteration and debugging.
- Use cloud only for long fixed-budget runs and performance validation.

## Fair-run templates

### Template A: architecture milestone
- Budget: fixed tokens (example `100M`).
- Runs: 3 seeds.
- Report: val loss mean/std, perplexity, tokens/sec.

### Template B: systems milestone
- Keep architecture + data fixed.
- Target: reach baseline val loss threshold.
- Report: wall-clock time and cost-to-target.

## Baseline config suggestions for 2060 SUPER

Start very small and scale only after stable convergence:
- `n_layers=4`
- `n_heads=4`
- `d_model=256`
- `d_ff=1024`
- `context_len=128` (then `256`)
- `vocab_size=8k`
- `micro_batch_size` set by VRAM
- gradient accumulation to reach effective batch target

Next size after stability:
- `n_layers=8`
- `n_heads=8`
- `d_model=512`
- `context_len=256`

## Project hygiene checklist (required per milestone)
- Unit tests for new ops and model logic.
- Numerical checks against trusted reference behavior.
- Reproducible run config committed.
- Benchmark results logged in a consistent format.
- Short design note: what changed, why, expected impact, actual impact.

## Definition of Done
A milestone is complete only if all are true:
- Functionally correct (tests pass).
- Numerically stable (no NaN/Inf in target run length).
- Reproducible (seeded rerun matches expected behavior).
- Measured (quality and/or systems impact recorded).
- Documented (brief change note with rationale).

## Immediate next actions
1. Create `train.py` + `configs/baseline_tinystories.yaml`.
2. Add dataset manifest generation and split freezing scripts.
3. Add tokenizer freeze script and tokenizer artifact versioning.
4. Add run profiles (for example `dev-cpu`, `local-2060`, `cloud`) so commands are reproducible across devices.
5. Implement Phase 0 logging/checkpointing before any architecture changes.
6. Start milestone tracking table in this file or a separate `docs/llm_leaderboard.md`.
