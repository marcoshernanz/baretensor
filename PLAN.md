# Autograd Implementation Plan

## Goals

- Match PyTorch autograd semantics where practical while keeping v1 intentionally small.
- Keep architecture scalable so new ops only need local backward formulas.
- Prioritize TinyGPT-critical path and correctness-first behavior.

## Milestones

### Milestone 1: Autograd Skeleton (current)

Scope:
- Tensor metadata: `requires_grad`, `grad`, `grad_fn`, `is_leaf`.
- Backward engine for dynamic graphs (reverse topological traversal).
- Gradient accumulation on leaf tensors.
- Basic differentiable ops: `add`, `mul`, `sum` (including broadcast-aware gradients).
- Python API: `requires_grad_`, `backward`, `detach`, `zero_grad`, constructor flags.

Acceptance criteria:
- `backward()` works for scalar losses.
- `backward(gradient=...)` required for non-scalar outputs.
- Gradient accumulation works when a leaf is used in multiple branches.
- Broadcasted `add`/`mul` gradients are reduced to input shapes.
- Unsupported autograd ops fail loudly instead of silently returning wrong grads.

### Milestone 2: Views + Shape Ops

Scope:
- `view`, `reshape`, `transpose`, `permute`, `contiguous` backward support.
- Correct gradient mapping through metadata-only views.

Acceptance criteria:
- Gradients through view chains match PyTorch on small reference cases.

### Milestone 3: Linear Algebra

Scope:
- `matmul` backward with batched + broadcasted semantics.

Acceptance criteria:
- `matmul` gradients pass finite-difference gradcheck on tiny tensors.

### Milestone 4: Core Unary + Reductions

Scope:
- `sub`, `div`, `exp`, `log`, `mean`, `max` (as needed for TinyGPT path).

Acceptance criteria:
- Stable gradients and parity checks against PyTorch for supported shapes.

### Milestone 5: TinyGPT-Critical NN Ops

Scope:
- `softmax`, `log_softmax`, `cross_entropy`, `layer_norm`, `embedding` backward.

Acceptance criteria:
- End-to-end TinyGPT training step runs with decreasing loss.

## Engineering Rules

- No in-place ops in v1 autograd.
- Keep backward formulas local to ops; keep traversal in one engine.
- Add tests with each new differentiable op.
- Fail fast for unimplemented autograd paths to avoid silent correctness bugs.
