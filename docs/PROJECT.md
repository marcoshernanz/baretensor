# BareTensor: Zero-Dependency Tensor Engine + TinyGPT

Updated: 2026-02-15

BareTensor is a from-scratch tensor + autograd engine with custom CUDA kernels, plus a TinyGPT training stack built on top.

This file is the complete project spec and context.

Setup instructions live in `SETUP.md`.

## Context (constraints + goals)

- Owner: Marcos Hernanz (UAM Computer Engineering; incoming Vercel internship in San Francisco starting 2026-06-15).
- North-star: ship a flagship repo that proves ML systems fundamentals (implement + debug training end-to-end).
- Time budget: ~40h/week for ML outside obligations (now -> mid-June 2026).
- Compute: RTX 2060 Super desktop for CUDA training; MacBook Air for CPU-only dev.
- Target demos: MNIST MLP (sanity) + Tiny Shakespeare TinyGPT (perplexity + samples).
- Naming: repo `baretensor`, Python module `bt`, C++ namespace `bt`.

## Project order (high level)

This project is done in this order:

1) Build the tensor library (Tensor + ops + autograd + CUDA)
2) Courses: skim CS229, then watch CS231n, then watch CS224n (time-boxed)
3) Implement TinyGPT using the tensor library

## Constraints (dependencies)

- No PyTorch, no NumPy, no training frameworks.
- You own the Tensor, ops, and gradient engine.
- You write custom CUDA kernels.
- Only third-party library code: `nanobind` (vendored) for C++ <-> Python bindings.
- Everything else: Python stdlib + C++ stdlib + CUDA toolkit.
- Dev tooling is allowed (uv/ruff/pyright), but runtime deps stay zero.

## Architecture (how the pieces fit)

Three layers, one API:

1) Python layer (ergonomics + experiments)
   - Defines models (GPT blocks), training loops, datasets, sampling.
   - Calls into the native backend for all tensor math.
   - Zero Python deps: use `argparse`, `csv`, `json`, `urllib`, `gzip`, `unittest`.

2) C++ core (Tensor + autograd + dispatch)
   - Own Tensor object, shape/stride/view semantics, dtype, device.
   - Autograd graph + backward engine.
   - CPU kernels for correctness + gradcheck.
   - Dispatch to CUDA kernels when `device=cuda`.

3) CUDA layer (kernels)
   - Custom kernels for elementwise, reductions, matmul, layernorm, softmax, embedding.
   - Optional: call cuBLAS for GEMM behind a build flag as an escape hatch (CUDA toolkit, not an extra dependency).

Key principle: correctness first on CPU, then port ops to CUDA one-by-one. Every CUDA op must have a CPU reference implementation and tests.

## Repo layout (suggested)

This is the structure for the *code repo* that implements this spec.

```
baretensor/
  pyproject.toml
  CMakeLists.txt
  third_party/
    nanobind/
  src/
    bt/                      # Python package (BareTensor)
      __init__.py
      tensor.py              # thin Python wrapper around native Tensor
      nn/
        __init__.py
        functional.py        # cross_entropy, gelu, layernorm wrappers
        modules.py           # Linear, Embedding, LayerNorm, Dropout, GPT
      optim.py               # AdamW, SGD (Python, uses Tensor ops)
      data/
        mnist.py
        tinyshakespeare.py
      train/
        train_mnist.py
        train_gpt.py
        sample.py
  native/
    include/
      bt/tensor.h
      bt/autograd.h
      bt/ops.h
      bt/device.h
    cpu/
      tensor_cpu.cc
      ops_cpu.cc
    cuda/
      tensor_cuda.cu
      ops_cuda.cu
      kernels/
        elementwise.cu
        reduce.cu
        matmul.cu
        layernorm.cu
        softmax.cu
        embedding.cu
    python/
      bindings.cc            # nanobind module
  tests/
    test_gradcheck.py
    test_ops_cpu.py
    test_ops_cuda.py
    test_gpt_shapes.py
  notes/
    autograd.md
    kernels.md
    gpt.md
```

## Tech stack (versions) + allowed dependencies

Languages:

- Python: 3.13, managed with `uv`
- C++: C++20 (compiler: clang++ 17+ or g++ 13+)
- CUDA: CUDA C++ (CUDA toolkit 12.x on the RTX 2060 box)

Python tooling (dev-only):

- `uv` (latest stable) for Python version + venv + running tasks
- `ruff` (latest stable) for lint + format
- `pyright` (latest stable) for static type checking

Build tooling:

- CMake

Bindings (only third-party library code in the repo):

- `nanobind` (vendored in `third_party/nanobind/`)

CUDA runtime interface:

- Compile `.cu` with NVCC and call from C++
- CUDA runtime API (`cudaMalloc`, `cudaMemcpy`, streams)

Optional (still within CUDA toolkit):

- cuBLAS (behind `BT_USE_CUBLAS=ON`) to avoid getting stuck on GEMM performance

## Tensor library: what it must support

### Data model

Tensor consists of:

- `Storage`: owns the raw buffer (CPU malloc / cudaMalloc), byte size, device, dtype, refcount.
- `TensorView`: shape + strides + offset (views into Storage).
- `AutogradMeta`: `requires_grad`, `grad` tensor, and `grad_fn` node pointer.

Start with contiguous tensors, but implement view metadata early:

- `reshape` is a view if contiguous.
- `transpose/permute` returns a strided view.
- `contiguous()` materializes.

This lets you keep user-facing API clean while allowing kernels to initially require contiguous inputs.

### Dtypes + devices

Baseline:

- `f32` for all trainable tensors.
- `i32` for indices (embedding, targets).
- Devices: `cpu`, `cuda`.

Stretch:

- `f16` for forward + matmul; keep grads in `f32` (mixed precision).

### Autograd engine

Dynamic graph, tape-style:

- Each op produces an output Tensor with a `grad_fn` node.
- Node stores:
  - pointers to parent tensors
  - any saved tensors/metadata needed for backward
  - a `backward(out_grad) -> in_grads` implementation

Backward algorithm:

- Build a topological order from the loss scalar.
- Propagate gradients; accumulate when a tensor is used multiple times.
- Free saved tensors ASAP (reference counting) to control VRAM.

Rules:

- No in-place ops in v1 (they complicate autograd and debugging).
- Explicit `detach()`.
- Broadcasting must reduce gradients along broadcasted dimensions.

### Minimum op set (to unlock GPT)

Core math:

- Elementwise: `add`, `sub`, `mul`, `div`.
- Unary: `exp`, `log`, `tanh`, `relu`, `gelu`.
- Reductions: `sum`, `mean`, `max` (needed for stable softmax).
- Linear algebra: `matmul`.

Shape/view:

- `reshape`, `view`, `transpose`, `permute`, `contiguous`.
- Broadcasting semantics (NumPy-like).

Indexing:

- `gather` (embedding lookup).

Normalization + losses:

- `layernorm`.
- `softmax` (stable).
- `cross_entropy(logits, targets)`.

Utilities:

- `zeros`, `ones`, `randn` (deterministic seed), `arange`.
- `where` (optional but useful).

## Which ops go to CUDA (and why)

Implement CPU first for each op, then CUDA.

Priority CUDA kernels (TinyGPT hot path):

1) Elementwise ops (memory bound, easy wins)
   - add/mul/div, relu/gelu, bias add.

2) Reductions
   - sum/mean, row-wise max + sumexp components for softmax.

3) Matmul / batched matmul
   - Start: correct naive GEMM.
   - Then: tiled shared-memory GEMM.
   - Escape hatch: cuBLAS (optional) so training is not blocked.

4) LayerNorm (fused)
   - Forward: mean/var + normalize + affine.
   - Backward: fused gradient to avoid multiple passes.

5) Softmax + CrossEntropy (fused)
   - Stable softmax and log-sum-exp.
   - Fuse loss + gradient for speed and numerical stability.

6) Embedding gather (+ grad scatter-add)
   - Forward gather.
   - Backward scatter-add into embedding matrix grad.

Later CUDA work (only after end-to-end GPT trains):

- Fused attention kernels.
- Optimizer step fused kernels (AdamW update).
- Mixed precision.

## CUDA vs C++ vs Python: clear ownership

CUDA:

- Kernels only: elementwise, reductions, matmul, layernorm, softmax/xent, embedding.
- Device memory utilities (memset, copies) where it helps.

C++:

- Tensor + Storage + view/stride logic.
- Autograd nodes + backward engine.
- CPU reference kernels.
- CUDA dispatch glue + stream management.
- Deterministic RNG (CPU and CUDA).
- Serialization: `state_dict` save/load (simple binary + JSON metadata).

Python:

- Public API surface (clean Pythonic wrapper).
- Model code (Transformer blocks) calling Tensor ops.
- Data loading (MNIST + Tiny Shakespeare downloader/parser).
- Training loop + evaluation + sampling.
- Logging (CSV) + simple plotting hook (optional).

## Final Python API (target)

Keep it small and familiar (PyTorch-like ergonomics, but minimal).

Core Tensor API:

```python
import bt

x = bt.tensor([[1, 2], [3, 4]], dtype="f32", device="cpu", requires_grad=True)
y = (x @ x).sum()
y.backward()
print(x.grad)

x = x.to("cuda")
```

Functional ops:

```python
import bt
from bt import nn

logits = bt.randn((32, 128, 65), device="cuda")
targets = bt.randint(0, 65, (32, 128), dtype="i32", device="cuda")
loss = nn.cross_entropy(logits, targets)
loss.backward()
```

Module-style API (Python-defined modules, native tensors):

```python
from bt import nn, optim

model = nn.GPT(
    vocab_size=65,
    context=128,
    n_layers=4,
    n_heads=4,
    d_model=256,
    d_ff=1024,
    dropout=0.0,
    device="cuda",
)

opt = optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

for step in range(steps):
    x, y = batcher.next()
    logits = model(x)
    loss = nn.cross_entropy(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
```

Non-negotiables for API design:

- Errors must be readable (shapes, devices, dtypes).
- Every op has deterministic CPU behavior for tests.
- The default path is correctness; performance comes from switching `device="cuda"`.

## Technical decisions + DO / DON'T DO

DO:

- Implement CPU correctness first, then CUDA.
- Add gradcheck for every differentiable op.
- Keep Tensor semantics explicit: view vs copy, contiguous vs strided.
- Prefer fused CUDA kernels only where it reduces memory traffic (layernorm, softmax/xent).
- Keep the public API tiny; add features only if GPT needs them.

DON'T DO:

- Don't try to match PyTorch feature-for-feature.
- Don't add in-place ops early.
- Don't chase matmul peak performance before TinyGPT trains end-to-end.
- Don't implement convolution, distributed training, or dataloader frameworks.
- Don't introduce extra deps for logging, configs, tokenizers, or plotting.

## Courses (time-boxed)

- CS229: 1-2 week skim (linear/logistic regression, regularization, bias/variance, optimization basics).
- CS231n: primary course while finishing the tensor engine (backprop, init, normalization, optimizers).
- CS224n: take after you can train a minimal char-level GPT (then the lectures click harder).

## Milestones (each milestone ends with running code)

Two blocks: (A) tensor library, (B) GPT.

### A) Tensor library milestones

1) Build + import works
   - Python can `import bt`.
   - Create a CPU tensor from Python lists.

2) Autograd skeleton
   - Ops: add/mul/sum.
   - `backward()` works for scalar loss.
   - `tests/test_gradcheck.py` passes for tiny cases.

3) Shapes, views, broadcasting
   - `reshape`, `transpose`, broadcasting add/mul.
   - Correct backward for broadcasted dims.

4) CPU matmul + MLP sanity
   - `matmul` CPU + backward.
   - Train a tiny MLP on a synthetic dataset (loss decreases).

5) CUDA device + elementwise
   - `tensor.to("cuda")`.
   - CUDA kernels for add/mul + backward.
   - CPU vs CUDA parity tests.

6) CUDA matmul (plus optional cuBLAS switch)
   - Naive + tiled GEMM kernel.
   - Benchmark script prints tokens/sec for a small matmul workload.
   - Optional: cuBLAS path behind a flag.

7) LayerNorm + Softmax + CrossEntropy
   - CPU first, then CUDA.
   - Stable numerics (no NaNs on random inputs).
   - Gradcheck on small tensors.

8) MNIST end-to-end
   - MNIST downloader/parser (stdlib).
   - MLP trains on CPU, then on CUDA.
   - Save/load weights.

### B) GPT milestones

1) Data + baseline model
   - Tiny Shakespeare downloader.
   - Char tokenizer + batching.
   - Bigram model trains + samples (end-to-end pipeline works).

2) Attention building block
   - Implement masked self-attention with your matmul + softmax.
   - Unit test for causal masking correctness.

3) Transformer block
   - Pre-norm block: LayerNorm -> Attn -> residual -> LayerNorm -> MLP -> residual.
   - Forward/backward runs on small random tensors.

4) TinyGPT trains
   - Train config (small): context=128, 4 layers, 4 heads, d_model=256.
   - Loss decreases; samples improve.

5) Speed + stability pass
   - AdamW + gradient clipping + warmup.
   - Add fused CUDA kernels where profiling shows wins (layernorm, softmax/xent, bias+gelu).

6) Portfolio polish
   - One-command reproduce script.
   - Short notes: autograd design + attention numerics + kernel learnings.
   - README includes results (loss/perplexity curves, sample text).
