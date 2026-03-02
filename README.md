# BareTensor

BareTensor is a learning-by-building project to master the full modern AI stack end-to-end.

## Mission
Build from first principles:
- tensor storage and views,
- autograd,
- neural-network primitives,
- training loops,
- tokenization,
- CUDA kernels,
- modern Transformer architectures.

The target is deep stack ownership, not just model usage.

## Why this exists
The goal is to become strong enough to build and debug ML systems across all layers:
- C++ core internals,
- CUDA performance paths,
- Python API ergonomics,
- model architecture and training behavior.

This repo is designed to make that progression explicit and measurable.

## Dependency philosophy
Target: near-zero external dependencies.
- Core principle: implement the ML stack inside BareTensor.
- Binding exception: `nanobind` for C++ <-> Python bindings.
- Everything else should be minimized or temporary.

## Scope
In scope:
- BareTensor-native model training and inference.
- BareTensor-native NN APIs (`bt.nn` style layers/modules).
- BareTensor-native kernels and performance work.

Out of scope:
- Building a wrapper around PyTorch as the final architecture.
- Hiding core learning behind heavy framework abstractions.

## Learning style
- Start simple, but move each milestone into BareTensor-native implementations.
- Keep benchmark assumptions stable while iterating.
- Prefer clarity and correctness before optimization.
- Add complexity only when the previous layer is understood.

## Roadmap
See [docs/llm_roadmap.md](/Users/marcoshernanz/dev/baretensor/docs/llm_roadmap.md) for the staged plan and exact transition points:
- when to add tokenizer,
- when to add `bt.nn` layer APIs,
- when to adopt robust `train.py` + config,
- when to begin custom CUDA kernel work.
