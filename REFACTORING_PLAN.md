# Refactoring Plan

## Goal

Refactor the native tensor/autograd implementation to keep the codebase scalable, maintainable, and PyTorch-aligned while preserving behavior.

## Current Pain Points

- `native/src/tensor.cpp` is too large (core, views, reductions, linalg, and factories mixed together).
- `native/src/tensor_nn.cpp` is large and mixes forward NN ops with backward node implementations.
- Some logic still relies on long anonymous-namespace sections that are hard to navigate.
- Build and ownership boundaries are not explicit enough for future contributors.

## Refactoring Principles

- No behavior changes during refactor.
- Keep each step small and testable.
- Prefer extracting shared utilities over duplicating logic.
- Keep public API unchanged.
- Preserve existing error messages unless intentionally improved.

## Target File Layout

### Core tensor files

- `native/src/tensor_core.cpp`
  - constructors, metadata (`requires_grad`, `grad`, `grad_fn`, `is_leaf`)
  - `backward`, `detach`, `zero_grad`, `accumulate_grad`
- `native/src/tensor_views.cpp`
  - `contiguous`, `view`, `reshape`, `permute`, `transpose`, `T`, `mT`
  - view/autograd node classes related to shape/layout ops
- `native/src/tensor_reductions.cpp`
  - `sum`, `mean`, `max`
  - reduction plans and reduction helpers
  - reduction/autograd node classes
- `native/src/tensor_linalg.cpp`
  - `matmul`
  - matmul canonicalization and kernel helpers
  - matmul backward node
- `native/src/tensor_factories.cpp`
  - `full`, `zeros`, `ones`

### NN files

- `native/src/tensor_nn_ops.cpp`
  - `softmax`, `log_softmax`, `layer_norm`, `cross_entropy`, `embedding` forward
- `native/src/tensor_nn_autograd.cpp`
  - NN autograd node classes and shared backward helpers

### Detail helpers

- Keep cross-cutting helpers in `native/include/bt/detail/*` (and `native/src/detail/*` if non-inline):
  - autograd recording helpers
  - tensor metadata validation
  - shape/dim formatting and broadcast helpers

## Execution Milestones

### Milestone 0: Baseline Safety (completed)

- Capture current status with:
  - `cmake --build --preset dev`
  - `make test`
  - `make lint`
  - `make typecheck`
- Record current file sizes and module responsibilities.

Acceptance:
- Baseline is green before refactor starts.

Baseline snapshot (`2026-03-01 20:49:03 CET`):
- Build: `cmake --build --preset dev` passed.
- Tests: `make test` passed (`198` tests).
- Lint: `make lint` passed.
- Typecheck: `make typecheck` passed.
- Translation unit sizes:
  - `native/src/tensor.cpp`: `1316` lines
  - `native/src/tensor_nn.cpp`: `900` lines
  - `native/src/ops.cpp`: `456` lines
  - `native/src/autograd.cpp`: `230` lines
  - `native/src/storage.cpp`: `78` lines
- Current module responsibilities:
  - `native/src/tensor.cpp`: tensor core metadata/state, views, reductions, linalg, and factories.
  - `native/src/tensor_nn.cpp`: NN forward ops and NN autograd nodes.
  - `native/src/ops.cpp`: elementwise ops and elementwise autograd nodes.

### Milestone 1: Core File Split (`tensor.cpp`) (completed)

- Move code from `tensor.cpp` into:
  - `tensor_core.cpp`
  - `tensor_views.cpp`
  - `tensor_reductions.cpp`
  - `tensor_linalg.cpp`
  - `tensor_factories.cpp`
- Keep declarations in `native/include/bt/tensor.h` unchanged.
- Update `CMakeLists.txt` source list.

Acceptance:
- Full suite green.
- `tensor.cpp` removed or reduced to minimal compatibility wrapper.

Milestone 1 snapshot (`2026-03-01 20:55:50 CET`):
- `native/src/tensor.cpp` removed.
- New split files:
  - `native/src/tensor_core.cpp`: `233` lines
  - `native/src/tensor_views.cpp`: `304` lines
  - `native/src/tensor_reductions.cpp`: `580` lines
  - `native/src/tensor_linalg.cpp`: `314` lines
  - `native/src/tensor_factories.cpp`: `44` lines
- `CMakeLists.txt` now builds split tensor sources instead of `native/src/tensor.cpp`.
- Validation:
  - `cmake --build --preset dev` passed.
  - `make test` passed (`198` tests).
  - `make lint` passed.
  - `make typecheck` passed.

### Milestone 2: NN Split (`tensor_nn.cpp`) (completed)

- Move forward APIs into `tensor_nn_ops.cpp`.
- Move node classes/backward kernels into `tensor_nn_autograd.cpp`.
- Keep symbols and call paths unchanged.

Acceptance:
- Full suite green.
- Both NN files are focused and easier to navigate.

Milestone 2 snapshot (`2026-03-01 21:05:23 CET`):
- `native/src/tensor_nn.cpp` removed.
- New split files:
  - `native/src/tensor_nn_ops.cpp`: `543` lines
  - `native/src/tensor_nn_autograd.cpp`: `418` lines
- Added shared NN autograd node-factory declarations:
  - `native/include/bt/detail/tensor_nn_autograd.h`
- `CMakeLists.txt` now builds the split NN sources instead of
  `native/src/tensor_nn.cpp`.
- Validation:
  - `cmake --build --preset dev` passed.
  - `make test` passed (`198` tests).
  - `make lint` passed.
  - `make typecheck` passed.

### Milestone 3: Utility Consolidation (completed)

- Review remaining duplication in anonymous namespaces.
- Promote repeated reusable internals into `bt/detail` headers/sources.
- Remove dead local helpers.

Acceptance:
- No duplicated core utility logic across tensor modules.

Milestone 3 snapshot (`2026-03-01 21:12:47 CET`):
- Consolidated shared dimension/permutation helpers into
  `bt/detail/dims`:
  - `native/include/bt/detail/dims.h`
  - `native/src/detail/dims.cpp`
- Added reusable helpers:
  - `make_axis_order(size_t rank)`
  - `invert_permutation(const std::vector<int64_t>& dims)`
- Removed duplicated local implementations from:
  - `native/src/tensor_views.cpp`
  - `native/src/tensor_reductions.cpp`
- Updated tensor modules to call shared `bt::detail` utilities.
- Validation:
  - `cmake --build --preset dev` passed.
  - `make test` passed (`198` tests).
  - `make lint` passed.
  - `make typecheck` passed.

### Milestone 4: Documentation and Include Hygiene

- Ensure each file has:
  - file header
  - namespace header
  - function/class headers where needed
- Trim unnecessary includes and keep each TU self-sufficient.

Acceptance:
- Consistent documentation style across all new files.
- Clean build with no missing/implicit include reliance.

### Milestone 5: Final Validation and Regression Guard

- Run:
  - `cmake --build --preset dev`
  - `make test`
  - `make lint`
  - `make typecheck`
- Optional: add a small CI check to cap max TU size (soft guard, not blocker).

Acceptance:
- All checks green.
- Refactor merged with zero functional regressions.

## Risks and Mitigations

- Risk: symbol duplication or missing definitions after split.
  - Mitigation: move in small chunks and rebuild after each chunk.
- Risk: accidental behavior changes.
  - Mitigation: keep existing tests green after every milestone.
- Risk: header/include cycles.
  - Mitigation: keep helpers in `detail` and avoid cross-including implementation headers.

## Done Criteria

- Tensor/NN implementations are split into focused translation units.
- Shared helper logic is centralized.
- Behavior is unchanged (tests and type/lint checks green).
- New structure is documented and easy to extend with future ops.
