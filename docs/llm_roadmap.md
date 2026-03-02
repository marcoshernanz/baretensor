# Learning-First LLM Roadmap (Simple Version)

## Objective
Learn by building tiny language models end-to-end in small, self-contained scripts.

This version intentionally avoids heavy infrastructure. You move fastest by:
- writing one script per milestone,
- keeping loss fixed,
- recording results manually.

## Core Decisions (for now)
- Dataset: Tiny Shakespeare (`input.txt` downloaded automatically by scripts).
- Split: deterministic 90% train / 10% val by character position.
- Tokenizer: character-level (no BPE yet).
- Loss: next-token cross-entropy.
- Process: one file per milestone in `experiments/`.

## Why this is better right now
- Zero config complexity.
- Immediate feedback loop.
- Easier to understand every line.
- You can refactor later, after you already trained a few models.

## Milestones (Simple Ladder)
1. Bigram LM (table lookup).
2. 1-hidden-layer MLP LM.
3. Add train/val loss plotting.
4. Add checkpoint save/load.
5. Add sampling options (temperature/top-k).
6. Increase context length.
7. Add token embeddings + positional embeddings.
8. First single-head attention block.
9. Multi-head attention + residual + LayerNorm.
10. Mini GPT stack.
11. Speed pass on CPU.
12. Move core kernels to CUDA/BareTensor backend work.

## Ground Rules
- Change one thing at a time.
- Keep the loss and dataset fixed while comparing milestones.
- Log each run in `docs/learning_log.md`.
- Do not optimize architecture and infrastructure at the same time.

## What "good progress" looks like
- You can explain every tensor shape in your current script.
- Validation loss decreases from one milestone to the next.
- You can generate samples at every milestone.

## Setup to Start Today
1. Build BareTensor extension once:
   - `make build`
2. Run bigram:
   - `PYTHONPATH=src uv run python experiments/001_bigram_bt.py`
3. Run MLP:
   - `PYTHONPATH=src uv run python experiments/002_mlp_1hidden_bt.py`
4. Record numbers in `docs/learning_log.md`.

## Later (when needed)
After 4-6 milestones, introduce:
- a shared training helper,
- a proper config system,
- fixed manifests/tokenizer artifacts.

Not before.
