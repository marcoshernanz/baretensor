# BareTensor LLM Roadmap

## Why This Roadmap Is Structured This Way
The goal of this project is to maximize learning, not to reach a modern architecture in the fewest calendar days.

That means the roadmap is intentionally granular.
We are not taking the shortest path to a GPT-like model.
We are taking the path that gives the most understanding per milestone.

The approach is:
- implement the milestone in PyTorch first,
- make the forward pass, loss, sampling path, and training dynamics understandable,
- be able to explain the tensor shapes and gradient flow clearly,
- then port the exact same milestone to BareTensor,
- only after both versions are understood do we move to the next milestone.

This is deliberately slower than jumping straight to attention or Transformers.
It is also better for learning because it isolates ideas.
You want to learn one new concept at a time:
- context,
- nonlinearity,
- hidden state,
- recurrence,
- gating,
- attention,
- residual structure,
- normalization,
- tokenization,
- stacking.

We also prefer direct conceptual stepping stones over historical completeness.
However, if an older architecture teaches something important for modern models, it belongs in the roadmap.
That is why RNNs are included here:
- not because they are the end goal,
- but because they teach sequence state, unrolling, and gradient flow in a very direct way.

CNNs are not on the main path.
They are useful, but for the TinyGPT-oriented goal they are lower learning-value than context-window MLPs, RNNs, and attention.

## Current Position
As of 2026-03-08, you are at the end of `002` and preparing for `003`.

In practical terms:
- `001` bigram is done.
- `002` 1-hidden-layer MLP is done or nearly done.
- This is the point where CS229 should start.
- The next model milestone should be a context-window model, not attention yet.

## Global Rules
- Every milestone is built in PyTorch first, then ported to BareTensor.
- Do not move on until you can explain the forward pass and gradient path.
- Keep each milestone runnable as a standalone experiment script.
- Do not add library abstractions too early.
- Add `bt.nn` only after raw experiment code becomes clearly repetitive.
- Add tokenizer work only after character-level modeling has earned the need.
- Do not start CUDA work before the model path is semantically stable.

## Course Timing
### CS229
Start CS229 now, during the late `002` / early `003` period.

Recommended use:
- early: Lectures 1, 2, 3,
- later, when you want stronger neural-net intuition: Lectures 10, 11, 12.

CS229 should support your understanding of optimization, supervised learning, neural nets, and debugging.
It should not block implementation progress.

### CS224N
Start CS224N when you begin attention-oriented milestones.

Recommended use:
- before or during early sequence-model work: Lectures 1, 2, 3,
- when attention becomes real: Lectures 7 and 8,
- when language-model scaling/pretraining questions become real: Lecture 9.

CS224N should support NLP and sequence-model understanding.
It should not replace the implementation path.

## Milestone 001: Bigram Counts LM
### Model
- Character-level bigram language model from counts.

### PyTorch First
- Build the count table.
- Normalize into probabilities.
- Compute cross-entropy manually from next-token probabilities.
- Sample text.

### BareTensor Port
- Rebuild the same pipeline in BareTensor.

### BareTensor Features Needed
- Tensor construction.
- Basic indexing.
- `sum`.
- `log`.
- `numpy()` for sampling/readback.

### Understanding Needed Before Implementing
- What a bigram model is.
- How probability normalization works.
- What cross-entropy means for next-token prediction.
- How a sampled next-token distribution maps to generated text.

### Learning Outcomes
You should come out of this milestone able to:
- explain why bigram is a one-step context model,
- compute cross-entropy by hand for a token pair,
- explain every tensor shape in the script,
- explain why the generated samples are weak but still meaningful.

### Exit Criteria
- The script runs end to end.
- Cross-entropy is stable.
- You can explain the full data path and sampling path.

## Milestone 002: Single-Token 1-Hidden-Layer MLP LM
### Model
- Character LM with embedding lookup, one hidden layer, and `tanh`.
- Still only one-token context.

### PyTorch First
- Build embedding table.
- Use one token as input.
- Project to hidden layer.
- Apply `tanh`.
- Project to logits.
- Train with cross-entropy.

### BareTensor Port
- Reproduce the same model and training loop in BareTensor.

### BareTensor Features Needed
- `matmul`.
- Elementwise autograd.
- `tanh`.
- `embedding`.
- `cross_entropy`.
- `no_grad`.
- In-place optimizer step under `no_grad`.

### Understanding Needed Before Implementing
- What embedding lookup is doing.
- Why hidden layers increase capacity.
- Why `tanh` changes the model class.
- What logits are and why softmax is implicit inside cross-entropy.
- How gradient descent updates parameters.

### Learning Outcomes
You should come out of this milestone able to:
- explain the role of embeddings vs hidden weights vs output weights,
- explain the gradient path through `tanh`,
- explain why this model is more expressive than bigram,
- explain where it still fails because context is too small.

### Exit Criteria
- Train/val/sample path runs stably.
- You can explain both forward and backward cleanly.
- The script feels like a genuine BareTensor model, not a placeholder.

## Milestone 003: Context-Window Linear LM
### Model
- Use the previous `k` characters as context.
- No hidden layer yet.
- This isolates the effect of more context.

### PyTorch First
- Build a sliding context window dataset.
- Concatenate or flatten context embeddings.
- Predict next token with a single linear projection.

### BareTensor Port
- Reproduce the same context-window pipeline in BareTensor.

### BareTensor Features Needed
- Better handling of reshape/view logic.
- Reliable indexing for context extraction.
- Comfortable batch handling over context windows.

### Understanding Needed Before Implementing
- Why more context is a separate idea from more nonlinearity.
- How a sliding window dataset is formed.
- How flattening context differs from maintaining sequence structure.

### Learning Outcomes
You should come out of this milestone able to:
- separate the idea of context size from model depth,
- explain exactly how context windows are built,
- see what longer context buys you before recurrence or attention.

### Exit Criteria
- The model runs end to end.
- You can compare it cleanly against `002`.
- You know whether the gain came from context, not hidden depth.

## Milestone 004: Context-Window MLP
### Model
- Same context window as `003`.
- Add a hidden layer and nonlinearity.

### PyTorch First
- Reuse the dataset path from `003`.
- Add one hidden layer over the flattened context representation.

### BareTensor Port
- Rebuild the same architecture in BareTensor.

### BareTensor Features Needed
- Nothing radically new if `002` and `003` are solid.
- The emphasis is on using existing pieces coherently.

### Understanding Needed Before Implementing
- Why this milestone combines two independent ideas:
  - more context,
  - more nonlinear capacity.
- How to tell which one is helping.

### Learning Outcomes
You should come out of this milestone able to:
- compare `002` vs `003` vs `004` honestly,
- explain what longer context plus an MLP buys you,
- understand why sequence order is still only implicit here.

### Exit Criteria
- The model runs stably.
- You can clearly articulate what problem remains unsolved.

## Milestone 005: Larger-Context MLP
### Model
- Keep the same basic context-window MLP idea.
- Increase context length enough to feel the limits of this family.

### PyTorch First
- Increase window size.
- Retune only minimally.

### BareTensor Port
- Rebuild the same thing in BareTensor.

### BareTensor Features Needed
- Mainly robustness and clarity in existing tensor ops.

### Understanding Needed Before Implementing
- Why scaling a limited architecture is still useful.
- How to distinguish “this family is helping” from “this family is fundamentally limited”.

### Learning Outcomes
You should come out of this milestone able to:
- feel the limits of fixed-window MLPs,
- explain why a fixed window is fundamentally different from sequence state,
- justify moving to recurrent models next.

### Exit Criteria
- You have clear evidence that fixed-window context is becoming limiting.

## Milestone 006: Vanilla RNN LM
### Model
- Simple recurrent language model with one hidden state.
- Use the sequence directly rather than a fixed context window.

### PyTorch First
- Build a vanilla recurrent cell.
- Unroll over sequence positions.
- Train with next-token loss.

### BareTensor Port
- Rebuild the same unrolled recurrence in BareTensor.

### BareTensor Features Needed
- No new special op is strictly required.
- The challenge is sequencing and keeping the computation graph understandable.

### Understanding Needed Before Implementing
- What hidden state means.
- Why recurrence gives variable-length context.
- What unrolling through time means.
- Why gradients now flow across time as well as layers.

### Learning Outcomes
You should come out of this milestone able to:
- explain hidden state as carried sequence memory,
- explain backpropagation through time at a high level,
- explain why RNNs are a meaningful conceptual bridge to attention even if they are not the end goal.

### Exit Criteria
- A simple RNN LM runs end to end.
- You can explain state update, unrolling, and gradient flow through time.

## Milestone 007: Better RNN Training Milestone
### Model
- Same vanilla RNN idea.
- Improve batching, sequencing, and training stability enough to really study it.

### PyTorch First
- Make the training setup cleaner.
- Study behavior across longer sequences.

### BareTensor Port
- Rebuild the improved training path in BareTensor.

### BareTensor Features Needed
- Mostly code discipline, not new tensor primitives.

### Understanding Needed Before Implementing
- Why a working prototype is different from a usable training setup.
- Why recurrent training gets unstable or weak on long dependencies.

### Learning Outcomes
You should come out of this milestone able to:
- recognize vanishing-gradient behavior in practice,
- explain where vanilla recurrence struggles,
- justify why gated recurrence exists.

### Exit Criteria
- The RNN behavior is clear enough that its failure modes are educational, not mysterious.

## Milestone 008: One Gated Recurrent Model (GRU or LSTM)
### Model
- Pick one: GRU or LSTM.
- Do not do both unless you really want the comparison.

### PyTorch First
- Implement the gated recurrent model.
- Compare it against the vanilla RNN.

### BareTensor Port
- Rebuild the same model in BareTensor.

### BareTensor Features Needed
- Again, likely no fundamentally new primitive.
- The real work is clearer sequence-model code.

### Understanding Needed Before Implementing
- Why gates help.
- What problem the update/forget/input gate is solving.
- Why recurrence still differs from attention even when improved.

### Learning Outcomes
You should come out of this milestone able to:
- explain why gated recurrence trains better than vanilla RNNs,
- articulate the limits of recurrence before seeing attention,
- understand attention as a solution to a problem you have now personally felt.

### Exit Criteria
- You can compare vanilla RNN vs gated RNN conceptually and empirically.

## Milestone 009: Single Causal Self-Attention Head
### Model
- First attention-capable language model.
- One causal self-attention head only.
- No `bt.nn` yet.
- No residual yet.
- No LayerNorm yet.
- No feedforward block yet.

### PyTorch First
- Build the smallest possible causal attention model.
- Use explicit Q/K/V projections.
- Use explicit causal masking.

### BareTensor Port
- Rebuild the same raw attention model in BareTensor.

### BareTensor Features Needed
- Attention score math.
- Careful `transpose` / `permute` use.
- Stable `softmax` over scores.
- Clean masking path.

### Understanding Needed Before Implementing
- Why attention replaces hidden-state recurrence with direct token-token interaction.
- How Q, K, and V differ.
- Why masking is required for autoregressive training.
- How attention weights map to information flow.

### Learning Outcomes
You should come out of this milestone able to:
- explain attention from first principles,
- explain every attention-related tensor shape,
- explain why attention is different from recurrence,
- point to the code repetition that proves `bt.nn` is now justified.

### Exit Criteria
- One-head causal attention runs end to end.
- You understand the mask, Q/K/V, and score normalization path.

### Course Checkpoint
- Start CS224N here if you have not already.
- Recommended first pass around this area:
  - Lecture 1: intro and word vectors.
  - Lecture 2: word vectors and language models.
  - Lecture 3: backpropagation and neural networks.
  - Lecture 7: attention and LLM intro.
  - Lecture 8: self-attention and Transformers.

## Break A: Implement `bt.nn`
### Why this break exists
At this point the repetition is earned, not hypothetical.
You have written enough raw model code to know what should become reusable.

### What to implement first
- `Module`.
- `Linear`.
- `Embedding`.
- `LayerNorm`.
- Minimal parameter traversal.

### What not to build yet
- No tokenizer.
- No full training runner.
- No heavy config system.
- No premature optimizer hierarchy unless it is clearly needed.

### Learning Outcomes
You should come out of this break able to:
- explain why each abstraction exists,
- distinguish useful abstraction from abstraction theater,
- make later model code clearer rather than more magical.

## Milestone 010: Rebuild 009 Using `bt.nn`
### Model
- Re-implement the same one-head attention model from `009`.
- Same architecture goal.
- Cleaner code through `bt.nn`.

### PyTorch First
- You likely do not need a new PyTorch milestone here.
- The real comparison is raw BareTensor model code vs modular BareTensor model code.

### BareTensor Work
- Rebuild the attention prototype through reusable modules.

### Understanding Needed Before Implementing
- Why architecture and library abstractions should be separated.
- What got simpler and what should remain explicit.

### Learning Outcomes
You should come out of this milestone able to:
- prove that `bt.nn` helps readability and reuse,
- separate model ideas from plumbing,
- know what should stay low-level vs become library API.

### Exit Criteria
- `010` is meaningfully cleaner than `009`.
- The model is still understandable, not hidden behind abstraction.

## Milestone 011: Attention + Residual
### Model
- Keep one attention head.
- Add residual connection.
- Still no LayerNorm or feedforward yet.

### PyTorch First
- Build the smallest residualized attention path.

### BareTensor Port
- Rebuild the same thing in BareTensor.

### Understanding Needed Before Implementing
- Why residuals matter.
- How residual connections change optimization and signal flow.

### Learning Outcomes
You should come out of this milestone able to:
- explain residuals mechanically,
- explain why they help deeper architectures,
- see the attention block becoming more Transformer-like one piece at a time.

## Milestone 012: Attention + Residual + LayerNorm
### Model
- Add LayerNorm around the attention path.
- Still no full feedforward block yet.

### PyTorch First
- Decide and understand the exact normalization placement.

### BareTensor Port
- Rebuild the same arrangement in BareTensor.

### Understanding Needed Before Implementing
- Why normalization matters.
- Why placement matters.
- What instability or scale issues normalization addresses.

### Learning Outcomes
You should come out of this milestone able to:
- explain exactly what LayerNorm is normalizing,
- justify the placement you chose,
- understand normalization as a training tool rather than a ritual.

## Milestone 013: Add Feedforward Block
### Model
- Add the feedforward sublayer.
- Now you have almost all pieces of a decoder block.

### PyTorch First
- Build attention path plus feedforward path clearly.

### BareTensor Port
- Rebuild in BareTensor with the same structure.

### Understanding Needed Before Implementing
- Why the feedforward sublayer exists.
- Why a decoder block is not just attention.

### Learning Outcomes
You should come out of this milestone able to:
- explain the role of the feedforward block,
- understand the decoder block as a composition of simpler parts.

## Milestone 014: First Single-Block Decoder-Only Transformer
### Model
- Full single decoder block.
- Still character-level.

### PyTorch First
- Build the smallest coherent decoder-only Transformer block.

### BareTensor Port
- Rebuild the same single-block model in BareTensor.

### Understanding Needed Before Implementing
- How all previous pieces fit together into one architecture.
- What makes this recognizably Transformer-like.

### Learning Outcomes
You should come out of this milestone able to:
- explain the whole decoder block from memory,
- compare it honestly against MLPs and RNNs,
- know which architectural ideas mattered most.

### Course Checkpoint
- Continue CS224N here with:
  - Lecture 8: self-attention and Transformers.
  - Lecture 9: pretraining.

## Break B: Tokenizer
### Why this break exists
Only now has character-level modeling done enough work to justify tokenization.

### What to implement
- Basic BPE training script.
- Frozen tokenizer artifacts.
- Deterministic encode/decode tests.

### Learning Outcomes
You should come out of this break able to:
- explain what tokenization changes in the modeling problem,
- reason about vocabulary granularity and sequence length tradeoffs.

## Milestone 015: Tokenized Single-Block Decoder
### Model
- Same decoder idea as `014`.
- Move from characters to tokens.

### PyTorch First
- Rebuild the model on tokenized data first.

### BareTensor Port
- Rebuild the tokenized version in BareTensor.

### Understanding Needed Before Implementing
- How tokenization changes sequence length, vocabulary size, and modeling difficulty.

### Learning Outcomes
You should come out of this milestone able to:
- explain why tokenization matters,
- interpret tokenized samples and losses coherently.

## Milestone 016: Small Multi-Layer Decoder
### Model
- Stack a few decoder blocks.
- Keep scale modest.

### PyTorch First
- Build the stacked version first.

### BareTensor Port
- Rebuild it in BareTensor.

### Understanding Needed Before Implementing
- Why stacking changes optimization and representation depth.
- Why this is a different milestone from just making one block bigger.

### Learning Outcomes
You should come out of this milestone able to:
- explain depth vs width tradeoffs more concretely,
- recognize when experiment boilerplate is now the real bottleneck.

## Break C: Training Runner and Config
Do this only after `016`.

What to build:
- One `train.py` entrypoint.
- One config format only.
- Logging, checkpoints, and reproducible run metadata.

Learning outcome:
- learn to build training infrastructure only after you have truly earned it.

## Later: Performance and CUDA
Only start this after the decoder path is semantically stable.

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
- If `002` is not stable, stay on `002`.
- If `002` is stable, start CS229 and move to `003`.
- Do not jump to attention before the context-window milestones are done.
- Do not jump to `bt.nn` before the first raw attention prototype is finished.
- Include RNNs because they maximize understanding of sequence state and gradient flow.
- Do not include CNNs on the main path unless you later want a side learning branch.
- Do not start tokenizer work before the first stable decoder block exists.
- Do not start CUDA work before the model semantics are stable.
