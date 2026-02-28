"""Functional neural-network primitives."""

from __future__ import annotations

from typing import Literal

from bt._C import Tensor, cross_entropy as _cross_entropy

Reduction = Literal["none", "mean", "sum"]


def cross_entropy(
    input: Tensor,
    target: Tensor,
    ignore_index: int = -100,
    reduction: Reduction = "mean",
) -> Tensor:
    """
    Compute cross-entropy loss for TinyGPT-style class-index targets.

    Current supported shapes:
    - input: [C] logits, target: [] scalar class index
    - input: [N, C, d1, ..., dK] logits
    - target: [N, d1, ..., dK] class indices (integer-valued float32)
    """
    return _cross_entropy(
        input=input,
        target=target,
        ignore_index=ignore_index,
        reduction=reduction,
    )
