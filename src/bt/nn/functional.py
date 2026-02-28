"""Functional neural-network primitives."""

from __future__ import annotations

from typing import Literal

from bt._C import Tensor, cross_entropy as _cross_entropy, embedding as _embedding

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


def embedding(
    input: Tensor,
    weight: Tensor,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    """
    Lookup embeddings for index tensor `input` from embedding matrix `weight`.

    Supported now:
    - input: arbitrary index tensor shape
    - weight: [V, D]
    """
    if max_norm is not None:
        raise NotImplementedError("embedding() does not support max_norm yet.")
    if norm_type != 2.0:
        raise NotImplementedError("embedding() does not support norm_type != 2.0 yet.")
    if scale_grad_by_freq:
        raise NotImplementedError("embedding() does not support scale_grad_by_freq yet.")
    if sparse:
        raise NotImplementedError("embedding() does not support sparse gradients yet.")

    return _embedding(input=input, weight=weight)
