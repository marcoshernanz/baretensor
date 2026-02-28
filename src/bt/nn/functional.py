"""Functional neural-network primitives."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

from bt._C import (
    Tensor,
    cross_entropy as _cross_entropy,
    embedding as _embedding,
    layer_norm as _layer_norm,
)

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


def _normalize_layer_norm_shape(normalized_shape: int | Sequence[int]) -> list[int]:
    if isinstance(normalized_shape, bool):
        raise TypeError("layer_norm() expected 'normalized_shape' to be an int or a sequence of ints.")

    if isinstance(normalized_shape, int):
        return [normalized_shape]

    try:
        dims = list(normalized_shape)
    except TypeError as exc:
        raise TypeError(
            "layer_norm() expected 'normalized_shape' to be an int or a sequence of ints."
        ) from exc

    dims_any: list[object] = list(dims)
    if any(isinstance(dim, bool) or not isinstance(dim, int) for dim in dims_any):
        raise TypeError("layer_norm() expected 'normalized_shape' to contain only ints.")
    return cast(list[int], dims_any)


def layer_norm(
    input: Tensor,
    normalized_shape: int | Sequence[int],
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """
    Apply layer normalization over the last len(normalized_shape) dimensions.

    TinyGPT scope:
    - input: arbitrary rank tensor
    - normalized_shape: int or sequence[int], matching trailing input dims
    - optional affine parameters weight and bias with shape normalized_shape
    """
    normalized_shape_list = _normalize_layer_norm_shape(normalized_shape)
    return _layer_norm(
        input=input,
        normalized_shape=normalized_shape_list,
        weight=weight,
        bias=bias,
        eps=eps,
    )
