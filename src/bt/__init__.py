"""BareTensor Python package."""

from collections.abc import Sequence
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from . import _C
from . import nn
from ._C import (
    DType,
    Tensor,
    cat as _cat,
    float32,
    full as _full,
    int64,
    ones as _ones,
    stack as _stack,
    tensor_from_numpy as _tensor_from_numpy,
    zeros as _zeros,
)

_INT64_INFO = np.iinfo(np.int64)


class _NoGradContext(AbstractContextManager[None]):
    def __init__(self) -> None:
        self._guard: Any | None = None

    def __enter__(self) -> None:
        guard = getattr(_C, "_NoGradGuard")()
        guard.__enter__()
        self._guard = guard
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        guard = self._guard
        if guard is None:
            return False
        try:
            guard.close()
        finally:
            self._guard = None
        return False


def _normalize_dtype(dtype: DType) -> DType:
    if dtype in (float32, int64):
        return dtype
    raise TypeError("dtype must be one of {bt.float32, bt.int64}.")


def _is_numpy_input(data: ArrayLike) -> bool:
    return isinstance(data, (np.ndarray, np.generic))


def _infer_dtype(array: np.ndarray[Any, Any], *, from_numpy: bool) -> DType:
    if from_numpy:
        if array.dtype == np.float32:
            return float32
        if array.dtype == np.int64:
            return int64
        raise TypeError(
            "Unsupported NumPy dtype "
            f"{array.dtype}. Pass dtype=bt.float32 or dtype=bt.int64 explicitly."
        )

    kind = array.dtype.kind
    if kind in ("i", "u"):
        return int64
    if kind == "f":
        return float32
    raise TypeError("bt.tensor() only supports integer and floating-point inputs.")


def _coerce_to_float32(array: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.float32]]:
    kind = array.dtype.kind
    if kind not in ("i", "u", "f"):
        raise TypeError("bt.tensor(..., dtype=bt.float32) only supports numeric inputs.")
    return np.asarray(array, dtype=np.float32, order="C")


def _coerce_to_int64(array: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.int64]]:
    kind = array.dtype.kind
    if kind in ("i", "u"):
        if kind == "u" and array.size != 0:
            unsigned = np.asarray(array, dtype=np.uint64, order="C")
            if np.any(unsigned > np.uint64(_INT64_INFO.max)):
                raise ValueError(
                    "bt.tensor(..., dtype=bt.int64) received a value outside int64 range."
                )
        return np.asarray(array, dtype=np.int64, order="C")

    if kind == "f":
        float_array = np.asarray(array, dtype=np.float64, order="C")
        if not np.isfinite(float_array).all():
            raise ValueError("bt.tensor(..., dtype=bt.int64) requires finite values.")
        truncated = np.trunc(float_array)
        if not np.array_equal(float_array, truncated):
            raise ValueError("bt.tensor(..., dtype=bt.int64) requires integer-valued floats.")
        if truncated.size != 0:
            if np.any(truncated < _INT64_INFO.min) or np.any(truncated > _INT64_INFO.max):
                raise ValueError(
                    "bt.tensor(..., dtype=bt.int64) received a value outside int64 range."
                )
        return np.asarray(truncated, dtype=np.int64, order="C")

    raise TypeError("bt.tensor(..., dtype=bt.int64) only supports numeric inputs.")


def no_grad() -> AbstractContextManager[None]:
    """Disable gradient recording within a ``with`` block."""
    return _NoGradContext()


def tensor(data: ArrayLike, *, dtype: DType | None = None, requires_grad: bool = False) -> Tensor:
    """Create a tensor from NumPy-compatible array-like input."""
    normalized_dtype = None if dtype is None else _normalize_dtype(dtype)
    array = np.asarray(data)
    target_dtype: DType = (
        normalized_dtype
        if normalized_dtype is not None
        else _infer_dtype(array, from_numpy=_is_numpy_input(data))
    )
    if target_dtype == float32:
        return _tensor_from_numpy(
            _coerce_to_float32(array), target_dtype, requires_grad=requires_grad
        )
    return _tensor_from_numpy(_coerce_to_int64(array), target_dtype, requires_grad=requires_grad)


def full(
    shape: Sequence[int],
    fill_value: float | int,
    *,
    dtype: DType = float32,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor filled with a constant value."""
    return _full(list(shape), fill_value, _normalize_dtype(dtype), requires_grad)


def zeros(shape: Sequence[int], *, dtype: DType = float32, requires_grad: bool = False) -> Tensor:
    """Create a tensor filled with zeros."""
    return _zeros(list(shape), _normalize_dtype(dtype), requires_grad)


def ones(shape: Sequence[int], *, dtype: DType = float32, requires_grad: bool = False) -> Tensor:
    """Create a tensor filled with ones."""
    return _ones(list(shape), _normalize_dtype(dtype), requires_grad)


def cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    """Concatenate tensors along an existing dimension."""
    return _cat(list(tensors), dim=dim)


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    """Stack tensors along a new dimension."""
    return _stack(list(tensors), dim=dim)


__all__ = [
    "DType",
    "Tensor",
    "cat",
    "float32",
    "full",
    "int64",
    "nn",
    "no_grad",
    "ones",
    "stack",
    "tensor",
    "zeros",
]
