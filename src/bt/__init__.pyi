from collections.abc import Sequence
from contextlib import AbstractContextManager

from numpy.typing import ArrayLike

from . import nn as nn
from ._C import DType as DType
from ._C import Tensor as Tensor
from ._C import float32 as float32
from ._C import int64 as int64

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

def no_grad() -> AbstractContextManager[None]: ...

def tensor(
    data: ArrayLike, *, dtype: DType | None = None, requires_grad: bool = False
) -> Tensor: ...

def full(
    shape: Sequence[int],
    fill_value: float | int,
    *,
    dtype: DType = float32,
    requires_grad: bool = False,
) -> Tensor: ...

def zeros(
    shape: Sequence[int], *, dtype: DType = float32, requires_grad: bool = False
) -> Tensor: ...

def ones(
    shape: Sequence[int], *, dtype: DType = float32, requires_grad: bool = False
) -> Tensor: ...

def cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor: ...

def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor: ...
