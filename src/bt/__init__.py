"""BareTensor Python package (bootstrap)."""

from ._C import Tensor, full, zeros, ones, tensor  # pylint: disable=no-name-in-module

__all__ = ["Tensor", "full", "zeros", "ones", "tensor"]
