"""BareTensor Python package (bootstrap)."""

from ._C import Tensor, full, ones, tensor, zeros  # pylint: disable=no-name-in-module

__all__ = ["Tensor", "full", "zeros", "ones", "tensor"]
