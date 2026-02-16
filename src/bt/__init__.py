"""BareTensor Python package (bootstrap)."""

from ._C import Tensor, fill, zeros, ones  # pylint: disable=no-name-in-module

__all__ = ["Tensor", "fill", "zeros", "ones"]
