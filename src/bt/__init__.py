"""BareTensor Python package (bootstrap)."""

from ._C import Tensor, full, zeros, ones  # pylint: disable=no-name-in-module

__all__ = ["Tensor", "full", "zeros", "ones"]
