"""BareTensor Python package (bootstrap).

This is intentionally tiny for milestone A1: prove native build + import works.
"""

from ._C import Tensor

__all__ = ["Tensor"]
