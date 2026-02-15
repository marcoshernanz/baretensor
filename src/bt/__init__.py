"""BareTensor Python package (bootstrap).

This is intentionally tiny for milestone A1: prove native build + import works.
"""

from ._C import Dog, add  # pylint: disable=no-name-in-module

__all__ = ["Dog", "add"]
