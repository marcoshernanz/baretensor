"""BareTensor native extension (bootstrap)"""

from typing import overload


def add(a: int, b: int = 1) -> int:
    """
    Add two integers (default b=1).

    This exists only to validate the nanobind toolchain.
    """

class Dog:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: str, /) -> None: ...

    def bark(self) -> str: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...
