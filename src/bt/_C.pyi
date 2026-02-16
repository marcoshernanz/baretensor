"""BareTensor native extension (bootstrap)"""

from collections.abc import Sequence
from typing import overload


class Tensor:
    @property
    def shape(self) -> list[int]: ...

    @overload
    def fill(self, arg0: Sequence[int], arg1: float, /) -> Tensor: ...

    @overload
    def fill(self, arg: Sequence[int], /) -> Tensor: ...

    @overload
    def fill(self, arg: Sequence[int], /) -> Tensor: ...
