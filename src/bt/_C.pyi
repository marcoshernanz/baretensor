"""BareTensor native extension (bootstrap)"""

from collections.abc import Sequence


class Tensor:
    @property
    def shape(self) -> list[int]: ...

def full(arg0: Sequence[int], arg1: float, /) -> Tensor: ...

def zeros(arg: Sequence[int], /) -> Tensor: ...

def ones(arg: Sequence[int], /) -> Tensor: ...
