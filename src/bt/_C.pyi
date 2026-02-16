"""BareTensor native extension (bootstrap)"""

from collections.abc import Sequence


class Tensor:
    def __init__(self, arg: Sequence[int], /) -> None: ...

    @property
    def shape(self) -> list[int]: ...
