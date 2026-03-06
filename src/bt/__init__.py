"""BareTensor Python package (bootstrap)."""

from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any

from . import _C
from ._C import Tensor, full, ones, tensor, zeros  # pylint: disable=no-name-in-module
from . import nn


class _NoGradContext(AbstractContextManager[None]):
    def __init__(self) -> None:
        self._guard: Any | None = None

    def __enter__(self) -> None:
        guard_factory = getattr(_C, "_NoGradGuard")
        self._guard = guard_factory()
        self._guard.__enter__()
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        guard = self._guard
        if guard is None:
            return False
        try:
            guard.close()
        finally:
            self._guard = None
        return False


def no_grad() -> AbstractContextManager[None]:
    """Disable gradient recording within a ``with`` block."""
    return _NoGradContext()


__all__ = ["Tensor", "full", "zeros", "ones", "tensor", "nn", "no_grad"]
