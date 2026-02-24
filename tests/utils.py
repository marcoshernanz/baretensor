from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any, cast

import bt


def to_numpy(tensor: bt.Tensor) -> NDArray[np.float32]:
    return np.asarray(cast(Any, tensor.numpy()), dtype=np.float32)
