from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import bt


def to_numpy(tensor: bt.Tensor) -> NDArray[np.float32]:
    data: NDArray[np.float32] = np.asarray(tensor.tolist(), dtype=np.float32)
    return np.asarray(data.reshape(tuple(tensor.shape)), dtype=np.float32)
