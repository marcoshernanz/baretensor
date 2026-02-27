import unittest
import warnings
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

import bt
from tests.utils import to_numpy

ArrayF32 = NDArray[np.float32]


def _as_f32(array: object) -> ArrayF32:
    return np.asarray(array, dtype=np.float32)


def _numpy_mean_expected(
    array: ArrayF32,
    *,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> ArrayF32:
    axis: int | tuple[int, ...] | None
    if dim is None:
        axis = None
    elif isinstance(dim, int):
        axis = dim
    else:
        axis = tuple(dim)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.mean(array, axis=axis, keepdims=keepdim)
    return np.asarray(result, dtype=np.float32)


def _assert_mean_matches_numpy(
    array: ArrayF32,
    *,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> None:
    tensor = bt.tensor(array)
    out = tensor.mean(dim=dim, keepdim=keepdim)
    expected = _numpy_mean_expected(array, dim=dim, keepdim=keepdim)
    np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6, equal_nan=True)


class MeanTests(unittest.TestCase):
    def test_mean_all_elements_returns_scalar(self) -> None:
        source = _as_f32(np.arange(12).reshape(3, 4))
        _assert_mean_matches_numpy(source)

    def test_mean_all_elements_with_keepdim(self) -> None:
        source = _as_f32(np.arange(2 * 3 * 4).reshape(2, 3, 4))
        _assert_mean_matches_numpy(source, keepdim=True)

    def test_mean_single_dim(self) -> None:
        source = _as_f32(np.arange(2 * 3 * 4).reshape(2, 3, 4))
        _assert_mean_matches_numpy(source, dim=1)

    def test_mean_single_dim_keepdim(self) -> None:
        source = _as_f32(np.arange(2 * 3 * 4).reshape(2, 3, 4))
        _assert_mean_matches_numpy(source, dim=1, keepdim=True)

    def test_mean_negative_dim(self) -> None:
        source = _as_f32(np.arange(2 * 3 * 4).reshape(2, 3, 4))
        _assert_mean_matches_numpy(source, dim=-1)

    def test_mean_multiple_dims_tuple(self) -> None:
        source = _as_f32(np.arange(4 * 5 * 6).reshape(4, 5, 6))
        _assert_mean_matches_numpy(source, dim=(2, 1))

    def test_mean_multiple_dims_list_keepdim(self) -> None:
        source = _as_f32(np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5))
        _assert_mean_matches_numpy(source, dim=[1, 3], keepdim=True)

    def test_mean_empty_dim_sequence_performs_no_reduction(self) -> None:
        source = _as_f32(np.arange(12).reshape(3, 4))
        _assert_mean_matches_numpy(source, dim=[])

    def test_mean_non_contiguous_input(self) -> None:
        source = _as_f32(np.arange(2 * 3 * 4).reshape(2, 3, 4))
        transposed = bt.tensor(source).transpose(0, 2)
        expected = _as_f32(np.mean(np.transpose(source, (2, 1, 0)), axis=1))
        np.testing.assert_allclose(
            to_numpy(transposed.mean(dim=1)),
            expected,
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )

    def test_mean_scalar_tensor_returns_scalar(self) -> None:
        source = _as_f32(np.asarray(7.5))
        _assert_mean_matches_numpy(source)

    def test_mean_zero_extent_all_dims_returns_nan(self) -> None:
        source = _as_f32(np.zeros((0, 3)))
        _assert_mean_matches_numpy(source)

    def test_mean_zero_extent_partial_dim_returns_nan(self) -> None:
        source = _as_f32(np.zeros((2, 0, 4)))
        _assert_mean_matches_numpy(source, dim=1)
        _assert_mean_matches_numpy(source, dim=1, keepdim=True)

    def test_mean_invalid_dim_raises_with_context(self) -> None:
        tensor = bt.tensor(_as_f32(np.zeros((2, 3))))
        with self.assertRaisesRegex(
            ValueError,
            r"mean failed for tensor with shape \[2, 3\]: dim\[0\]=2 is out of range for rank 2\.",
        ):
            _ = tensor.mean(dim=2)

    def test_mean_duplicate_dims_raise_with_context(self) -> None:
        tensor = bt.tensor(_as_f32(np.zeros((2, 3, 4))))
        with self.assertRaisesRegex(
            ValueError,
            r"mean failed for tensor with shape \[2, 3, 4\]: dimension 1 appears more than once in dim\.",
        ):
            _ = tensor.mean(dim=[1, -2])

    def test_mean_rejects_non_integer_dim(self) -> None:
        tensor = bt.tensor(_as_f32(np.zeros((2, 3))))
        with self.assertRaisesRegex(
            TypeError,
            r"mean\(\) expected 'dim' to be an int, a sequence of ints, or None\.",
        ):
            _ = tensor.mean(dim=cast(Any, "1"))


if __name__ == "__main__":
    unittest.main()
