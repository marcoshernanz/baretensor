import unittest
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

import bt
from tests.utils import to_numpy

ArrayF32 = NDArray[np.float32]


def _as_f32(array: object) -> ArrayF32:
    return np.asarray(array, dtype=np.float32)


def _numpy_max_expected(
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
    return np.asarray(np.max(array, axis=axis, keepdims=keepdim), dtype=np.float32)


def _assert_max_matches_numpy(
    array: ArrayF32,
    *,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> None:
    tensor = bt.tensor(array)
    out = tensor.max(dim=dim, keepdim=keepdim)
    expected = _numpy_max_expected(array, dim=dim, keepdim=keepdim)
    np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)


class MaxTests(unittest.TestCase):
    def test_max_all_elements_returns_scalar(self) -> None:
        source = _as_f32(np.arange(12).reshape(3, 4))
        _assert_max_matches_numpy(source)

    def test_max_all_elements_with_keepdim(self) -> None:
        source = _as_f32(np.arange(2 * 3 * 4).reshape(2, 3, 4))
        _assert_max_matches_numpy(source, keepdim=True)

    def test_max_single_dim(self) -> None:
        source = _as_f32(np.arange(2 * 3 * 4).reshape(2, 3, 4))
        _assert_max_matches_numpy(source, dim=1)

    def test_max_single_dim_keepdim(self) -> None:
        source = _as_f32(np.arange(2 * 3 * 4).reshape(2, 3, 4))
        _assert_max_matches_numpy(source, dim=1, keepdim=True)

    def test_max_negative_dim(self) -> None:
        source = _as_f32(np.arange(2 * 3 * 4).reshape(2, 3, 4))
        _assert_max_matches_numpy(source, dim=-1)

    def test_max_multiple_dims_tuple(self) -> None:
        source = _as_f32(np.arange(4 * 5 * 6).reshape(4, 5, 6))
        _assert_max_matches_numpy(source, dim=(2, 1))

    def test_max_multiple_dims_list_keepdim(self) -> None:
        source = _as_f32(np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5))
        _assert_max_matches_numpy(source, dim=[1, 3], keepdim=True)

    def test_max_empty_dim_sequence_performs_no_reduction(self) -> None:
        source = _as_f32(np.arange(12).reshape(3, 4))
        _assert_max_matches_numpy(source, dim=[])

    def test_max_non_contiguous_input(self) -> None:
        source = _as_f32(np.arange(2 * 3 * 4).reshape(2, 3, 4))
        transposed = bt.tensor(source).transpose(0, 2)
        expected = _as_f32(np.max(np.transpose(source, (2, 1, 0)), axis=1))
        np.testing.assert_allclose(
            to_numpy(transposed.max(dim=1)),
            expected,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_max_scalar_tensor_returns_scalar(self) -> None:
        source = _as_f32(np.asarray(7.5))
        _assert_max_matches_numpy(source)

    def test_max_empty_tensor_without_reduction_returns_empty_tensor(self) -> None:
        source = _as_f32(np.zeros((0, 3)))
        _assert_max_matches_numpy(source, dim=[])

    def test_max_all_dims_reduction_on_empty_tensor_raises_with_context(self) -> None:
        tensor = bt.tensor(_as_f32(np.zeros((0, 3))))
        with self.assertRaisesRegex(
            ValueError,
            r"max failed for tensor with shape \[0, 3\] and dim \[0, 1\]: cannot perform reduction over zero elements\.",
        ):
            _ = tensor.max()

    def test_max_reduction_over_zero_extent_dim_raises_with_context(self) -> None:
        tensor = bt.tensor(_as_f32(np.zeros((2, 0, 4))))
        with self.assertRaisesRegex(
            ValueError,
            r"max failed for tensor with shape \[2, 0, 4\] and dim \[1\]: cannot perform reduction over zero elements\.",
        ):
            _ = tensor.max(dim=1)

    def test_max_invalid_dim_raises_with_context(self) -> None:
        tensor = bt.tensor(_as_f32(np.zeros((2, 3))))
        with self.assertRaisesRegex(
            ValueError,
            r"max failed for tensor with shape \[2, 3\]: dim\[0\]=2 is out of range for rank 2\.",
        ):
            _ = tensor.max(dim=2)

    def test_max_duplicate_dims_raise_with_context(self) -> None:
        tensor = bt.tensor(_as_f32(np.zeros((2, 3, 4))))
        with self.assertRaisesRegex(
            ValueError,
            r"max failed for tensor with shape \[2, 3, 4\]: dimension 1 appears more than once in dim\.",
        ):
            _ = tensor.max(dim=[1, -2])

    def test_max_rejects_non_integer_dim(self) -> None:
        tensor = bt.tensor(_as_f32(np.zeros((2, 3))))
        with self.assertRaisesRegex(
            TypeError,
            r"max\(\) expected 'dim' to be an int, a sequence of ints, or None\.",
        ):
            _ = tensor.max(dim=cast(Any, "1"))


if __name__ == "__main__":
    unittest.main()
