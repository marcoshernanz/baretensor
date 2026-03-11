import unittest
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

import bt
from tests.utils import to_numpy

ArrayF32: TypeAlias = NDArray[np.float32]


def _as_f32(array: object) -> ArrayF32:
    return np.asarray(array, dtype=np.float32)


def _assert_matmul_matches_numpy(
    left: ArrayF32,
    right: ArrayF32,
) -> None:
    left_bt = bt.tensor(left)
    right_bt = bt.tensor(right)
    expected = _as_f32(np.matmul(left, right))

    out_method = left_bt.matmul(right_bt)
    out_operator = left_bt @ right_bt

    np.testing.assert_allclose(to_numpy(out_method), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(to_numpy(out_operator), expected, rtol=1e-6, atol=1e-6)


class MatmulTests(unittest.TestCase):
    def test_vector_vector_returns_scalar(self) -> None:
        left = _as_f32(np.array([1.0, 2.0, 3.0]))
        right = _as_f32(np.array([4.0, 5.0, 6.0]))
        _assert_matmul_matches_numpy(left, right)

    def test_matrix_matrix(self) -> None:
        left = _as_f32(np.arange(12).reshape(3, 4))
        right = _as_f32(np.arange(20).reshape(4, 5))
        _assert_matmul_matches_numpy(left, right)

    def test_vector_matrix(self) -> None:
        left = _as_f32(np.array([1.0, -2.0, 3.0, -4.0]))
        right = _as_f32(np.arange(12).reshape(4, 3))
        _assert_matmul_matches_numpy(left, right)

    def test_matrix_vector(self) -> None:
        left = _as_f32(np.arange(12).reshape(3, 4))
        right = _as_f32(np.array([0.5, 1.0, -1.0, 2.0]))
        _assert_matmul_matches_numpy(left, right)

    def test_batched_matrix_vector(self) -> None:
        left = _as_f32(np.arange(2 * 3 * 4).reshape(2, 3, 4))
        right = _as_f32(np.array([1.0, 2.0, 3.0, 4.0]))
        _assert_matmul_matches_numpy(left, right)

    def test_vector_batched_matrix(self) -> None:
        left = _as_f32(np.array([1.0, 0.5, -1.0, 2.0]))
        right = _as_f32(np.arange(2 * 4 * 3).reshape(2, 4, 3))
        _assert_matmul_matches_numpy(left, right)

    def test_batched_matrix_matrix(self) -> None:
        left = _as_f32(np.arange(2 * 3 * 4).reshape(2, 3, 4))
        right = _as_f32(np.arange(2 * 4 * 5).reshape(2, 4, 5))
        _assert_matmul_matches_numpy(left, right)

    def test_batched_matrix_with_broadcasted_matrix(self) -> None:
        left = _as_f32(np.arange(2 * 3 * 4).reshape(2, 3, 4))
        right = _as_f32(np.arange(4 * 5).reshape(4, 5))
        _assert_matmul_matches_numpy(left, right)

    def test_batched_matrix_broadcasts_batch_dimensions(self) -> None:
        left = _as_f32(np.arange(2 * 1 * 3 * 4).reshape(2, 1, 3, 4))
        right = _as_f32(np.arange(7 * 4 * 5).reshape(7, 4, 5))
        _assert_matmul_matches_numpy(left, right)

    def test_non_contiguous_inputs(self) -> None:
        left = bt.tensor(_as_f32(np.arange(12).reshape(3, 4))).transpose(0, 1)
        right = bt.tensor(_as_f32(np.arange(15).reshape(3, 5)))
        expected = _as_f32(np.matmul(_as_f32(np.arange(12).reshape(3, 4)).T, np.arange(15).reshape(3, 5)))

        out = left.matmul(right)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_zero_k_dimension(self) -> None:
        left = _as_f32(np.zeros((2, 0)))
        right = _as_f32(np.zeros((0, 3)))
        _assert_matmul_matches_numpy(left, right)

    def test_rank_zero_input_raises_with_context(self) -> None:
        scalar = bt.tensor(_as_f32(np.array(2.0)))
        vector = bt.tensor(_as_f32(np.array([1.0, 2.0])))
        with self.assertRaisesRegex(
            ValueError,
            r"matmul failed for tensors with shapes \[\] and \[2\]: both tensors must be at least 1-D",
        ):
            _ = scalar.matmul(vector)

    def test_inner_dimension_mismatch_raises_with_context(self) -> None:
        left = bt.tensor(_as_f32(np.zeros((3, 4))))
        right = bt.tensor(_as_f32(np.zeros((5, 2))))
        with self.assertRaisesRegex(
            ValueError,
            r"inner dimensions must match \(lhs.shape\[-1\] == rhs.shape\[-2\]\), got 4 and 5",
        ):
            _ = left.matmul(right)

    def test_batch_broadcast_mismatch_raises_with_context(self) -> None:
        left = bt.tensor(_as_f32(np.zeros((2, 3, 4))))
        right = bt.tensor(_as_f32(np.zeros((4, 4, 5))))
        with self.assertRaisesRegex(
            ValueError,
            r"batch dimensions are not broadcastable",
        ):
            _ = left.matmul(right)

    def test_mixed_dtype_matmul_rejects(self) -> None:
        left = bt.tensor(np.asarray([[1.0, 2.0]], dtype=np.float32))
        right = bt.tensor(np.asarray([[1], [2]], dtype=np.int64))

        with self.assertRaisesRegex(ValueError, r"same dtype"):
            _ = left.matmul(right)


if __name__ == "__main__":
    unittest.main()
