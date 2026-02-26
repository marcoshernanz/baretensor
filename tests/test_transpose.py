import unittest

import numpy as np

import bt
from tests.utils import to_numpy


class TransposeTests(unittest.TestCase):
    def test_transpose_swaps_requested_dimensions(self) -> None:
        source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        tensor = bt.tensor(source)

        transposed = tensor.transpose(0, 2)

        self.assertEqual(transposed.shape, [4, 3, 2])
        self.assertEqual(transposed.strides, [1, 4, 12])
        self.assertFalse(transposed.is_contiguous())
        np.testing.assert_allclose(to_numpy(transposed), np.transpose(source, (2, 1, 0)))

    def test_transpose_accepts_negative_dimensions(self) -> None:
        source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        tensor = bt.tensor(source)

        transposed = tensor.transpose(-1, -3)

        self.assertEqual(transposed.shape, [4, 3, 2])
        self.assertEqual(transposed.strides, [1, 4, 12])
        np.testing.assert_allclose(to_numpy(transposed), np.transpose(source, (2, 1, 0)))

    def test_transpose_with_same_dimension_returns_equivalent_view(self) -> None:
        source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        tensor = bt.tensor(source)

        transposed = tensor.transpose(1, 1)

        self.assertEqual(transposed.shape, [2, 3, 4])
        self.assertEqual(transposed.strides, [12, 4, 1])
        self.assertTrue(transposed.is_contiguous())
        np.testing.assert_allclose(to_numpy(transposed), source)

    def test_transpose_rejects_out_of_range_dimensions(self) -> None:
        tensor = bt.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4))

        with self.assertRaisesRegex(
            ValueError,
            r"transpose failed for tensor with shape \[2, 3, 4\]: dim0=-4, dim1=1\.",
        ):
            _ = tensor.transpose(-4, 1)

    def test_T_transposes_2d_tensor(self) -> None:
        source = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor = bt.tensor(source)

        transposed = tensor.T

        self.assertEqual(transposed.shape, [4, 3])
        self.assertEqual(transposed.strides, [1, 4])
        np.testing.assert_allclose(to_numpy(transposed), source.T)

    def test_T_rejects_non_2d_tensor(self) -> None:
        tensor = bt.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4))

        with self.assertRaisesRegex(
            ValueError,
            r"T failed for tensor with shape \[2, 3, 4\]: expected ndim\(\) == 2, but got 3\.",
        ):
            _ = tensor.T

    def test_mT_transposes_last_two_dimensions(self) -> None:
        source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        tensor = bt.tensor(source)

        transposed = tensor.mT

        self.assertEqual(transposed.shape, [2, 4, 3])
        self.assertEqual(transposed.strides, [12, 1, 4])
        np.testing.assert_allclose(to_numpy(transposed), np.swapaxes(source, -1, -2))

    def test_mT_rejects_rank_less_than_two(self) -> None:
        tensor = bt.tensor(np.arange(5, dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"mT failed for tensor with shape \[5\]: expected ndim\(\) >= 2, but got 1\.",
        ):
            _ = tensor.mT


if __name__ == "__main__":
    unittest.main()
