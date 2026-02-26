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


if __name__ == "__main__":
    unittest.main()
