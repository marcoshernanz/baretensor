import unittest
from typing import Any, cast

import numpy as np

import bt
from tests.utils import to_numpy


class ConstructorsAndMetadataTests(unittest.TestCase):
    def test_full_zeros_ones_values(self) -> None:
        shape = [2, 3]

        full = bt.full(shape, 7.5)
        zeros = bt.zeros(shape)
        ones = bt.ones(shape)

        np.testing.assert_allclose(to_numpy(full), np.full(shape, 7.5, dtype=np.float32))
        np.testing.assert_allclose(to_numpy(zeros), np.zeros(shape, dtype=np.float32))
        np.testing.assert_allclose(to_numpy(ones), np.ones(shape, dtype=np.float32))

    def test_tensor_from_numpy_copies_data(self) -> None:
        source = np.arange(6, dtype=np.float32).reshape(2, 3)
        tensor = bt.tensor(source)

        source[0, 0] = 999.0
        expected = np.arange(6, dtype=np.float32).reshape(2, 3)
        np.testing.assert_allclose(to_numpy(tensor), expected)

    def test_scalar_tensor_metadata(self) -> None:
        tensor = bt.tensor(np.asarray(3.5, dtype=np.float32))

        self.assertEqual(tensor.shape, [])
        self.assertEqual(tensor.strides, [])
        self.assertEqual(tensor.dim(), 0)
        self.assertEqual(tensor.numel(), 1)
        self.assertTrue(tensor.is_contiguous())
        np.testing.assert_allclose(to_numpy(tensor), np.asarray(3.5, dtype=np.float32))

    def test_zero_dim_extent_metadata(self) -> None:
        tensor = bt.zeros([0, 3])

        self.assertEqual(tensor.shape, [0, 3])
        self.assertEqual(tensor.strides, [3, 1])
        self.assertEqual(tensor.dim(), 2)
        self.assertEqual(tensor.numel(), 0)
        self.assertTrue(tensor.is_contiguous())
        np.testing.assert_allclose(to_numpy(tensor), np.zeros((0, 3), dtype=np.float32))

    def test_negative_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = bt.zeros([2, -1])

        with self.assertRaises(ValueError):
            _ = bt.full([-3], 1.0)

    def test_tensor_accepts_non_float32_and_non_contiguous_numpy(self) -> None:
        float64 = np.arange(6, dtype=np.float64).reshape(2, 3)
        non_contiguous = np.arange(6, dtype=np.float32).reshape(3, 2).T
        self.assertFalse(non_contiguous.flags["C_CONTIGUOUS"])

        from_float64 = bt.tensor(cast(Any, float64))
        from_non_contiguous = bt.tensor(cast(Any, non_contiguous))

        np.testing.assert_allclose(to_numpy(from_float64), np.asarray(float64, dtype=np.float32))
        np.testing.assert_allclose(
            to_numpy(from_non_contiguous), np.asarray(non_contiguous, dtype=np.float32)
        )


if __name__ == "__main__":
    unittest.main()
