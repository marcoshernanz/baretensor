import unittest

import numpy as np

import bt
from tests.utils import to_numpy


class PermuteTests(unittest.TestCase):
    def test_permute_reorders_shape_strides_and_values(self) -> None:
        source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        tensor = bt.tensor(source)

        permuted = tensor.permute([2, 0, 1])

        self.assertEqual(permuted.shape, [4, 2, 3])
        self.assertEqual(permuted.strides, [1, 12, 4])
        self.assertFalse(permuted.is_contiguous())
        np.testing.assert_allclose(to_numpy(permuted), np.transpose(source, (2, 0, 1)))

    def test_permute_accepts_negative_dimensions(self) -> None:
        source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        tensor = bt.tensor(source)

        permuted = tensor.permute([-1, -3, -2])

        self.assertEqual(permuted.shape, [4, 2, 3])
        self.assertEqual(permuted.strides, [1, 12, 4])
        np.testing.assert_allclose(to_numpy(permuted), np.transpose(source, (2, 0, 1)))

    def test_permute_rejects_rank_mismatch(self) -> None:
        tensor = bt.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4))

        with self.assertRaisesRegex(
            ValueError,
            r"expected 3 dims but got 2",
        ):
            _ = tensor.permute([0, 1])

    def test_permute_rejects_out_of_range_dimension(self) -> None:
        tensor = bt.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4))

        with self.assertRaisesRegex(
            ValueError,
            r"dims\[1\]=5 is out of range for rank 3",
        ):
            _ = tensor.permute([0, 5, 1])

    def test_permute_rejects_duplicate_dimension(self) -> None:
        tensor = bt.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4))

        with self.assertRaisesRegex(
            ValueError,
            r"dimension 1 appears more than once",
        ):
            _ = tensor.permute([1, 1, 2])

    def test_permute_matches_transpose_dimension_swap(self) -> None:
        source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        tensor = bt.tensor(source)

        from_permute = tensor.permute([2, 1, 0])
        from_transpose = tensor.transpose(0, 2)

        self.assertEqual(from_permute.shape, from_transpose.shape)
        self.assertEqual(from_permute.strides, from_transpose.strides)
        np.testing.assert_allclose(to_numpy(from_permute), to_numpy(from_transpose))


if __name__ == "__main__":
    unittest.main()
