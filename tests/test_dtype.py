import unittest

import numpy as np

import bt
from tests.utils import to_numpy


class DTypeTests(unittest.TestCase):
    def test_int64_structural_ops_preserve_dtype(self) -> None:
        source = bt.tensor(np.arange(12, dtype=np.int64).reshape(3, 4))

        viewed = source.view([2, 6])
        transposed = source.transpose(0, 1)
        sliced = transposed[:, 1:3]
        contiguous = sliced.contiguous()

        self.assertEqual(viewed.dtype, bt.int64)
        self.assertEqual(transposed.dtype, bt.int64)
        self.assertEqual(sliced.dtype, bt.int64)
        self.assertEqual(contiguous.dtype, bt.int64)
        self.assertEqual(to_numpy(contiguous).dtype, np.int64)
        np.testing.assert_array_equal(
            to_numpy(contiguous),
            np.ascontiguousarray(np.arange(12, dtype=np.int64).reshape(3, 4).T[:, 1:3]),
        )

    def test_cat_preserves_int64_dtype(self) -> None:
        out = bt.cat([bt.tensor([1, 2]), bt.tensor([3, 4])], dim=0)

        self.assertEqual(out.dtype, bt.int64)
        np.testing.assert_array_equal(to_numpy(out), np.asarray([1, 2, 3, 4], dtype=np.int64))

    def test_cat_rejects_mixed_dtype(self) -> None:
        with self.assertRaisesRegex(ValueError, r"dtype"):
            _ = bt.cat([bt.tensor([1, 2]), bt.tensor([3.0, 4.0])], dim=0)

    def test_numpy_and_item_preserve_int64(self) -> None:
        scalar = bt.tensor(5)

        array = to_numpy(scalar)

        self.assertEqual(array.dtype, np.int64)
        self.assertEqual(scalar.item(), 5)


if __name__ == "__main__":
    unittest.main()
