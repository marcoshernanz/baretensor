import unittest

import numpy as np

import bt
from tests.utils import to_numpy


class ComparisonTests(unittest.TestCase):
    def test_tensor_scalar_greater_than_returns_bool_tensor(self) -> None:
        source_np = np.asarray([[-1.0, 0.0, 2.0], [3.0, -4.0, 5.0]], dtype=np.float32)
        source = bt.tensor(source_np, requires_grad=True)

        out = source > 0.0

        self.assertEqual(out.dtype, bt.bool)
        self.assertFalse(out.requires_grad)
        np.testing.assert_array_equal(to_numpy(out), source_np > 0.0)

    def test_tensor_tensor_broadcast_comparisons_match_numpy(self) -> None:
        lhs_np = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        rhs_np = np.asarray([[2], [4]], dtype=np.int64)
        lhs = bt.tensor(lhs_np)
        rhs = bt.tensor(rhs_np)

        np.testing.assert_array_equal(to_numpy(lhs == rhs), lhs_np == rhs_np)
        np.testing.assert_array_equal(to_numpy(lhs != rhs), lhs_np != rhs_np)
        np.testing.assert_array_equal(to_numpy(lhs < rhs), lhs_np < rhs_np)
        np.testing.assert_array_equal(to_numpy(lhs <= rhs), lhs_np <= rhs_np)
        np.testing.assert_array_equal(to_numpy(lhs > rhs), lhs_np > rhs_np)
        np.testing.assert_array_equal(to_numpy(lhs >= rhs), lhs_np >= rhs_np)

    def test_bool_tensor_comparison_is_supported(self) -> None:
        lhs = bt.tensor(np.asarray([True, False, True], dtype=np.bool_))
        rhs = bt.tensor(np.asarray([False, False, True], dtype=np.bool_))

        out = lhs == rhs

        self.assertEqual(out.dtype, bt.bool)
        np.testing.assert_array_equal(
            to_numpy(out), np.asarray([False, True, True], dtype=np.bool_)
        )


if __name__ == "__main__":
    unittest.main()
