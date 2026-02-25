import unittest

import numpy as np

import bt
from tests.utils import to_numpy


class ReshapeTests(unittest.TestCase):
    def test_reshape_preserves_values(self) -> None:
        source = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor = bt.tensor(source)

        reshaped = tensor.reshape([2, 6])

        self.assertEqual(reshaped.shape, [2, 6])
        np.testing.assert_allclose(to_numpy(reshaped), source.reshape(2, 6))

    def test_reshape_infers_single_negative_dimension(self) -> None:
        source = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor = bt.tensor(source)

        reshaped = tensor.reshape([-1, 2])

        self.assertEqual(reshaped.shape, [6, 2])
        np.testing.assert_allclose(to_numpy(reshaped), source.reshape(6, 2))

    def test_reshape_rejects_multiple_negative_one_dimensions(self) -> None:
        tensor = bt.tensor(np.arange(6, dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"at most one '-1' dimension is allowed",
        ):
            _ = tensor.reshape([-1, -1])

    def test_reshape_rejects_invalid_negative_dimension(self) -> None:
        tensor = bt.tensor(np.arange(6, dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"has invalid size -2",
        ):
            _ = tensor.reshape([3, -2])

    def test_reshape_rejects_mismatched_numel(self) -> None:
        tensor = bt.tensor(np.arange(6, dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"element counts differ",
        ):
            _ = tensor.reshape([4, 2])

    def test_reshape_rejects_ambiguous_zero_product_inference(self) -> None:
        tensor = bt.zeros([0, 4])

        with self.assertRaisesRegex(
            ValueError,
            r"cannot infer '-1' when known dimensions multiply to zero",
        ):
            _ = tensor.reshape([-1, 0])


if __name__ == "__main__":
    unittest.main()
