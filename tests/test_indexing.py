import unittest

import numpy as np

import bt
from tests.utils import to_numpy


def _require_grad(tensor: bt.Tensor) -> bt.Tensor:
    grad = tensor.grad
    assert grad is not None
    return grad


class IndexingTests(unittest.TestCase):
    def test_integer_index_drops_dimension(self) -> None:
        x_np = np.asarray(np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4), dtype=np.float32)
        x = bt.tensor(x_np)

        out = x[1]

        self.assertEqual(out.shape, [3, 4])
        np.testing.assert_allclose(to_numpy(out), x_np[1], rtol=1e-6, atol=1e-6)

    def test_negative_integer_index(self) -> None:
        x_np = np.asarray(np.arange(3 * 4, dtype=np.float32).reshape(3, 4), dtype=np.float32)
        x = bt.tensor(x_np)

        out = x[-1]

        self.assertEqual(out.shape, [4])
        np.testing.assert_allclose(to_numpy(out), x_np[-1], rtol=1e-6, atol=1e-6)

    def test_slice_with_step_matches_numpy(self) -> None:
        x_np = np.asarray(np.arange(5 * 6, dtype=np.float32).reshape(5, 6), dtype=np.float32)
        x = bt.tensor(x_np)

        out = x[1:5:2, 1:6:3]

        np.testing.assert_allclose(to_numpy(out), x_np[1:5:2, 1:6:3], rtol=1e-6, atol=1e-6)

    def test_comma_indexing_matches_chained_indexing(self) -> None:
        x_np = np.asarray(np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5), dtype=np.float32)
        x = bt.tensor(x_np)

        comma = x[1, 2]
        chained = x[1][2]

        np.testing.assert_allclose(to_numpy(comma), x_np[1, 2], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(to_numpy(comma), to_numpy(chained), rtol=1e-6, atol=1e-6)

    def test_missing_trailing_dims_are_full_slices(self) -> None:
        x_np = np.asarray(np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4), dtype=np.float32)
        x = bt.tensor(x_np)

        out = x[1, 2]

        np.testing.assert_allclose(to_numpy(out), x_np[1, 2], rtol=1e-6, atol=1e-6)

    def test_empty_tuple_index_returns_scalar(self) -> None:
        scalar_np = np.asarray(3.5, dtype=np.float32)
        scalar = bt.tensor(scalar_np)

        out = scalar[()]

        self.assertEqual(out.shape, [])
        np.testing.assert_allclose(to_numpy(out), scalar_np, rtol=1e-6, atol=1e-6)

    def test_too_many_indices_raises(self) -> None:
        x = bt.zeros([2, 3])
        with self.assertRaisesRegex(IndexError, r"too many indices for tensor of dimension 2"):
            _ = x[0, 1, 2]

    def test_ellipsis_is_not_supported_yet(self) -> None:
        x = bt.zeros([2, 3])
        with self.assertRaisesRegex(TypeError, r"does not support ellipsis"):
            _ = x[...]

    def test_none_index_is_not_supported_yet(self) -> None:
        x = bt.zeros([2, 3])
        with self.assertRaisesRegex(TypeError, r"does not support None/newaxis indexing yet"):
            _ = x[None]

    def test_list_index_is_not_supported_yet(self) -> None:
        x = bt.zeros([4])
        with self.assertRaisesRegex(TypeError, r"only supports int, slice, and tuples thereof"):
            _ = x[[0, 1]]

    def test_boolean_index_is_not_supported_yet(self) -> None:
        x = bt.zeros([4])
        with self.assertRaisesRegex(TypeError, r"does not support boolean indices"):
            _ = x[True]

    def test_slice_rejects_non_positive_step(self) -> None:
        x = bt.zeros([4])
        with self.assertRaisesRegex(ValueError, r"step must be greater than 0, got 0"):
            _ = x[::0]
        with self.assertRaisesRegex(ValueError, r"step must be greater than 0, got -1"):
            _ = x[::-1]

    def test_select_backward_scatter(self) -> None:
        x_np = np.asarray(np.arange(3 * 4, dtype=np.float32).reshape(3, 4), dtype=np.float32)
        w_np = np.asarray(np.linspace(-1.0, 1.0, num=4, dtype=np.float32), dtype=np.float32)

        x = bt.tensor(x_np, requires_grad=True)
        w = bt.tensor(w_np)
        loss = (x[1] * w).sum()
        loss.backward()

        expected_grad = np.zeros_like(x_np, dtype=np.float32)
        expected_grad[1] = w_np
        np.testing.assert_allclose(
            to_numpy(_require_grad(x)),
            expected_grad,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_slice_backward_scatter_with_step(self) -> None:
        x_np = np.asarray(np.arange(5 * 4, dtype=np.float32).reshape(5, 4), dtype=np.float32)
        w_np = np.asarray(np.arange(2 * 4, dtype=np.float32).reshape(2, 4), dtype=np.float32)

        x = bt.tensor(x_np, requires_grad=True)
        w = bt.tensor(w_np)
        loss = (x[1:5:2, :] * w).sum()
        loss.backward()

        expected_grad = np.zeros_like(x_np, dtype=np.float32)
        expected_grad[1] = w_np[0]
        expected_grad[3] = w_np[1]
        np.testing.assert_allclose(
            to_numpy(_require_grad(x)),
            expected_grad,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_mixed_indexing_backward_scatter(self) -> None:
        x_np = np.asarray(np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5), dtype=np.float32)
        w_np = np.asarray(
            np.linspace(-0.5, 0.5, num=2 * 5, dtype=np.float32).reshape(2, 5), dtype=np.float32
        )

        x = bt.tensor(x_np, requires_grad=True)
        w = bt.tensor(w_np)
        loss = (x[1:, 2, :] * w).sum()
        loss.backward()

        expected_grad = np.zeros_like(x_np, dtype=np.float32)
        expected_grad[1:, 2, :] = w_np
        np.testing.assert_allclose(
            to_numpy(_require_grad(x)),
            expected_grad,
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
