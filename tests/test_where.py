import unittest

import numpy as np

import bt
from tests.utils import to_numpy


def _require_grad(tensor: bt.Tensor) -> bt.Tensor:
    grad = tensor.grad
    assert grad is not None
    return grad


class WhereTests(unittest.TestCase):
    def test_same_shape_float32_matches_numpy(self) -> None:
        condition_np = np.asarray([[True, False], [False, True]], dtype=np.bool_)
        input_np = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        other_np = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)

        out = bt.where(bt.tensor(condition_np), bt.tensor(input_np), bt.tensor(other_np))

        np.testing.assert_allclose(
            to_numpy(out),
            np.where(condition_np, input_np, other_np).astype(np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_broadcasts_all_three_operands(self) -> None:
        condition_np = np.asarray([[True], [False]], dtype=np.bool_)
        input_np = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        other_np = np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)

        out = bt.where(bt.tensor(condition_np), bt.tensor(input_np), bt.tensor(other_np))

        np.testing.assert_allclose(
            to_numpy(out),
            np.where(condition_np, input_np, other_np).astype(np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_scalar_branches_are_supported(self) -> None:
        condition = bt.tensor(np.asarray([[True, False], [False, True]], dtype=np.bool_))

        out = bt.where(condition, 1.0, 0.0)

        self.assertEqual(out.dtype, bt.float32)
        np.testing.assert_allclose(
            to_numpy(out),
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_int64_and_bool_branches_are_supported(self) -> None:
        condition = bt.tensor(np.asarray([True, False, True], dtype=np.bool_))
        int_out = bt.where(condition, bt.tensor([1, 2, 3]), bt.tensor([4, 5, 6]))
        bool_out = bt.where(
            condition,
            bt.tensor(np.asarray([True, False, True], dtype=np.bool_)),
            bt.tensor(np.asarray([False, True, False], dtype=np.bool_)),
        )

        self.assertEqual(int_out.dtype, bt.int64)
        self.assertEqual(bool_out.dtype, bt.bool)
        np.testing.assert_array_equal(to_numpy(int_out), np.asarray([1, 5, 3], dtype=np.int64))
        np.testing.assert_array_equal(
            to_numpy(bool_out), np.asarray([True, True, True], dtype=np.bool_)
        )

    def test_zero_size_outputs_are_supported(self) -> None:
        condition = bt.tensor(np.zeros((0, 3), dtype=np.bool_))
        input = bt.tensor(np.zeros((0, 3), dtype=np.float32))
        other = bt.tensor(np.ones((1, 3), dtype=np.float32))

        out = bt.where(condition, input, other)

        self.assertEqual(out.shape, [0, 3])
        np.testing.assert_allclose(to_numpy(out), np.zeros((0, 3), dtype=np.float32))

    def test_non_bool_condition_rejects(self) -> None:
        with self.assertRaisesRegex(TypeError, r"condition to have dtype bt.bool"):
            _ = bt.where(bt.tensor([1, 0]), bt.tensor([1.0, 2.0]), bt.tensor([3.0, 4.0]))

    def test_mixed_branch_dtypes_reject(self) -> None:
        condition = bt.tensor(np.asarray([True, False], dtype=np.bool_))

        with self.assertRaisesRegex(ValueError, r"same dtype"):
            _ = bt.where(condition, bt.tensor([1.0, 2.0]), bt.tensor([1, 2]))

    def test_mixed_scalar_kinds_reject_without_promotion(self) -> None:
        condition = bt.tensor(np.asarray([True, False], dtype=np.bool_))

        with self.assertRaisesRegex(ValueError, r"same dtype"):
            _ = bt.where(condition, 1, 0.0)

    def test_non_broadcastable_shapes_reject(self) -> None:
        condition = bt.tensor(np.asarray([[True, False], [False, True]], dtype=np.bool_))
        input = bt.tensor(np.zeros((2, 3), dtype=np.float32))
        other = bt.tensor(np.zeros((2, 3), dtype=np.float32))

        with self.assertRaisesRegex(ValueError, r"Cannot broadcast shapes"):
            _ = bt.where(condition, input, other)

    def test_backward_splits_gradients_between_branches(self) -> None:
        condition = bt.tensor(np.asarray([True, False, True, False], dtype=np.bool_))
        input = bt.tensor(np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32), requires_grad=True)
        other = bt.tensor(np.asarray([5.0, 6.0, 7.0, 8.0], dtype=np.float32), requires_grad=True)

        loss = bt.where(condition, input, other).sum()
        loss.backward()

        np.testing.assert_allclose(
            to_numpy(_require_grad(input)),
            np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            to_numpy(_require_grad(other)),
            np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_backward_reduces_broadcasted_gradients(self) -> None:
        condition = bt.tensor(np.asarray([[True, False, True], [False, True, False]], dtype=np.bool_))
        input = bt.tensor(np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32), requires_grad=True)
        other = bt.tensor(np.asarray([[10.0], [20.0]], dtype=np.float32), requires_grad=True)

        loss = bt.where(condition, input, other).sum()
        loss.backward()

        np.testing.assert_allclose(
            to_numpy(_require_grad(input)),
            np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            to_numpy(_require_grad(other)),
            np.asarray([[1.0], [2.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_backward_with_scalar_other_masks_inactive_positions(self) -> None:
        input = bt.tensor(np.asarray([-2.0, -1.0, 1.0, 3.0], dtype=np.float32), requires_grad=True)

        loss = bt.where(input > 0.0, input, 0.0).sum()
        loss.backward()

        np.testing.assert_allclose(
            to_numpy(_require_grad(input)),
            np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
