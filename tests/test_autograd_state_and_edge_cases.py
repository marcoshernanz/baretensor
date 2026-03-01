import unittest

import numpy as np

import bt
from tests.utils import to_numpy


def _require_grad(tensor: bt.Tensor) -> bt.Tensor:
    grad = tensor.grad
    assert grad is not None
    return grad


class AutogradStateAndEdgeCaseTests(unittest.TestCase):
    def test_requires_grad_property_assignment_toggles_tracking(self) -> None:
        x = bt.tensor(np.asarray([1.0, -2.0, 3.0], dtype=np.float32))
        self.assertFalse(x.requires_grad)

        x.requires_grad = True
        self.assertTrue(x.requires_grad)

        (x * 2.0).sum().backward()
        np.testing.assert_allclose(
            to_numpy(_require_grad(x)),
            np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_set_requires_grad_method_toggles_tracking(self) -> None:
        x = bt.tensor(np.asarray([0.5, -1.0], dtype=np.float32))
        self.assertFalse(x.requires_grad)

        returned = x.set_requires_grad(True)
        self.assertTrue(returned.requires_grad)
        self.assertTrue(x.requires_grad)

        x.set_requires_grad(False)
        self.assertFalse(x.requires_grad)

    def test_is_leaf_transitions_for_leaf_nonleaf_and_detach(self) -> None:
        x = bt.tensor(np.asarray([1.0, -2.0, 3.0], dtype=np.float32), requires_grad=True)
        y = x * 2.0
        z = y.detach()

        self.assertTrue(x.is_leaf)
        self.assertFalse(y.is_leaf)
        self.assertFalse(z.requires_grad)
        self.assertTrue(z.is_leaf)

    def test_requires_grad_false_blocks_history_for_existing_nonleaf(self) -> None:
        x = bt.tensor(np.asarray([1.0, 2.0], dtype=np.float32), requires_grad=True)
        y = x * 3.0
        y.set_requires_grad(False)

        self.assertFalse(y.requires_grad)
        self.assertTrue(y.is_leaf)

        with self.assertRaisesRegex(ValueError, r"does not require gradients"):
            y.backward(bt.tensor(np.asarray([1.0, 1.0], dtype=np.float32)))

        self.assertIsNone(x.grad)

    def test_repeated_backward_accumulates_gradients(self) -> None:
        x = bt.tensor(np.asarray([0.5, -1.5, 2.0], dtype=np.float32), requires_grad=True)
        loss = (x * 3.0).sum()

        loss.backward()
        loss.backward()

        expected = np.asarray([6.0, 6.0, 6.0], dtype=np.float32)
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=1e-6, atol=1e-6)

    def test_multiple_views_of_same_base_accumulate_gradients(self) -> None:
        x = bt.tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)
        v1 = x.view([3, 2])
        v2 = x.transpose(0, 1)

        w1_np = np.asarray([[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0]], dtype=np.float32)
        w2_np = np.asarray([[0.5, -1.0], [2.5, 3.0], [-4.0, 1.5]], dtype=np.float32)
        w1 = bt.tensor(w1_np)
        w2 = bt.tensor(w2_np)

        loss = (v1 * w1).sum() + (v2 * w2).sum()
        loss.backward()

        expected = w1_np.reshape(2, 3) + w2_np.T
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=1e-6, atol=1e-6)

    def test_mixed_view_and_non_view_branches_accumulate_gradients(self) -> None:
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        x = bt.tensor(x_np, requires_grad=True)

        view_branch = x.permute([1, 0])
        dense_branch = x * 2.0

        w_view_np = np.linspace(-1.0, 1.0, num=12, dtype=np.float32).reshape(4, 3)
        w_dense_np = np.linspace(0.2, 2.4, num=12, dtype=np.float32).reshape(3, 4)
        w_view = bt.tensor(w_view_np)
        w_dense = bt.tensor(w_dense_np)

        loss = (view_branch * w_view).sum() + (dense_branch * w_dense).sum()
        loss.backward()

        expected = w_view_np.T + (2.0 * w_dense_np)
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=1e-6, atol=1e-6)

    def test_backward_on_zero_size_composite_graph_produces_zero_grad(self) -> None:
        x = bt.tensor(np.zeros((0, 3), dtype=np.float32), requires_grad=True)
        loss = ((x + 2.0).exp() * 5.0).sum()
        loss.backward()

        np.testing.assert_allclose(
            to_numpy(_require_grad(x)),
            np.zeros((0, 3), dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_non_scalar_zero_size_backward_accepts_explicit_gradient(self) -> None:
        x = bt.tensor(np.zeros((0, 4), dtype=np.float32), requires_grad=True)
        y = (x * 7.0).sum(1)

        y.backward(bt.tensor(np.zeros((0,), dtype=np.float32)))

        np.testing.assert_allclose(
            to_numpy(_require_grad(x)),
            np.zeros((0, 4), dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_composed_non_scalar_backward_rejects_bad_gradient_shape(self) -> None:
        x = bt.tensor(np.asarray([[1.0, 2.0, 3.0], [0.5, -1.0, 0.3]], dtype=np.float32), requires_grad=True)
        y = (x.softmax(1) * 2.0).sum(1)

        with self.assertRaisesRegex(ValueError, r"gradient shape mismatch"):
            y.backward(bt.tensor(np.asarray([1.0], dtype=np.float32)))

    def test_detached_computation_does_not_require_grad(self) -> None:
        x = bt.tensor(np.asarray([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
        y = x.detach()
        loss = (y * 4.0).sum()

        with self.assertRaisesRegex(ValueError, r"does not require gradients"):
            loss.backward()

        self.assertIsNone(x.grad)


if __name__ == "__main__":
    unittest.main()
