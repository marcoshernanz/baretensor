import unittest

import numpy as np

import bt
from tests.utils import to_numpy


def _require_grad(tensor: bt.Tensor) -> bt.Tensor:
    grad = tensor.grad
    assert grad is not None
    return grad


class AutogradOpsTests(unittest.TestCase):
    def test_neg_backward_matches_closed_form(self) -> None:
        x_np = np.asarray([-1.0, 0.0, 2.0, 4.0], dtype=np.float32)
        x = bt.tensor(x_np, requires_grad=True)

        loss = (-x).sum()
        loss.backward()

        expected = np.full_like(x_np, -1.0, dtype=np.float32)
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=1e-6, atol=1e-6)

    def test_sub_backward_with_broadcast(self) -> None:
        a_np = np.arange(6, dtype=np.float32).reshape(2, 3)
        b_np = np.asarray([[10.0, 20.0, 30.0]], dtype=np.float32)

        a = bt.tensor(a_np, requires_grad=True)
        b = bt.tensor(b_np, requires_grad=True)

        loss = (a - b).sum()
        loss.backward()

        np.testing.assert_allclose(
            to_numpy(_require_grad(a)),
            np.ones((2, 3), dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            to_numpy(_require_grad(b)),
            np.asarray([[-2.0, -2.0, -2.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_div_backward_matches_closed_form(self) -> None:
        a_np = np.asarray([[2.0, 4.0], [8.0, 16.0]], dtype=np.float32)
        b_np = np.asarray([[1.0, 2.0], [4.0, 8.0]], dtype=np.float32)

        a = bt.tensor(a_np, requires_grad=True)
        b = bt.tensor(b_np, requires_grad=True)

        loss = (a / b).sum()
        loss.backward()

        expected_a = 1.0 / b_np
        expected_b = -a_np / (b_np * b_np)

        np.testing.assert_allclose(to_numpy(_require_grad(a)), expected_a, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(to_numpy(_require_grad(b)), expected_b, rtol=1e-6, atol=1e-6)

    def test_exp_backward_matches_closed_form(self) -> None:
        x_np = np.asarray([-1.0, 0.0, 0.5, 2.0], dtype=np.float32)
        x = bt.tensor(x_np, requires_grad=True)

        loss = x.exp().sum()
        loss.backward()

        expected = np.exp(x_np).astype(np.float32)
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=1e-6, atol=1e-6)

    def test_log_backward_matches_closed_form(self) -> None:
        x_np = np.asarray([0.5, 1.0, 2.0, 8.0], dtype=np.float32)
        x = bt.tensor(x_np, requires_grad=True)

        loss = x.log().sum()
        loss.backward()

        expected = (1.0 / x_np).astype(np.float32)
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=1e-6, atol=1e-6)

    def test_tanh_backward_matches_closed_form(self) -> None:
        x_np = np.asarray([-2.0, -0.5, 0.0, 1.5], dtype=np.float32)
        x = bt.tensor(x_np, requires_grad=True)

        loss = x.tanh().sum()
        loss.backward()

        expected = (1.0 - np.tanh(x_np) ** 2).astype(np.float32)
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=1e-6, atol=1e-6)

    def test_mean_backward_reduction_scales_gradients(self) -> None:
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        x = bt.tensor(x_np, requires_grad=True)
        weights_np = np.asarray([2.0, -1.0, 4.0], dtype=np.float32)
        weights = bt.tensor(weights_np)

        y = x.mean([0, 2], False)
        loss = (y * weights).sum()
        loss.backward()

        expected = np.broadcast_to((weights_np / 8.0)[None, :, None], (2, 3, 4)).astype(np.float32)
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=1e-6, atol=1e-6)

    def test_max_backward_without_ties(self) -> None:
        x_np = np.asarray([[1.0, 5.0, 3.0], [7.0, 4.0, 6.0]], dtype=np.float32)
        x = bt.tensor(x_np, requires_grad=True)
        weights_np = np.asarray([10.0, 20.0], dtype=np.float32)
        weights = bt.tensor(weights_np)

        y = x.max(1)
        loss = (y * weights).sum()
        loss.backward()

        expected = np.asarray([[0.0, 10.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=1e-6, atol=1e-6)

    def test_max_backward_splits_gradient_across_ties(self) -> None:
        x_np = np.asarray([[2.0, 2.0, 1.0], [3.0, 3.0, 3.0]], dtype=np.float32)
        x = bt.tensor(x_np, requires_grad=True)
        weights_np = np.asarray([6.0, 9.0], dtype=np.float32)
        weights = bt.tensor(weights_np)

        y = x.max(1)
        loss = (y * weights).sum()
        loss.backward()

        expected = np.asarray([[3.0, 3.0, 0.0], [3.0, 3.0, 3.0]], dtype=np.float32)
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
