import unittest
from collections.abc import Callable

import numpy as np

import bt
from tests.utils import to_numpy


def _require_grad(tensor: bt.Tensor) -> bt.Tensor:
    grad = tensor.grad
    assert grad is not None
    return grad


def _numerical_grad(
    value: np.ndarray,
    loss_fn: Callable[[np.ndarray], np.float32],
    *,
    eps: float = 1e-3,
) -> np.ndarray:
    grad = np.zeros_like(value, dtype=np.float32)
    for index in np.ndindex(*value.shape):
        value_pos = value.copy()
        value_neg = value.copy()
        value_pos[index] += eps
        value_neg[index] -= eps
        loss_pos = loss_fn(value_pos)
        loss_neg = loss_fn(value_neg)
        grad[index] = (loss_pos - loss_neg) / (2.0 * eps)
    return grad


class AutogradExtraGradcheckTests(unittest.TestCase):
    def test_sum_gradcheck_finite_difference(self) -> None:
        x_np = np.asarray([[0.1, -0.4, 0.9], [1.2, -0.7, 0.3]], dtype=np.float32)
        g_np = np.asarray([1.5, -2.0, 0.8], dtype=np.float32)

        x = bt.tensor(x_np, requires_grad=True)
        g = bt.tensor(g_np)
        loss = (x.sum(0) * g).sum()
        loss.backward()

        def loss_for_x(x_value: np.ndarray) -> np.float32:
            reduced = np.sum(x_value, axis=0, dtype=np.float32)
            return np.float32(np.sum(reduced * g_np, dtype=np.float32))

        expected = _numerical_grad(x_np, loss_for_x)
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=5e-3, atol=5e-3)

    def test_mean_gradcheck_finite_difference(self) -> None:
        x_np = np.asarray(
            [[[0.2, -0.7], [1.1, 0.4], [0.3, -0.2]], [[-1.3, 0.8], [0.5, 1.7], [0.9, -0.6]]],
            dtype=np.float32,
        )
        g_np = np.asarray([0.5, -1.2, 2.0], dtype=np.float32)

        x = bt.tensor(x_np, requires_grad=True)
        g = bt.tensor(g_np)
        loss = (x.mean([0, 2]) * g).sum()
        loss.backward()

        def loss_for_x(x_value: np.ndarray) -> np.float32:
            reduced = np.mean(x_value, axis=(0, 2), dtype=np.float32)
            return np.float32(np.sum(reduced * g_np, dtype=np.float32))

        expected = _numerical_grad(x_np, loss_for_x)
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=5e-3, atol=5e-3)

    def test_max_gradcheck_finite_difference_without_ties(self) -> None:
        x_np = np.asarray([[0.1, -2.0, 3.5], [4.2, 0.3, -1.8]], dtype=np.float32)
        g_np = np.asarray([1.75, -0.5], dtype=np.float32)

        x = bt.tensor(x_np, requires_grad=True)
        g = bt.tensor(g_np)
        loss = (x.max(1) * g).sum()
        loss.backward()

        def loss_for_x(x_value: np.ndarray) -> np.float32:
            reduced = np.max(x_value, axis=1)
            return np.float32(np.sum(reduced * g_np, dtype=np.float32))

        expected = _numerical_grad(x_np, loss_for_x)
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=5e-3, atol=5e-3)

    def test_softmax_gradcheck_finite_difference(self) -> None:
        x_np = np.asarray([[0.4, -1.0, 2.0], [1.5, 0.2, -0.3]], dtype=np.float32)
        g_np = np.asarray([[1.0, -0.2, 0.7], [0.5, -1.5, 2.0]], dtype=np.float32)

        x = bt.tensor(x_np, requires_grad=True)
        g = bt.tensor(g_np)
        loss = (x.softmax(1) * g).sum()
        loss.backward()

        def loss_for_x(x_value: np.ndarray) -> np.float32:
            shifted = x_value - np.max(x_value, axis=1, keepdims=True)
            probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
            return np.float32(np.sum(probs * g_np, dtype=np.float32))

        expected = _numerical_grad(x_np, loss_for_x)
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=7e-3, atol=7e-3)

    def test_log_softmax_gradcheck_finite_difference(self) -> None:
        x_np = np.asarray([[0.7, -0.4, 1.2], [-1.1, 0.3, 2.2]], dtype=np.float32)
        g_np = np.asarray([[0.5, -1.0, 0.8], [1.3, -0.6, 0.2]], dtype=np.float32)

        x = bt.tensor(x_np, requires_grad=True)
        g = bt.tensor(g_np)
        loss = (x.log_softmax(1) * g).sum()
        loss.backward()

        def loss_for_x(x_value: np.ndarray) -> np.float32:
            shifted = x_value - np.max(x_value, axis=1, keepdims=True)
            log_softmax = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
            return np.float32(np.sum(log_softmax * g_np, dtype=np.float32))

        expected = _numerical_grad(x_np, loss_for_x)
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=7e-3, atol=7e-3)

    def test_tanh_gradcheck_finite_difference(self) -> None:
        x_np = np.asarray([[0.7, -0.4, 1.2], [-1.1, 0.3, 2.2]], dtype=np.float32)
        g_np = np.asarray([[0.5, -1.0, 0.8], [1.3, -0.6, 0.2]], dtype=np.float32)

        x = bt.tensor(x_np, requires_grad=True)
        g = bt.tensor(g_np)
        loss = (x.tanh() * g).sum()
        loss.backward()

        def loss_for_x(x_value: np.ndarray) -> np.float32:
            return np.float32(np.sum(np.tanh(x_value) * g_np, dtype=np.float32))

        expected = _numerical_grad(x_np, loss_for_x)
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=5e-3, atol=5e-3)

    def test_elementwise_mul_gradcheck_with_broadcast(self) -> None:
        a_np = np.asarray([[0.5, -1.0, 2.0], [1.5, 0.3, -0.7]], dtype=np.float32)
        b_np = np.asarray([[0.2, 1.1, -0.4]], dtype=np.float32)
        g_np = np.asarray([[1.0, -2.0, 3.0], [0.5, -0.6, 0.7]], dtype=np.float32)

        a = bt.tensor(a_np, requires_grad=True)
        b = bt.tensor(b_np, requires_grad=True)
        g = bt.tensor(g_np)
        loss = ((a * b) * g).sum()
        loss.backward()

        def loss_for_a(a_value: np.ndarray) -> np.float32:
            return np.float32(np.sum((a_value * b_np) * g_np, dtype=np.float32))

        def loss_for_b(b_value: np.ndarray) -> np.float32:
            return np.float32(np.sum((a_np * b_value) * g_np, dtype=np.float32))

        expected_a = _numerical_grad(a_np, loss_for_a)
        expected_b = _numerical_grad(b_np, loss_for_b)
        np.testing.assert_allclose(to_numpy(_require_grad(a)), expected_a, rtol=5e-3, atol=5e-3)
        np.testing.assert_allclose(to_numpy(_require_grad(b)), expected_b, rtol=5e-3, atol=5e-3)

    def test_elementwise_div_gradcheck_with_broadcast(self) -> None:
        a_np = np.asarray([[0.8, -1.5, 2.0], [1.2, 0.4, -0.9]], dtype=np.float32)
        b_np = np.asarray([[1.7, -2.2, 0.6]], dtype=np.float32)
        g_np = np.asarray([[0.3, -1.1, 2.4], [1.5, 0.2, -0.8]], dtype=np.float32)

        a = bt.tensor(a_np, requires_grad=True)
        b = bt.tensor(b_np, requires_grad=True)
        g = bt.tensor(g_np)
        loss = ((a / b) * g).sum()
        loss.backward()

        def loss_for_a(a_value: np.ndarray) -> np.float32:
            return np.float32(np.sum((a_value / b_np) * g_np, dtype=np.float32))

        def loss_for_b(b_value: np.ndarray) -> np.float32:
            return np.float32(np.sum((a_np / b_value) * g_np, dtype=np.float32))

        expected_a = _numerical_grad(a_np, loss_for_a)
        expected_b = _numerical_grad(b_np, loss_for_b)
        np.testing.assert_allclose(to_numpy(_require_grad(a)), expected_a, rtol=8e-3, atol=8e-3)
        np.testing.assert_allclose(to_numpy(_require_grad(b)), expected_b, rtol=8e-3, atol=8e-3)


if __name__ == "__main__":
    unittest.main()
