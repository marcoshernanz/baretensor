import unittest

import numpy as np

import bt
from tests.utils import to_numpy


def _require_grad(tensor: bt.Tensor) -> bt.Tensor:
    grad = tensor.grad
    assert grad is not None
    return grad


def _numerical_grad_wrt_a(
    a_np: np.ndarray,
    b_np: np.ndarray,
    g_np: np.ndarray,
    *,
    eps: float = 1e-3,
) -> np.ndarray:
    grad = np.zeros_like(a_np, dtype=np.float32)
    for index in np.ndindex(*a_np.shape):
        a_pos = a_np.copy()
        a_neg = a_np.copy()
        a_pos[index] += eps
        a_neg[index] -= eps
        loss_pos = np.sum(np.matmul(a_pos, b_np) * g_np, dtype=np.float32)
        loss_neg = np.sum(np.matmul(a_neg, b_np) * g_np, dtype=np.float32)
        grad[index] = (loss_pos - loss_neg) / (2.0 * eps)
    return grad


def _numerical_grad_wrt_b(
    a_np: np.ndarray,
    b_np: np.ndarray,
    g_np: np.ndarray,
    *,
    eps: float = 1e-3,
) -> np.ndarray:
    grad = np.zeros_like(b_np, dtype=np.float32)
    for index in np.ndindex(*b_np.shape):
        b_pos = b_np.copy()
        b_neg = b_np.copy()
        b_pos[index] += eps
        b_neg[index] -= eps
        loss_pos = np.sum(np.matmul(a_np, b_pos) * g_np, dtype=np.float32)
        loss_neg = np.sum(np.matmul(a_np, b_neg) * g_np, dtype=np.float32)
        grad[index] = (loss_pos - loss_neg) / (2.0 * eps)
    return grad


class AutogradMatmulTests(unittest.TestCase):
    def test_matrix_matrix_backward_matches_numpy(self) -> None:
        a_np = np.arange(6, dtype=np.float32).reshape(2, 3)
        b_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        g_np = np.arange(1, 9, dtype=np.float32).reshape(2, 4)

        a = bt.tensor(a_np, requires_grad=True)
        b = bt.tensor(b_np, requires_grad=True)
        g = bt.tensor(g_np)

        loss = (a.matmul(b) * g).sum()
        loss.backward()

        expected_a = g_np @ b_np.T
        expected_b = a_np.T @ g_np

        np.testing.assert_allclose(to_numpy(_require_grad(a)), expected_a, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(to_numpy(_require_grad(b)), expected_b, rtol=1e-6, atol=1e-6)

    def test_batched_broadcast_backward_matches_numpy(self) -> None:
        a_np = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
        b_np = np.arange(60, dtype=np.float32).reshape(4, 3, 5)
        g_np = np.arange(40, dtype=np.float32).reshape(4, 2, 5)

        a = bt.tensor(a_np, requires_grad=True)
        b = bt.tensor(b_np, requires_grad=True)
        g = bt.tensor(g_np)

        loss = (a.matmul(b) * g).sum()
        loss.backward()

        grad_a_full = np.matmul(g_np, np.swapaxes(b_np, -1, -2))
        expected_a = grad_a_full.sum(axis=0, keepdims=True)
        expected_b = np.matmul(np.swapaxes(a_np, -1, -2), g_np)

        np.testing.assert_allclose(to_numpy(_require_grad(a)), expected_a, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(to_numpy(_require_grad(b)), expected_b, rtol=1e-6, atol=1e-6)

    def test_vector_matrix_backward_matches_numpy(self) -> None:
        x_np = np.asarray([1.0, -2.0, 3.0], dtype=np.float32)
        w_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        g_np = np.asarray([0.5, -1.0, 2.0, 3.0], dtype=np.float32)

        x = bt.tensor(x_np, requires_grad=True)
        w = bt.tensor(w_np, requires_grad=True)
        g = bt.tensor(g_np)

        loss = (x.matmul(w) * g).sum()
        loss.backward()

        expected_x = g_np @ w_np.T
        expected_w = np.outer(x_np, g_np)

        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected_x, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(to_numpy(_require_grad(w)), expected_w, rtol=1e-6, atol=1e-6)

    def test_matrix_vector_backward_matches_numpy(self) -> None:
        x_np = np.arange(6, dtype=np.float32).reshape(2, 3)
        v_np = np.asarray([2.0, -1.0, 4.0], dtype=np.float32)
        g_np = np.asarray([3.0, -5.0], dtype=np.float32)

        x = bt.tensor(x_np, requires_grad=True)
        v = bt.tensor(v_np, requires_grad=True)
        g = bt.tensor(g_np)

        loss = (x.matmul(v) * g).sum()
        loss.backward()

        expected_x = np.outer(g_np, v_np)
        expected_v = x_np.T @ g_np

        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected_x, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(to_numpy(_require_grad(v)), expected_v, rtol=1e-6, atol=1e-6)

    def test_vector_vector_backward_with_explicit_scalar_gradient(self) -> None:
        x_np = np.asarray([1.0, 2.0, -3.0], dtype=np.float32)
        y_np = np.asarray([4.0, -5.0, 6.0], dtype=np.float32)

        x = bt.tensor(x_np, requires_grad=True)
        y = bt.tensor(y_np, requires_grad=True)

        out = x.matmul(y)
        out.backward(bt.tensor(np.asarray(np.float32(2.5))))

        np.testing.assert_allclose(to_numpy(_require_grad(x)), 2.5 * y_np, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(to_numpy(_require_grad(y)), 2.5 * x_np, rtol=1e-6, atol=1e-6)

    def test_non_scalar_matmul_output_requires_explicit_gradient(self) -> None:
        a = bt.tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)
        b = bt.tensor(np.arange(12, dtype=np.float32).reshape(3, 4), requires_grad=True)

        out = a.matmul(b)
        with self.assertRaisesRegex(ValueError, r"requires an explicit gradient"):
            out.backward()

    def test_matrix_matrix_gradcheck_with_finite_difference(self) -> None:
        a_np = np.asarray([[0.2, -0.7], [1.3, 0.5]], dtype=np.float32)
        b_np = np.asarray([[1.1, -2.0], [0.3, 0.9]], dtype=np.float32)
        g_np = np.asarray([[0.4, -1.2], [2.5, 0.8]], dtype=np.float32)

        a = bt.tensor(a_np, requires_grad=True)
        b = bt.tensor(b_np, requires_grad=True)
        g = bt.tensor(g_np)

        loss = (a.matmul(b) * g).sum()
        loss.backward()

        numerical_a = _numerical_grad_wrt_a(a_np, b_np, g_np)
        numerical_b = _numerical_grad_wrt_b(a_np, b_np, g_np)

        np.testing.assert_allclose(
            to_numpy(_require_grad(a)),
            numerical_a,
            rtol=5e-3,
            atol=5e-3,
        )
        np.testing.assert_allclose(
            to_numpy(_require_grad(b)),
            numerical_b,
            rtol=5e-3,
            atol=5e-3,
        )


if __name__ == "__main__":
    unittest.main()
