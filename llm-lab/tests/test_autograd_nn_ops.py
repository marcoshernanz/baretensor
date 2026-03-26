import unittest
from collections.abc import Callable

import numpy as np

import bt
import bt.nn.functional as F
from tests.utils import to_numpy


def _require_grad(tensor: bt.Tensor) -> bt.Tensor:
    grad = tensor.grad
    assert grad is not None
    return grad


def _stable_softmax_expected(source: np.ndarray, dim: int) -> np.ndarray:
    shifted = source - np.max(source, axis=dim, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=dim, keepdims=True)


def _layer_norm_expected(
    input_value: np.ndarray,
    normalized_shape: tuple[int, ...],
    *,
    weight: np.ndarray | None,
    bias: np.ndarray | None,
    eps: float,
) -> np.ndarray:
    axis = tuple(range(input_value.ndim - len(normalized_shape), input_value.ndim))
    mean = np.mean(input_value, axis=axis, keepdims=True)
    variance = np.mean((input_value - mean) ** 2, axis=axis, keepdims=True)
    output = (input_value - mean) / np.sqrt(variance + eps)

    if weight is not None:
        affine_shape = (1,) * (input_value.ndim - len(normalized_shape)) + normalized_shape
        output = output * weight.reshape(affine_shape)
    if bias is not None:
        affine_shape = (1,) * (input_value.ndim - len(normalized_shape)) + normalized_shape
        output = output + bias.reshape(affine_shape)

    return output.astype(np.float32)


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


class AutogradNnOpsTests(unittest.TestCase):
    def test_softmax_backward_matches_closed_form(self) -> None:
        x_np = np.asarray([[0.2, -1.0, 3.0], [2.5, -0.3, 0.7]], dtype=np.float32)
        g_np = np.asarray([[1.0, -2.0, 0.5], [-1.5, 3.0, 2.0]], dtype=np.float32)

        x = bt.tensor(x_np, requires_grad=True)
        g = bt.tensor(g_np)

        loss = (x.softmax(1) * g).sum()
        loss.backward()

        probs = _stable_softmax_expected(x_np, 1)
        expected = probs * (g_np - np.sum(g_np * probs, axis=1, keepdims=True))

        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=1e-6, atol=1e-6)

    def test_log_softmax_backward_matches_closed_form(self) -> None:
        x_np = np.asarray([[1.2, -0.2, 2.1], [0.0, 0.5, -1.5]], dtype=np.float32)
        g_np = np.asarray([[2.0, -1.0, 0.25], [1.5, -0.7, 3.0]], dtype=np.float32)

        x = bt.tensor(x_np, requires_grad=True)
        g = bt.tensor(g_np)

        loss = (x.log_softmax(1) * g).sum()
        loss.backward()

        probs = _stable_softmax_expected(x_np, 1)
        expected = g_np - probs * np.sum(g_np, axis=1, keepdims=True)

        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=1e-6, atol=1e-6)

    def test_cross_entropy_backward_mean_with_ignore_index(self) -> None:
        logits_np = np.asarray([[1.0, -0.5, 2.0], [0.2, 0.3, -1.0]], dtype=np.float32)
        target_np = np.asarray([2, -100], dtype=np.int64)

        logits = bt.tensor(logits_np, requires_grad=True)
        target = bt.tensor(target_np)

        loss = F.cross_entropy(logits, target, ignore_index=-100, reduction="mean")
        loss.backward()

        probs = _stable_softmax_expected(logits_np, 1)
        expected = np.zeros_like(logits_np, dtype=np.float32)
        expected[0] = probs[0]
        expected[0, 2] -= 1.0

        np.testing.assert_allclose(to_numpy(_require_grad(logits)), expected, rtol=1e-6, atol=1e-6)

    def test_cross_entropy_backward_none_with_explicit_gradient(self) -> None:
        logits_np = np.asarray([[0.4, -0.7, 1.8], [1.3, -2.0, 0.6]], dtype=np.float32)
        target_np = np.asarray([1, 0], dtype=np.int64)
        out_grad_np = np.asarray([2.0, -0.5], dtype=np.float32)

        logits = bt.tensor(logits_np, requires_grad=True)
        target = bt.tensor(target_np)

        losses = F.cross_entropy(logits, target, reduction="none")
        losses.backward(bt.tensor(out_grad_np))

        probs = _stable_softmax_expected(logits_np, 1)
        expected = np.zeros_like(logits_np, dtype=np.float32)
        expected[0] = out_grad_np[0] * probs[0]
        expected[0, 1] -= out_grad_np[0]
        expected[1] = out_grad_np[1] * probs[1]
        expected[1, 0] -= out_grad_np[1]

        np.testing.assert_allclose(to_numpy(_require_grad(logits)), expected, rtol=1e-6, atol=1e-6)

    def test_layer_norm_backward_matches_finite_difference(self) -> None:
        x_np = np.asarray([[0.5, -1.0, 2.0], [1.5, 0.2, -0.3]], dtype=np.float32)
        w_np = np.asarray([0.7, -1.2, 0.9], dtype=np.float32)
        b_np = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
        g_np = np.asarray([[2.0, -0.5, 1.2], [0.7, -1.5, 0.4]], dtype=np.float32)
        eps = 1e-5

        x = bt.tensor(x_np, requires_grad=True)
        w = bt.tensor(w_np, requires_grad=True)
        b = bt.tensor(b_np, requires_grad=True)
        g = bt.tensor(g_np)

        loss = (F.layer_norm(x, normalized_shape=(3,), weight=w, bias=b, eps=eps) * g).sum()
        loss.backward()

        def loss_for_x(x_value: np.ndarray) -> np.float32:
            out = _layer_norm_expected(x_value, (3,), weight=w_np, bias=b_np, eps=eps)
            return np.float32(np.sum(out * g_np, dtype=np.float32))

        def loss_for_w(w_value: np.ndarray) -> np.float32:
            out = _layer_norm_expected(x_np, (3,), weight=w_value, bias=b_np, eps=eps)
            return np.float32(np.sum(out * g_np, dtype=np.float32))

        def loss_for_b(b_value: np.ndarray) -> np.float32:
            out = _layer_norm_expected(x_np, (3,), weight=w_np, bias=b_value, eps=eps)
            return np.float32(np.sum(out * g_np, dtype=np.float32))

        expected_x = _numerical_grad(x_np, loss_for_x)
        expected_w = _numerical_grad(w_np, loss_for_w)
        expected_b = _numerical_grad(b_np, loss_for_b)

        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected_x, rtol=2e-2, atol=2e-2)
        np.testing.assert_allclose(to_numpy(_require_grad(w)), expected_w, rtol=2e-2, atol=2e-2)
        np.testing.assert_allclose(to_numpy(_require_grad(b)), expected_b, rtol=2e-2, atol=2e-2)

    def test_embedding_backward_scatter_add(self) -> None:
        indices_np = np.asarray([[0, 2, 0], [1, 2, 1]], dtype=np.int64)
        weight_np = np.asarray(
            [[1.0, 0.0, -1.0], [2.0, -2.0, 1.0], [0.5, 1.5, -0.5], [3.0, -1.0, 2.0]],
            dtype=np.float32,
        )
        out_grad_np = np.asarray(
            [
                [[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0], [0.5, 0.5, 0.5]],
                [[2.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-2.0, 0.5, -0.5]],
            ],
            dtype=np.float32,
        )

        indices = bt.tensor(indices_np)
        weight = bt.tensor(weight_np, requires_grad=True)
        out_grad = bt.tensor(out_grad_np)

        loss = (F.embedding(indices, weight) * out_grad).sum()
        loss.backward()

        expected_weight_grad = np.zeros_like(weight_np, dtype=np.float32)
        for index in np.ndindex(*indices_np.shape):
            row = int(indices_np[index])
            expected_weight_grad[row] += out_grad_np[index]

        np.testing.assert_allclose(
            to_numpy(_require_grad(weight)),
            expected_weight_grad,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_embedding_indices_gradient_is_zero(self) -> None:
        indices_np = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
        weight_np = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

        indices = bt.tensor(indices_np)
        weight = bt.tensor(weight_np, requires_grad=True)

        F.embedding(indices, weight).sum().backward()

        self.assertIsNone(indices.grad)


if __name__ == "__main__":
    unittest.main()
