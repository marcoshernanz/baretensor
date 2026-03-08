import unittest

import numpy as np

import bt
from tests.utils import to_numpy


def _require_grad(tensor: bt.Tensor) -> bt.Tensor:
    grad = tensor.grad
    assert grad is not None
    return grad


class AutogradViewAndShapeTests(unittest.TestCase):
    def test_view_backward_maps_gradients_to_original_shape(self) -> None:
        x = bt.tensor(np.arange(12, dtype=np.float32).reshape(3, 4), requires_grad=True)

        y = x.view([2, 6])
        weight_np = np.linspace(-1.0, 1.0, num=12, dtype=np.float32).reshape(2, 6)
        weight = bt.tensor(weight_np)

        loss = (y * weight).sum()
        loss.backward()

        expected = weight_np.reshape(3, 4)
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=1e-6, atol=1e-6)

    def test_reshape_backward_from_non_contiguous_input(self) -> None:
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        x = bt.tensor(x_np, requires_grad=True)

        y = x.permute([1, 0, 2]).reshape([3, 8])
        weight_np = np.arange(24, dtype=np.float32).reshape(3, 8)
        weight = bt.tensor(weight_np)

        loss = (y * weight).sum()
        loss.backward()

        expected = np.transpose(weight_np.reshape(3, 2, 4), (1, 0, 2))
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=1e-6, atol=1e-6)

    def test_flatten_backward_maps_gradients_to_original_shape(self) -> None:
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        x = bt.tensor(x_np, requires_grad=True)

        y = x.flatten(1)
        weight_np = np.arange(24, dtype=np.float32).reshape(2, 12)
        weight = bt.tensor(weight_np)

        loss = (y * weight).sum()
        loss.backward()

        expected = weight_np.reshape(2, 3, 4)
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=1e-6, atol=1e-6)

    def test_permute_backward_uses_inverse_permutation(self) -> None:
        x = bt.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4), requires_grad=True)

        y = x.permute([2, 0, 1])
        weight_np = np.arange(1, 25, dtype=np.float32).reshape(4, 2, 3)
        weight = bt.tensor(weight_np)

        loss = (y * weight).sum()
        loss.backward()

        expected = np.transpose(weight_np, (1, 2, 0))
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=1e-6, atol=1e-6)

    def test_transpose_backward_swaps_dimensions_back(self) -> None:
        x = bt.tensor(np.arange(10, dtype=np.float32).reshape(2, 5), requires_grad=True)

        y = x.transpose(0, 1)
        weight_np = np.arange(1, 11, dtype=np.float32).reshape(5, 2)
        weight = bt.tensor(weight_np)

        loss = (y * weight).sum()
        loss.backward()

        expected = weight_np.T
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=1e-6, atol=1e-6)

    def test_contiguous_backward_passes_gradients_to_source(self) -> None:
        x = bt.tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)

        y = x.transpose(0, 1).contiguous()
        weight_np = np.arange(1, 7, dtype=np.float32).reshape(3, 2)
        weight = bt.tensor(weight_np)

        loss = (y * weight).sum()
        loss.backward()

        expected = weight_np.T
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=1e-6, atol=1e-6)

    def test_contiguous_on_already_contiguous_tensor_keeps_grad_flow(self) -> None:
        x = bt.tensor(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)

        y = x.contiguous()
        self.assertTrue(y.requires_grad)

        loss = (y * 2.0).sum()
        loss.backward()

        expected = np.full((2, 2), 2.0, dtype=np.float32)
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
