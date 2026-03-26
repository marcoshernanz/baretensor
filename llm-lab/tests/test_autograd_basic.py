import unittest

import numpy as np

import bt
from tests.utils import to_numpy


def _require_grad(tensor: bt.Tensor) -> bt.Tensor:
    grad = tensor.grad
    assert grad is not None
    return grad


class AutogradBasicTests(unittest.TestCase):
    def test_backward_scalar_chain_add_mul_sum(self) -> None:
        x = bt.tensor(np.asarray([2.0, -3.0], dtype=np.float32), requires_grad=True)
        y = bt.tensor(np.asarray([4.0, 5.0], dtype=np.float32), requires_grad=True)

        loss = ((x * y) + x).sum()
        loss.backward()

        np.testing.assert_allclose(
            to_numpy(_require_grad(x)),
            np.asarray([5.0, 6.0], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            to_numpy(_require_grad(y)),
            np.asarray([2.0, -3.0], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_branch_accumulation_single_leaf(self) -> None:
        x = bt.tensor(np.asarray([1.5, -2.0, 0.25], dtype=np.float32), requires_grad=True)

        loss = (x * x + x).sum()
        loss.backward()

        expected = 2.0 * np.asarray([1.5, -2.0, 0.25], dtype=np.float32) + 1.0
        np.testing.assert_allclose(to_numpy(_require_grad(x)), expected, rtol=1e-6, atol=1e-6)

    def test_broadcast_gradient_reduction(self) -> None:
        a = bt.tensor(np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32), requires_grad=True)
        b = bt.tensor(np.asarray([[10.0, 20.0, 30.0]], dtype=np.float32), requires_grad=True)

        loss = (a + b).sum()
        loss.backward()

        np.testing.assert_allclose(
            to_numpy(_require_grad(a)),
            np.ones((2, 3), dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            to_numpy(_require_grad(b)),
            np.asarray([[2.0, 2.0, 2.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_backward_non_scalar_requires_explicit_gradient(self) -> None:
        x = bt.tensor(np.asarray([1.0, 2.0], dtype=np.float32), requires_grad=True)
        y = x * 3.0

        with self.assertRaisesRegex(ValueError, r"requires an explicit gradient"):
            y.backward()

    def test_backward_non_scalar_with_explicit_gradient(self) -> None:
        x = bt.tensor(np.asarray([1.0, 2.0], dtype=np.float32), requires_grad=True)
        y = x * 3.0
        grad = bt.tensor(np.asarray([10.0, 20.0], dtype=np.float32))

        y.backward(grad)

        np.testing.assert_allclose(
            to_numpy(_require_grad(x)),
            np.asarray([30.0, 60.0], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_zero_grad_clears_accumulated_gradient(self) -> None:
        x = bt.tensor(np.asarray([2.0, 3.0], dtype=np.float32), requires_grad=True)
        y = (x + 1.0).sum()
        y.backward()

        self.assertIsNotNone(x.grad)
        x.zero_grad()
        self.assertIsNone(x.grad)

    def test_backward_rejects_gradient_shape_mismatch(self) -> None:
        x = bt.tensor(np.asarray([1.0, 2.0], dtype=np.float32), requires_grad=True)
        y = x * 3.0

        with self.assertRaisesRegex(ValueError, r"gradient shape mismatch"):
            y.backward(bt.tensor(np.asarray([1.0], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
