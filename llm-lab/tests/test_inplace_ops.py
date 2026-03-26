import unittest

import numpy as np

import bt
from tests.utils import to_numpy


class InplaceOpsTests(unittest.TestCase):
    def test_inplace_sub_updates_original_reference_under_no_grad(self) -> None:
        x = bt.tensor(np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
        alias = x

        with bt.no_grad():
            alias -= 1.5

        np.testing.assert_allclose(
            to_numpy(x),
            np.asarray([-0.5, 0.5, 1.5], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_inplace_tensor_rhs_mutates_storage_under_no_grad(self) -> None:
        x = bt.tensor(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        rhs = bt.tensor(np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32))

        with bt.no_grad():
            x += rhs

        np.testing.assert_allclose(
            to_numpy(x),
            np.asarray([[11.0, 22.0], [33.0, 44.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_inplace_ops_require_no_grad_context(self) -> None:
        for name in ("add", "sub", "mul", "div"):
            with self.subTest(op=name):
                x = bt.tensor(np.asarray([2.0, 4.0], dtype=np.float32))
                rhs = bt.tensor(np.asarray([1.0, 2.0], dtype=np.float32))

                with self.assertRaisesRegex(RuntimeError, r"only supported inside bt\.no_grad\(\)"):
                    if name == "add":
                        x += rhs
                    elif name == "sub":
                        x -= rhs
                    elif name == "mul":
                        x *= rhs
                    else:
                        x /= rhs

    def test_inplace_ops_reject_shape_expanding_broadcast(self) -> None:
        x = bt.tensor(np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
        rhs = bt.tensor(np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))

        with bt.no_grad():
            with self.assertRaisesRegex(ValueError, r"cannot change tensor shape"):
                x += rhs

    def test_inplace_update_preserves_requires_grad_and_future_backward(self) -> None:
        x = bt.tensor(np.asarray([1.0, 2.0, 4.0], dtype=np.float32), requires_grad=True)

        with bt.no_grad():
            x /= 2.0

        self.assertTrue(x.requires_grad)

        loss = (x * 3.0).sum()
        loss.backward()

        np.testing.assert_allclose(
            to_numpy(x),
            np.asarray([0.5, 1.0, 2.0], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        grad = x.grad
        assert grad is not None
        np.testing.assert_allclose(
            to_numpy(grad),
            np.asarray([3.0, 3.0, 3.0], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
