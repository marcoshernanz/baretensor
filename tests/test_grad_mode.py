import unittest

import numpy as np

import bt


class GradModeTests(unittest.TestCase):
    def test_no_grad_disables_history_inside_context(self) -> None:
        x = bt.tensor(np.asarray([1.0, -2.0, 3.0], dtype=np.float32), requires_grad=True)

        with bt.no_grad():
            y = x * 2.0

        self.assertFalse(y.requires_grad)
        self.assertTrue(y.is_leaf)

    def test_no_grad_restores_grad_tracking_after_context(self) -> None:
        x = bt.tensor(np.asarray([1.0, -2.0, 3.0], dtype=np.float32), requires_grad=True)

        with bt.no_grad():
            _ = x + 1.0

        y = x + 1.0
        self.assertTrue(y.requires_grad)
        self.assertFalse(y.is_leaf)

    def test_no_grad_nested_contexts_restore_outer_state(self) -> None:
        x = bt.tensor(np.asarray([1.0, -2.0, 3.0], dtype=np.float32), requires_grad=True)

        with bt.no_grad():
            outer = x * 2.0
            with bt.no_grad():
                inner = x * 3.0
            after_inner = x * 4.0

        after_outer = x * 5.0

        self.assertFalse(outer.requires_grad)
        self.assertFalse(inner.requires_grad)
        self.assertFalse(after_inner.requires_grad)
        self.assertTrue(after_outer.requires_grad)

    def test_no_grad_restores_grad_tracking_after_exception(self) -> None:
        x = bt.tensor(np.asarray([1.0, -2.0, 3.0], dtype=np.float32), requires_grad=True)

        with self.assertRaisesRegex(RuntimeError, "boom"):
            with bt.no_grad():
                _ = x + 1.0
                raise RuntimeError("boom")

        y = x + 1.0
        self.assertTrue(y.requires_grad)


if __name__ == "__main__":
    unittest.main()
