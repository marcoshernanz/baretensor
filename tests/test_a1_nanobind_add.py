import unittest

import numpy as np

class TestTensorOps(unittest.TestCase):
    def test_factories_and_binary_ops(self) -> None:
        import bt

        a = bt.full([2, 3], 2.0)
        b = bt.ones([2, 3])
        c = a + b
        d = c * 2.0
        e = d / b

        self.assertEqual(a.shape, [2, 3])
        self.assertEqual(c.shape, [2, 3])
        self.assertEqual(e.shape, [2, 3])
        self.assertEqual(e.dim(), 2)
        self.assertEqual(e.numel(), 6)
        self.assertTrue(e.is_contiguous())

    def test_scalar_tensor_metadata(self) -> None:
        import bt

        scalar = bt.tensor(np.array(7.0, dtype=np.float32))
        self.assertEqual(scalar.shape, [])
        self.assertEqual(scalar.dim(), 0)
        self.assertEqual(scalar.numel(), 1)
        self.assertTrue(scalar.is_contiguous())


if __name__ == "__main__":
    unittest.main()
