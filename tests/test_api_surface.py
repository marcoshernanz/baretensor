import unittest

import bt


class ApiSurfaceTests(unittest.TestCase):
    def test_public_exports(self) -> None:
        self.assertTrue(hasattr(bt, "Tensor"))
        self.assertTrue(hasattr(bt, "full"))
        self.assertTrue(hasattr(bt, "zeros"))
        self.assertTrue(hasattr(bt, "ones"))
        self.assertTrue(hasattr(bt, "tensor"))
        self.assertTrue(hasattr(bt, "nn"))
        self.assertTrue(hasattr(bt.nn, "functional"))
        self.assertFalse(hasattr(bt, "matmul"))
        self.assertTrue(hasattr(bt.Tensor, "matmul"))
        self.assertTrue(hasattr(bt.Tensor, "exp"))
        self.assertTrue(hasattr(bt.Tensor, "log"))
        self.assertTrue(hasattr(bt.Tensor, "softmax"))
        self.assertTrue(hasattr(bt.Tensor, "log_softmax"))
        self.assertTrue(hasattr(bt.Tensor, "sum"))
        self.assertTrue(hasattr(bt.Tensor, "mean"))
        self.assertTrue(hasattr(bt.Tensor, "max"))
        self.assertTrue(hasattr(bt.Tensor, "__getitem__"))
        self.assertTrue(hasattr(bt.nn.functional, "cross_entropy"))
        self.assertTrue(hasattr(bt.nn.functional, "layer_norm"))
        self.assertTrue(hasattr(bt.nn.functional, "embedding"))


if __name__ == "__main__":
    unittest.main()
