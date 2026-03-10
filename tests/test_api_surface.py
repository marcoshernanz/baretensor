import unittest

import bt


class ApiSurfaceTests(unittest.TestCase):
    def test_public_exports(self) -> None:
        self.assertTrue(hasattr(bt, "DType"))
        self.assertTrue(hasattr(bt, "Tensor"))
        self.assertTrue(hasattr(bt, "full"))
        self.assertTrue(hasattr(bt, "float32"))
        self.assertTrue(hasattr(bt, "int64"))
        self.assertTrue(hasattr(bt, "zeros"))
        self.assertTrue(hasattr(bt, "ones"))
        self.assertTrue(hasattr(bt, "tensor"))
        self.assertTrue(hasattr(bt, "cat"))
        self.assertTrue(hasattr(bt, "nn"))
        self.assertTrue(hasattr(bt, "no_grad"))
        self.assertTrue(hasattr(bt.nn, "functional"))
        self.assertFalse(hasattr(bt, "matmul"))
        self.assertFalse(hasattr(bt.Tensor, "cat"))
        self.assertTrue(hasattr(bt.Tensor, "matmul"))
        self.assertTrue(hasattr(bt.Tensor, "exp"))
        self.assertTrue(hasattr(bt.Tensor, "log"))
        self.assertTrue(hasattr(bt.Tensor, "tanh"))
        self.assertTrue(hasattr(bt.Tensor, "softmax"))
        self.assertTrue(hasattr(bt.Tensor, "log_softmax"))
        self.assertTrue(hasattr(bt.Tensor, "sum"))
        self.assertTrue(hasattr(bt.Tensor, "mean"))
        self.assertTrue(hasattr(bt.Tensor, "max"))
        self.assertTrue(hasattr(bt.Tensor, "flatten"))
        self.assertTrue(hasattr(bt.Tensor, "item"))
        self.assertTrue(hasattr(bt.Tensor, "dtype"))
        self.assertTrue(hasattr(bt.Tensor, "to"))
        self.assertTrue(hasattr(bt.Tensor, "__getitem__"))
        self.assertTrue(hasattr(bt.Tensor, "__neg__"))
        self.assertTrue(hasattr(bt.Tensor, "__radd__"))
        self.assertTrue(hasattr(bt.Tensor, "__rsub__"))
        self.assertTrue(hasattr(bt.Tensor, "__rmul__"))
        self.assertTrue(hasattr(bt.Tensor, "__rtruediv__"))
        self.assertTrue(hasattr(bt.Tensor, "__iadd__"))
        self.assertTrue(hasattr(bt.Tensor, "__isub__"))
        self.assertTrue(hasattr(bt.Tensor, "__imul__"))
        self.assertTrue(hasattr(bt.Tensor, "__itruediv__"))
        self.assertTrue(hasattr(bt.nn.functional, "cross_entropy"))
        self.assertTrue(hasattr(bt.nn.functional, "layer_norm"))
        self.assertTrue(hasattr(bt.nn.functional, "embedding"))


if __name__ == "__main__":
    unittest.main()
