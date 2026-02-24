import unittest

import numpy as np

class TestTensorFromNumpy(unittest.TestCase):
    def test_tensor_from_contiguous_array(self) -> None:
        import bt

        array = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor = bt.tensor(array)

        self.assertEqual(tensor.shape, [3, 4])
        self.assertEqual(tensor.strides, [4, 1])
        self.assertEqual(tensor.dim(), 2)
        self.assertEqual(tensor.numel(), 12)
        self.assertTrue(tensor.is_contiguous())

    def test_tensor_from_empty_dimension(self) -> None:
        import bt

        array = np.zeros((0, 5), dtype=np.float32)
        tensor = bt.tensor(array)

        self.assertEqual(tensor.shape, [0, 5])
        self.assertEqual(tensor.numel(), 0)
        self.assertTrue(tensor.is_contiguous())


if __name__ == "__main__":
    unittest.main()
