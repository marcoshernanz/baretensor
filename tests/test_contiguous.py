import unittest

import numpy as np

import bt
from tests.utils import to_numpy


class ContiguousTests(unittest.TestCase):
    def test_contiguous_preserves_shape_and_values(self) -> None:
        source = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor = bt.tensor(source)

        out = tensor.contiguous()

        self.assertEqual(out.shape, [3, 4])
        self.assertTrue(out.is_contiguous())
        np.testing.assert_allclose(to_numpy(out), source)

    def test_contiguous_scalar(self) -> None:
        tensor = bt.tensor(np.asarray(7.0, dtype=np.float32))

        out = tensor.contiguous()

        self.assertEqual(out.shape, [])
        self.assertTrue(out.is_contiguous())
        np.testing.assert_allclose(to_numpy(out), np.asarray(7.0, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
