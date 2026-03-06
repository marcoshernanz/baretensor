import unittest

import numpy as np

import bt
from tests.utils import to_numpy


class TanhTests(unittest.TestCase):
    def test_tanh_contiguous_tensor_matches_numpy(self) -> None:
        source = np.linspace(-3.0, 3.0, num=12, dtype=np.float32).reshape(3, 4)
        tensor = bt.tensor(source)

        out = tensor.tanh()

        np.testing.assert_allclose(
            to_numpy(out),
            np.asarray(np.tanh(source), dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_tanh_scalar_matches_numpy(self) -> None:
        source = np.asarray(0.5, dtype=np.float32)
        tensor = bt.tensor(source)

        out = tensor.tanh()

        np.testing.assert_allclose(
            to_numpy(out),
            np.asarray(np.tanh(source), dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_tanh_non_contiguous_tensor_matches_numpy(self) -> None:
        source = np.linspace(-4.0, 4.0, num=2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        transposed = bt.tensor(source).transpose(0, 2)

        out = transposed.tanh()

        expected = np.asarray(np.tanh(np.transpose(source, (2, 1, 0))), dtype=np.float32)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_tanh_zero_extent_tensor_returns_zero_extent_tensor(self) -> None:
        tensor = bt.zeros([0, 3])

        out = tensor.tanh()

        self.assertEqual(out.shape, [0, 3])
        np.testing.assert_allclose(
            to_numpy(out),
            np.asarray(np.tanh(np.zeros((0, 3), dtype=np.float32)), dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
