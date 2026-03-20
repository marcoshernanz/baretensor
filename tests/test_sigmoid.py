import unittest

import numpy as np

import bt
from tests.utils import to_numpy


def _sigmoid_expected(source: np.ndarray) -> np.ndarray:
    positive = source >= 0.0
    out = np.empty_like(source, dtype=np.float32)

    positive_values = np.asarray(source[positive], dtype=np.float32)
    positive_exp = np.exp(-positive_values, dtype=np.float32)
    out[positive] = np.asarray(1.0 / (1.0 + positive_exp), dtype=np.float32)

    negative_values = np.asarray(source[~positive], dtype=np.float32)
    negative_exp = np.exp(negative_values, dtype=np.float32)
    out[~positive] = np.asarray(negative_exp / (1.0 + negative_exp), dtype=np.float32)

    return out


class SigmoidTests(unittest.TestCase):
    def test_sigmoid_contiguous_tensor_matches_numpy(self) -> None:
        source = np.linspace(-6.0, 6.0, num=12, dtype=np.float32).reshape(3, 4)
        tensor = bt.tensor(source)

        out = tensor.sigmoid()

        np.testing.assert_allclose(to_numpy(out), _sigmoid_expected(source), rtol=1e-6, atol=1e-6)

    def test_sigmoid_scalar_matches_numpy(self) -> None:
        source = np.asarray(-0.75, dtype=np.float32)
        tensor = bt.tensor(source)

        out = tensor.sigmoid()

        np.testing.assert_allclose(to_numpy(out), _sigmoid_expected(source), rtol=1e-6, atol=1e-6)

    def test_sigmoid_non_contiguous_tensor_matches_numpy(self) -> None:
        source = np.linspace(-8.0, 8.0, num=2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        transposed = bt.tensor(source).transpose(0, 2)

        out = transposed.sigmoid()

        expected = _sigmoid_expected(np.transpose(source, (2, 1, 0)))
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_sigmoid_zero_extent_tensor_returns_zero_extent_tensor(self) -> None:
        tensor = bt.zeros([0, 3])

        out = tensor.sigmoid()

        self.assertEqual(out.shape, [0, 3])
        np.testing.assert_allclose(
            to_numpy(out),
            _sigmoid_expected(np.zeros((0, 3), dtype=np.float32)),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_sigmoid_large_magnitude_inputs_saturate_without_infinities(self) -> None:
        source = np.asarray([-100.0, -40.0, 0.0, 40.0, 100.0], dtype=np.float32)
        tensor = bt.tensor(source)

        out = tensor.sigmoid()
        out_np = to_numpy(out)

        self.assertTrue(np.isfinite(out_np).all())
        np.testing.assert_allclose(out_np, _sigmoid_expected(source), rtol=1e-6, atol=1e-6)
        self.assertLess(float(out_np[0]), 1e-16)
        self.assertGreater(float(out_np[-1]), 1.0 - 1e-6)


if __name__ == "__main__":
    unittest.main()
