import unittest

import numpy as np

import bt
from tests.utils import to_numpy


def _numpy_log_expected(source: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(np.log(source), dtype=np.float32)


class LogTests(unittest.TestCase):
    def test_log_contiguous_tensor_matches_numpy(self) -> None:
        source = np.linspace(0.1, 3.0, num=12, dtype=np.float32).reshape(3, 4)
        tensor = bt.tensor(source)

        out = tensor.log()

        np.testing.assert_allclose(
            to_numpy(out),
            _numpy_log_expected(source),
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )

    def test_log_scalar_matches_numpy(self) -> None:
        source = np.asarray(np.e, dtype=np.float32)
        tensor = bt.tensor(source)

        out = tensor.log()

        np.testing.assert_allclose(
            to_numpy(out),
            _numpy_log_expected(source),
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )

    def test_log_non_contiguous_tensor_matches_numpy(self) -> None:
        source = np.linspace(0.1, 4.8, num=2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        transposed = bt.tensor(source).transpose(0, 2)

        out = transposed.log()

        expected = _numpy_log_expected(np.transpose(source, (2, 1, 0)))
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6, equal_nan=True)

    def test_log_zero_extent_tensor_returns_zero_extent_tensor(self) -> None:
        tensor = bt.zeros([0, 3]) + 1.0

        out = tensor.log()

        self.assertEqual(out.shape, [0, 3])
        expected = _numpy_log_expected(np.ones((0, 3), dtype=np.float32))
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6, equal_nan=True)

    def test_log_handles_zero_and_negative_inputs_like_numpy(self) -> None:
        source = np.asarray([[1.0, 0.0, -1.0]], dtype=np.float32)
        tensor = bt.tensor(source)

        out = tensor.log()

        expected = _numpy_log_expected(source)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
