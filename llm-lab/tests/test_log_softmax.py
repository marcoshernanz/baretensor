import unittest

import numpy as np

import bt
from tests.utils import to_numpy


def _stable_log_softmax_expected(source: np.ndarray, dim: int) -> np.ndarray:
    axis = dim if dim >= 0 else dim + source.ndim
    if source.shape[axis] == 0:
        return np.asarray(source, dtype=np.float32)

    shifted = source - np.max(source, axis=axis, keepdims=True)
    log_denom = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return np.asarray(shifted - log_denom, dtype=np.float32)


class LogSoftmaxTests(unittest.TestCase):
    def test_log_softmax_matches_numpy_on_contiguous_tensor(self) -> None:
        source = np.asarray(np.arange(12, dtype=np.float32).reshape(3, 4), dtype=np.float32)
        tensor = bt.tensor(source)

        out = tensor.log_softmax(dim=1)

        expected = _stable_log_softmax_expected(source, 1)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_log_softmax_accepts_negative_dim(self) -> None:
        source = np.asarray(np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4), dtype=np.float32)
        tensor = bt.tensor(source)

        out = tensor.log_softmax(dim=-1)

        expected = _stable_log_softmax_expected(source, -1)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_log_softmax_matches_numpy_on_non_contiguous_tensor(self) -> None:
        source = np.asarray(np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4), dtype=np.float32)
        transposed = bt.tensor(source).transpose(0, 2)

        out = transposed.log_softmax(dim=1)

        expected = _stable_log_softmax_expected(np.transpose(source, (2, 1, 0)), 1)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_log_softmax_is_numerically_stable_for_large_values(self) -> None:
        source = np.asarray([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
        tensor = bt.tensor(source)

        out = tensor.log_softmax(dim=1)

        expected = _stable_log_softmax_expected(source, 1)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_exp_of_log_softmax_sums_to_one_along_dim(self) -> None:
        source = np.asarray(np.linspace(-2.0, 2.0, num=2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5))
        tensor = bt.tensor(source)

        out = to_numpy(tensor.log_softmax(dim=2))
        probs = np.exp(out)
        sums = np.sum(probs, axis=2)
        np.testing.assert_allclose(sums, np.ones_like(sums, dtype=np.float32), rtol=1e-6, atol=1e-6)

    def test_log_softmax_matches_log_of_softmax(self) -> None:
        source = np.asarray(np.linspace(-3.0, 3.0, num=2 * 4, dtype=np.float32).reshape(2, 4))
        tensor = bt.tensor(source)

        direct = to_numpy(tensor.log_softmax(dim=1))
        composed = to_numpy(tensor.softmax(dim=1).log())
        np.testing.assert_allclose(direct, composed, rtol=1e-6, atol=1e-6)

    def test_log_softmax_empty_dim_returns_empty_tensor(self) -> None:
        source = np.asarray(np.zeros((2, 0, 4), dtype=np.float32), dtype=np.float32)
        tensor = bt.tensor(source)

        out = tensor.log_softmax(dim=1)

        self.assertEqual(out.shape, [2, 0, 4])
        expected = _stable_log_softmax_expected(source, 1)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_log_softmax_invalid_dim_raises_with_context(self) -> None:
        tensor = bt.tensor(np.asarray(np.zeros((2, 3), dtype=np.float32), dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"log_softmax failed for tensor with shape \[2, 3\]: dim=2 is out of range for rank 2\.",
        ):
            _ = tensor.log_softmax(dim=2)


if __name__ == "__main__":
    unittest.main()
