import unittest

import numpy as np

import bt
import bt.nn.functional as F
from tests.utils import to_numpy


def _expected_embedding(indices: np.ndarray, weight: np.ndarray) -> np.ndarray:
    gathered = np.take(weight, indices.astype(np.int64), axis=0)
    return np.asarray(gathered, dtype=np.float32)


class EmbeddingTests(unittest.TestCase):
    def test_embedding_matches_numpy_for_matrix_indices(self) -> None:
        indices = np.asarray([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=np.int64)
        weight = np.asarray(np.arange(10 * 3, dtype=np.float32).reshape(10, 3), dtype=np.float32)

        out = F.embedding(bt.tensor(indices), bt.tensor(weight))

        expected = _expected_embedding(indices, weight)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_embedding_scalar_index_returns_embedding_vector(self) -> None:
        index = np.asarray(2, dtype=np.int64)
        weight = np.asarray(np.arange(6 * 4, dtype=np.float32).reshape(6, 4), dtype=np.float32)

        out = F.embedding(bt.tensor(index), bt.tensor(weight))

        self.assertEqual(out.shape, [4])
        expected = _expected_embedding(index, weight)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_embedding_supports_non_contiguous_indices(self) -> None:
        source = np.asarray([[1, 3, 0], [2, 1, 4]], dtype=np.int64)
        indices = bt.tensor(source).transpose(0, 1)
        weight = np.asarray(np.arange(5 * 2, dtype=np.float32).reshape(5, 2), dtype=np.float32)

        out = F.embedding(indices, bt.tensor(weight))

        expected = _expected_embedding(np.transpose(source, (1, 0)), weight)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_embedding_supports_non_contiguous_weight(self) -> None:
        base = np.asarray(np.arange(3 * 7, dtype=np.float32).reshape(3, 7), dtype=np.float32)
        weight = bt.tensor(base).transpose(0, 1)
        indices = np.asarray([[0, 2], [6, 1]], dtype=np.int64)

        out = F.embedding(bt.tensor(indices), weight)

        expected_weight = np.transpose(base, (1, 0))
        expected = _expected_embedding(indices, expected_weight)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_embedding_empty_input_returns_empty_output_with_embedding_dim(self) -> None:
        indices = np.asarray(np.zeros((0, 3), dtype=np.int64), dtype=np.int64)
        weight = np.asarray(np.arange(5 * 4, dtype=np.float32).reshape(5, 4), dtype=np.float32)

        out = F.embedding(bt.tensor(indices), bt.tensor(weight))

        self.assertEqual(out.shape, [0, 3, 4])
        expected = _expected_embedding(indices, weight)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_embedding_rejects_weight_rank_not_two(self) -> None:
        indices = bt.tensor(np.asarray([0, 1], dtype=np.int64))
        weight = bt.tensor(np.asarray(np.zeros((2, 3, 4), dtype=np.float32), dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"embedding failed for input shape \[2\] and weight shape \[2, 3, 4\]: "
            r"weight must have rank 2 with shape \[V, D\]\.",
        ):
            _ = F.embedding(indices, weight)

    def test_embedding_rejects_wrong_input_dtype(self) -> None:
        indices = bt.tensor(np.asarray([0.0, 1.0], dtype=np.float32))
        weight = bt.tensor(np.asarray(np.arange(3 * 2, dtype=np.float32).reshape(3, 2), dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"embedding failed for input shape \[2\] and weight shape \[3, 2\]: "
            r"input must have dtype int64\.",
        ):
            _ = F.embedding(indices, weight)

    def test_embedding_rejects_index_out_of_range(self) -> None:
        indices = bt.tensor(np.asarray([0, 3], dtype=np.int64))
        weight = bt.tensor(np.asarray(np.arange(3 * 2, dtype=np.float32).reshape(3, 2), dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"embedding failed for input shape \[2\] and weight shape \[3, 2\]: "
            r"index 3 is out of range for vocab size 3\.",
        ):
            _ = F.embedding(indices, weight)

    def test_embedding_rejects_unsupported_max_norm(self) -> None:
        indices = bt.tensor(np.asarray([0, 1], dtype=np.int64))
        weight = bt.tensor(np.asarray(np.arange(3 * 2, dtype=np.float32).reshape(3, 2), dtype=np.float32))

        with self.assertRaisesRegex(
            NotImplementedError,
            r"embedding\(\) does not support max_norm yet\.",
        ):
            _ = F.embedding(indices, weight, max_norm=1.0)

    def test_embedding_rejects_unsupported_norm_type(self) -> None:
        indices = bt.tensor(np.asarray([0, 1], dtype=np.int64))
        weight = bt.tensor(np.asarray(np.arange(3 * 2, dtype=np.float32).reshape(3, 2), dtype=np.float32))

        with self.assertRaisesRegex(
            NotImplementedError,
            r"embedding\(\) does not support norm_type != 2.0 yet\.",
        ):
            _ = F.embedding(indices, weight, norm_type=1.0)

    def test_embedding_rejects_unsupported_scale_grad_by_freq(self) -> None:
        indices = bt.tensor(np.asarray([0, 1], dtype=np.int64))
        weight = bt.tensor(np.asarray(np.arange(3 * 2, dtype=np.float32).reshape(3, 2), dtype=np.float32))

        with self.assertRaisesRegex(
            NotImplementedError,
            r"embedding\(\) does not support scale_grad_by_freq yet\.",
        ):
            _ = F.embedding(indices, weight, scale_grad_by_freq=True)

    def test_embedding_rejects_unsupported_sparse(self) -> None:
        indices = bt.tensor(np.asarray([0, 1], dtype=np.int64))
        weight = bt.tensor(np.asarray(np.arange(3 * 2, dtype=np.float32).reshape(3, 2), dtype=np.float32))

        with self.assertRaisesRegex(
            NotImplementedError,
            r"embedding\(\) does not support sparse gradients yet\.",
        ):
            _ = F.embedding(indices, weight, sparse=True)

if __name__ == "__main__":
    unittest.main()
