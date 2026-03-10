import unittest
from typing import Any, cast

import numpy as np

import bt
import bt.nn.functional as F
from tests.utils import to_numpy


def _stable_log_softmax_expected(logits: np.ndarray) -> np.ndarray:
    class_dim = 0 if logits.ndim == 1 else 1
    shifted = logits - np.max(logits, axis=class_dim, keepdims=True)
    log_denom = np.log(np.sum(np.exp(shifted), axis=class_dim, keepdims=True))
    return np.asarray(shifted - log_denom, dtype=np.float32)


def _expected_losses(
    logits: np.ndarray, target: np.ndarray, ignore_index: int
) -> tuple[np.ndarray, float, int]:
    log_probs = _stable_log_softmax_expected(logits)
    class_dim = 0 if logits.ndim == 1 else 1
    losses = np.zeros(target.shape, dtype=np.float32)
    total = 0.0
    valid = 0
    for index in np.ndindex(target.shape):
        target_value = target[index]
        class_index = int(target_value)
        if class_index == ignore_index:
            continue

        if class_dim == 0:
            log_prob_index = (class_index,)
        else:
            log_prob_index = (index[0], class_index, *index[1:])
        loss = float(-log_probs[log_prob_index])
        losses[index] = np.float32(loss)
        total += loss
        valid += 1
    return losses, float(total), valid


class CrossEntropyTests(unittest.TestCase):
    def test_cross_entropy_unbatched_mean_matches_numpy(self) -> None:
        logits = np.asarray([2.0, 0.0, -1.0], dtype=np.float32)
        target = np.asarray(0, dtype=np.int64)

        out = F.cross_entropy(bt.tensor(logits), bt.tensor(target))

        _, total, valid = _expected_losses(logits, target, ignore_index=-100)
        expected = np.asarray(total / valid, dtype=np.float32)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_cross_entropy_unbatched_none_returns_scalar(self) -> None:
        logits = np.asarray([2.0, 0.0, -1.0], dtype=np.float32)
        target = np.asarray(0, dtype=np.int64)

        out = F.cross_entropy(bt.tensor(logits), bt.tensor(target), reduction="none")

        self.assertEqual(out.shape, [])
        expected, _, _ = _expected_losses(logits, target, ignore_index=-100)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_cross_entropy_unbatched_ignore_index_mean_is_nan(self) -> None:
        logits = np.asarray([1.0, 0.0], dtype=np.float32)
        target = np.asarray(-100, dtype=np.int64)

        out = F.cross_entropy(
            bt.tensor(logits),
            bt.tensor(target),
            ignore_index=-100,
            reduction="mean",
        )
        self.assertTrue(np.isnan(to_numpy(out)).item())

    def test_cross_entropy_mean_matches_numpy(self) -> None:
        logits = np.asarray([[2.0, 0.0, -1.0], [0.1, 0.2, 3.0]], dtype=np.float32)
        target = np.asarray([0, 2], dtype=np.int64)

        out = F.cross_entropy(bt.tensor(logits), bt.tensor(target))

        _, total, valid = _expected_losses(logits, target, ignore_index=-100)
        expected = np.asarray(total / valid, dtype=np.float32)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_cross_entropy_mean_matches_numpy_for_sequence_logits(self) -> None:
        logits = np.asarray(
            [
                [[3.0, 2.0, 1.0, 0.0], [0.5, 0.1, 0.2, 0.3], [-1.0, -2.0, -3.0, -4.0]],
                [[0.0, 1.0, 2.0, 3.0], [2.0, 1.0, 0.0, -1.0], [-0.2, -0.1, 0.0, 0.1]],
            ],
            dtype=np.float32,
        )
        target = np.asarray([[0, 1, 2, 1], [2, 0, 1, 0]], dtype=np.int64)

        out = F.cross_entropy(bt.tensor(logits), bt.tensor(target))

        _, total, valid = _expected_losses(logits, target, ignore_index=-100)
        expected = np.asarray(total / valid, dtype=np.float32)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_cross_entropy_none_with_ignore_index_for_sequence_logits(self) -> None:
        logits = np.asarray(
            [
                [[2.0, 0.0, -1.0, 0.5], [1.0, 1.5, -2.0, 1.0], [0.2, 0.7, 0.1, 0.3]],
                [[1.0, 0.5, 0.2, -0.1], [0.0, -0.5, -1.0, -1.5], [2.0, 2.5, 3.0, 3.5]],
            ],
            dtype=np.float32,
        )
        target = np.asarray([[0, -100, 1, 2], [2, 2, -100, 1]], dtype=np.int64)

        out = F.cross_entropy(
            bt.tensor(logits),
            bt.tensor(target),
            ignore_index=-100,
            reduction="none",
        )

        expected, _, _ = _expected_losses(logits, target, ignore_index=-100)
        self.assertEqual(out.shape, [2, 4])
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_cross_entropy_sum_with_ignore_index(self) -> None:
        logits = np.asarray([[2.0, 0.0, -1.0], [1.0, 1.5, -2.0], [0.2, 0.7, 0.1]], dtype=np.float32)
        target = np.asarray([0, -100, 1], dtype=np.int64)

        out = F.cross_entropy(
            bt.tensor(logits),
            bt.tensor(target),
            ignore_index=-100,
            reduction="sum",
        )

        _, total, _ = _expected_losses(logits, target, ignore_index=-100)
        expected = np.asarray(total, dtype=np.float32)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_cross_entropy_all_ignored_mean_is_nan(self) -> None:
        logits = np.asarray([[[1.0, 0.0], [2.0, -3.0]]], dtype=np.float32)
        target = np.asarray([[-100, -100]], dtype=np.int64)

        out = F.cross_entropy(bt.tensor(logits), bt.tensor(target), reduction="mean")

        self.assertTrue(np.isnan(to_numpy(out)).item())

    def test_cross_entropy_non_contiguous_sequence_input(self) -> None:
        source = np.asarray(np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3), dtype=np.float32)
        logits = bt.tensor(source).transpose(1, 2)
        target = np.asarray([[0, 1, 2, 1], [2, 0, 1, 2]], dtype=np.int64)

        out = F.cross_entropy(logits, bt.tensor(target))

        expected_logits = np.transpose(source, (0, 2, 1))
        _, total, valid = _expected_losses(expected_logits, target, ignore_index=-100)
        expected = np.asarray(total / valid, dtype=np.float32)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_cross_entropy_invalid_reduction_raises(self) -> None:
        logits = bt.tensor(np.asarray([[0.1, 0.2]], dtype=np.float32))
        target = bt.tensor(np.asarray([0], dtype=np.int64))

        with self.assertRaisesRegex(
            ValueError,
            r"cross_entropy\(\) expected 'reduction' to be one of \{'none', 'mean', 'sum'\}\.",
        ):
            _ = F.cross_entropy(logits, target, reduction=cast(Any, "avg"))

    def test_cross_entropy_rejects_wrong_target_dtype(self) -> None:
        logits = bt.tensor(np.asarray([[0.1, 0.2], [0.4, 0.3]], dtype=np.float32))
        target = bt.tensor(np.asarray([0.0, 1.0], dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"cross_entropy failed for input shape \[2, 2\] and target shape \[2\]: "
            r"target must have dtype int64\.",
        ):
            _ = F.cross_entropy(logits, target)

    def test_cross_entropy_rejects_out_of_range_target(self) -> None:
        logits = bt.tensor(np.asarray([[0.1, 0.2], [0.4, 0.3]], dtype=np.float32))
        target = bt.tensor(np.asarray([0, 2], dtype=np.int64))

        with self.assertRaisesRegex(
            ValueError,
            r"cross_entropy failed for input shape \[2, 2\] and target shape \[2\]: "
            r"target class index 2 is out of range for 2 classes\.",
        ):
            _ = F.cross_entropy(logits, target)

    def test_cross_entropy_rejects_target_shape_mismatch(self) -> None:
        logits = bt.tensor(np.asarray(np.zeros((2, 3, 4), dtype=np.float32), dtype=np.float32))
        target = bt.tensor(np.asarray(np.zeros((2, 3), dtype=np.int64), dtype=np.int64))

        with self.assertRaisesRegex(
            ValueError,
            r"cross_entropy failed for input shape \[2, 3, 4\] and target shape \[2, 3\]: "
            r"target shape must be \[2, 4\] to match input by removing the class dimension\.",
        ):
            _ = F.cross_entropy(logits, target)

    def test_cross_entropy_rejects_input_rank_below_one(self) -> None:
        logits = bt.tensor(np.asarray(1.0, dtype=np.float32))
        target = bt.tensor(np.asarray(0, dtype=np.int64))

        with self.assertRaisesRegex(
            ValueError,
            r"cross_entropy failed for input shape \[\] and target shape \[\]: "
            r"input must have rank >= 1 with shape \[C\] or \[N, C, \.\.\.\]\.",
        ):
            _ = F.cross_entropy(logits, target)

    def test_cross_entropy_rejects_non_positive_class_count(self) -> None:
        logits = bt.zeros([2, 0, 4])
        target = bt.tensor(np.asarray(np.zeros((2, 4), dtype=np.int64), dtype=np.int64))

        with self.assertRaisesRegex(
            ValueError,
            r"cross_entropy failed for input shape \[2, 0, 4\] and target shape \[2, 4\]: "
            r"input class dimension size must be positive, got 0\.",
        ):
            _ = F.cross_entropy(logits, target)


if __name__ == "__main__":
    unittest.main()
