import unittest
from typing import Any, cast

import numpy as np

import bt
import bt.nn.functional as F
from tests.utils import to_numpy


def _stable_log_softmax_expected(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_denom = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    return np.asarray(shifted - log_denom, dtype=np.float32)


def _expected_losses(
    logits: np.ndarray, target: np.ndarray, ignore_index: int
) -> tuple[np.ndarray, float, int]:
    log_probs = _stable_log_softmax_expected(logits)
    losses = np.zeros((target.shape[0],), dtype=np.float32)
    total = 0.0
    valid = 0
    for n, target_value in enumerate(target):
        class_index = int(target_value)
        if class_index == ignore_index:
            continue
        loss = float(-log_probs[n, class_index])
        losses[n] = np.float32(loss)
        total += loss
        valid += 1
    return losses, float(total), valid


class CrossEntropyTests(unittest.TestCase):
    def test_cross_entropy_mean_matches_numpy(self) -> None:
        logits = np.asarray([[2.0, 0.0, -1.0], [0.1, 0.2, 3.0]], dtype=np.float32)
        target = np.asarray([0.0, 2.0], dtype=np.float32)

        out = F.cross_entropy(bt.tensor(logits), bt.tensor(target))

        _, total, valid = _expected_losses(logits, target, ignore_index=-100)
        expected = np.asarray(total / valid, dtype=np.float32)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_cross_entropy_none_with_ignore_index(self) -> None:
        logits = np.asarray([[2.0, 0.0, -1.0], [1.0, 1.5, -2.0], [0.2, 0.7, 0.1]], dtype=np.float32)
        target = np.asarray([0.0, -100.0, 1.0], dtype=np.float32)

        out = F.cross_entropy(
            bt.tensor(logits),
            bt.tensor(target),
            ignore_index=-100,
            reduction="none",
        )

        expected, _, _ = _expected_losses(logits, target, ignore_index=-100)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_cross_entropy_sum_with_ignore_index(self) -> None:
        logits = np.asarray([[2.0, 0.0, -1.0], [1.0, 1.5, -2.0], [0.2, 0.7, 0.1]], dtype=np.float32)
        target = np.asarray([0.0, -100.0, 1.0], dtype=np.float32)

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
        logits = np.asarray([[1.0, 0.0], [2.0, -3.0]], dtype=np.float32)
        target = np.asarray([-100.0, -100.0], dtype=np.float32)

        out = F.cross_entropy(bt.tensor(logits), bt.tensor(target), reduction="mean")

        self.assertTrue(np.isnan(to_numpy(out)).item())

    def test_cross_entropy_non_contiguous_input(self) -> None:
        source = np.asarray(np.arange(12, dtype=np.float32).reshape(3, 4), dtype=np.float32)
        logits = bt.tensor(source).transpose(0, 1)
        target = np.asarray([0.0, 1.0, 2.0, 1.0], dtype=np.float32)

        out = F.cross_entropy(logits, bt.tensor(target))

        expected_logits = np.transpose(source, (1, 0))
        _, total, valid = _expected_losses(expected_logits, target, ignore_index=-100)
        expected = np.asarray(total / valid, dtype=np.float32)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_cross_entropy_invalid_reduction_raises(self) -> None:
        logits = bt.tensor(np.asarray([[0.1, 0.2]], dtype=np.float32))
        target = bt.tensor(np.asarray([0.0], dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"cross_entropy\(\) expected 'reduction' to be one of \{'none', 'mean', 'sum'\}\.",
        ):
            _ = F.cross_entropy(logits, target, reduction=cast(Any, "avg"))

    def test_cross_entropy_rejects_non_integer_targets(self) -> None:
        logits = bt.tensor(np.asarray([[0.1, 0.2], [0.4, 0.3]], dtype=np.float32))
        target = bt.tensor(np.asarray([0.0, 1.5], dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"cross_entropy failed for input shape \[2, 2\] and target shape \[2\]: "
            r"target values must be integer class indices, but target\[1\]=1.5\.",
        ):
            _ = F.cross_entropy(logits, target)

    def test_cross_entropy_rejects_out_of_range_target(self) -> None:
        logits = bt.tensor(np.asarray([[0.1, 0.2], [0.4, 0.3]], dtype=np.float32))
        target = bt.tensor(np.asarray([0.0, 2.0], dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"cross_entropy failed for input shape \[2, 2\] and target shape \[2\]: "
            r"target\[1\]=2 is out of range for 2 classes\.",
        ):
            _ = F.cross_entropy(logits, target)

    def test_cross_entropy_rejects_shape_mismatch(self) -> None:
        logits = bt.tensor(np.asarray([[0.1, 0.2], [0.4, 0.3]], dtype=np.float32))
        target = bt.tensor(np.asarray([0.0], dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"cross_entropy failed for input shape \[2, 2\] and target shape \[1\]: "
            r"target length must match input batch size N; got target\.shape\[0\]=1 and input\.shape\[0\]=2\.",
        ):
            _ = F.cross_entropy(logits, target)


if __name__ == "__main__":
    unittest.main()
