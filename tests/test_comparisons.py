import unittest

import numpy as np

import bt
from tests.utils import to_numpy


def _comparison_expected(op: str, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if op == "eq":
        return np.asarray(lhs == rhs, dtype=np.bool_)
    if op == "ne":
        return np.asarray(lhs != rhs, dtype=np.bool_)
    if op == "lt":
        return np.asarray(lhs < rhs, dtype=np.bool_)
    if op == "le":
        return np.asarray(lhs <= rhs, dtype=np.bool_)
    if op == "gt":
        return np.asarray(lhs > rhs, dtype=np.bool_)
    if op == "ge":
        return np.asarray(lhs >= rhs, dtype=np.bool_)
    raise ValueError(f"Unsupported comparison op: {op}")


def _comparison_actual(op: str, lhs: bt.Tensor, rhs: bt.Tensor) -> bt.Tensor:
    if op == "eq":
        return lhs == rhs
    if op == "ne":
        return lhs != rhs
    if op == "lt":
        return lhs < rhs
    if op == "le":
        return lhs <= rhs
    if op == "gt":
        return lhs > rhs
    if op == "ge":
        return lhs >= rhs
    raise ValueError(f"Unsupported comparison op: {op}")


class ComparisonTests(unittest.TestCase):
    def test_tensor_scalar_greater_than_returns_bool_tensor(self) -> None:
        source_np = np.asarray([[-1.0, 0.0, 2.0], [3.0, -4.0, 5.0]], dtype=np.float32)
        source = bt.tensor(source_np, requires_grad=True)

        out = source > 0.0

        self.assertEqual(out.dtype, bt.bool)
        self.assertFalse(out.requires_grad)
        np.testing.assert_array_equal(to_numpy(out), source_np > 0.0)

    def test_tensor_tensor_broadcast_comparisons_match_numpy(self) -> None:
        lhs_np = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        rhs_np = np.asarray([[2], [4]], dtype=np.int64)
        lhs = bt.tensor(lhs_np)
        rhs = bt.tensor(rhs_np)

        np.testing.assert_array_equal(to_numpy(lhs == rhs), lhs_np == rhs_np)
        np.testing.assert_array_equal(to_numpy(lhs != rhs), lhs_np != rhs_np)
        np.testing.assert_array_equal(to_numpy(lhs < rhs), lhs_np < rhs_np)
        np.testing.assert_array_equal(to_numpy(lhs <= rhs), lhs_np <= rhs_np)
        np.testing.assert_array_equal(to_numpy(lhs > rhs), lhs_np > rhs_np)
        np.testing.assert_array_equal(to_numpy(lhs >= rhs), lhs_np >= rhs_np)

    def test_bool_tensor_comparison_is_supported(self) -> None:
        lhs = bt.tensor(np.asarray([True, False, True], dtype=np.bool_))
        rhs = bt.tensor(np.asarray([False, False, True], dtype=np.bool_))

        out = lhs == rhs

        self.assertEqual(out.dtype, bt.bool)
        np.testing.assert_array_equal(
            to_numpy(out), np.asarray([False, True, True], dtype=np.bool_)
        )

    def test_non_contiguous_comparisons_match_numpy(self) -> None:
        lhs_base = np.arange(12, dtype=np.float32).reshape(3, 4)
        rhs_base = np.linspace(2.0, 9.0, num=12, dtype=np.float32).reshape(3, 4)
        lhs = bt.tensor(lhs_base).transpose(0, 1)
        rhs = bt.tensor(rhs_base).transpose(0, 1)
        lhs_np = lhs_base.T
        rhs_np = rhs_base.T

        np.testing.assert_array_equal(to_numpy(lhs < rhs), np.asarray(lhs_np < rhs_np, dtype=np.bool_))
        np.testing.assert_array_equal(
            to_numpy(lhs >= rhs), np.asarray(lhs_np >= rhs_np, dtype=np.bool_)
        )

    def test_randomized_broadcast_comparisons_match_numpy(self) -> None:
        shape_pool = [(), (1,), (3,), (1, 3), (2, 1), (1, 3, 1), (2, 3, 4), (2, 0, 4)]
        ops = ("eq", "ne", "lt", "le", "gt", "ge")

        for seed in range(30):
            rng = np.random.default_rng(500 + seed)
            op = ops[seed % len(ops)]
            lhs_shape = shape_pool[int(rng.integers(0, len(shape_pool)))]
            rhs_shape = shape_pool[int(rng.integers(0, len(shape_pool)))]
            try:
                np.broadcast_shapes(lhs_shape, rhs_shape)
            except ValueError:
                continue

            if seed % 3 == 0:
                lhs_np = rng.normal(size=lhs_shape).astype(np.float32)
                rhs_np = rng.normal(size=rhs_shape).astype(np.float32)
            elif seed % 3 == 1:
                lhs_np = rng.integers(-5, 6, size=lhs_shape, dtype=np.int64)
                rhs_np = rng.integers(-5, 6, size=rhs_shape, dtype=np.int64)
            else:
                if op not in ("eq", "ne"):
                    continue
                lhs_np = rng.integers(0, 2, size=lhs_shape, dtype=np.int64).astype(np.bool_)
                rhs_np = rng.integers(0, 2, size=rhs_shape, dtype=np.int64).astype(np.bool_)

            lhs = bt.tensor(lhs_np)
            rhs = bt.tensor(rhs_np)
            actual = _comparison_actual(op, lhs, rhs)
            expected = _comparison_expected(op, lhs_np, rhs_np)

            self.assertEqual(actual.dtype, bt.bool)
            np.testing.assert_array_equal(to_numpy(actual), expected)


if __name__ == "__main__":
    unittest.main()
