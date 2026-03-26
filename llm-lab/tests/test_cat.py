import unittest

import numpy as np

import bt
from tests.utils import to_numpy


def _require_grad(tensor: bt.Tensor) -> bt.Tensor:
    grad = tensor.grad
    assert grad is not None
    return grad


class CatTests(unittest.TestCase):
    def test_cat_along_dim_zero_matches_numpy(self) -> None:
        left = np.arange(6, dtype=np.float32).reshape(2, 3)
        right = np.arange(6, 18, dtype=np.float32).reshape(4, 3)

        out = bt.cat([bt.tensor(left), bt.tensor(right)], dim=0)

        expected = np.concatenate([left, right], axis=0)
        self.assertEqual(out.shape, [6, 3])
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_cat_along_dim_one_matches_numpy(self) -> None:
        left = np.arange(6, dtype=np.float32).reshape(2, 3)
        right = np.arange(6, 10, dtype=np.float32).reshape(2, 2)

        out = bt.cat([bt.tensor(left), bt.tensor(right)], dim=1)

        expected = np.concatenate([left, right], axis=1)
        self.assertEqual(out.shape, [2, 5])
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_cat_accepts_negative_dim(self) -> None:
        left = np.arange(6, dtype=np.float32).reshape(2, 3)
        right = np.arange(6, 10, dtype=np.float32).reshape(2, 2)

        out = bt.cat([bt.tensor(left), bt.tensor(right)], dim=-1)

        expected = np.concatenate([left, right], axis=-1)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_cat_supports_non_contiguous_inputs(self) -> None:
        left_source = np.arange(12, dtype=np.float32).reshape(3, 4)
        right_source = np.arange(12, 24, dtype=np.float32).reshape(3, 4)
        left = bt.tensor(left_source).transpose(0, 1)
        right = bt.tensor(right_source).transpose(0, 1)

        out = bt.cat([left, right], dim=1)

        expected = np.concatenate([left_source.T, right_source.T], axis=1)
        self.assertEqual(out.shape, [4, 6])
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_cat_ignores_special_case_empty_vector(self) -> None:
        matrix = np.arange(6, dtype=np.float32).reshape(2, 3)
        empty = np.asarray([], dtype=np.float32)

        out = bt.cat([bt.tensor(matrix), bt.tensor(empty)], dim=1)

        self.assertEqual(out.shape, [2, 3])
        np.testing.assert_allclose(to_numpy(out), matrix, rtol=1e-6, atol=1e-6)

    def test_cat_all_special_case_empty_vectors_returns_empty_vector(self) -> None:
        out = bt.cat(
            [
                bt.tensor(np.asarray([], dtype=np.float32)),
                bt.tensor(np.asarray([], dtype=np.float32)),
            ],
            dim=0,
        )

        self.assertEqual(out.shape, [0])
        np.testing.assert_allclose(to_numpy(out), np.asarray([], dtype=np.float32), rtol=1e-6, atol=1e-6)

    def test_cat_rejects_empty_sequence(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"cat\(\) expected a non-empty sequence of tensors",
        ):
            _ = bt.cat([])

    def test_cat_rejects_scalar_tensor(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"zero-dimensional tensor at position 0 cannot be concatenated",
        ):
            _ = bt.cat(
                [
                    bt.tensor(np.asarray(1.0, dtype=np.float32)),
                    bt.tensor(np.asarray(2.0, dtype=np.float32)),
                ]
            )

    def test_cat_rejects_rank_mismatch(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"tensor at position 1 has rank 3 but expected rank 2",
        ):
            _ = bt.cat(
                [
                    bt.tensor(np.zeros((2, 3), dtype=np.float32)),
                    bt.tensor(np.zeros((2, 3, 4), dtype=np.float32)),
                ],
                dim=0,
            )

    def test_cat_rejects_shape_mismatch_outside_concat_dim(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"sizes must match except in dimension 1",
        ):
            _ = bt.cat(
                [
                    bt.tensor(np.zeros((2, 3), dtype=np.float32)),
                    bt.tensor(np.zeros((4, 2), dtype=np.float32)),
                ],
                dim=1,
            )

    def test_cat_backward_splits_gradients_along_concat_dim(self) -> None:
        left = bt.tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)
        right = bt.tensor(np.arange(4, dtype=np.float32).reshape(2, 2), requires_grad=True)
        weight_np = np.arange(10, dtype=np.float32).reshape(2, 5)
        weight = bt.tensor(weight_np)

        out = bt.cat([left, right], dim=1)
        loss = (out * weight).sum()
        loss.backward()

        np.testing.assert_allclose(
            to_numpy(_require_grad(left)),
            weight_np[:, :3],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            to_numpy(_require_grad(right)),
            weight_np[:, 3:],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_cat_backward_gives_empty_gradient_to_special_case_empty_vector(self) -> None:
        left = bt.tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)
        empty = bt.tensor(np.asarray([], dtype=np.float32), requires_grad=True)
        weight = bt.tensor(np.arange(6, dtype=np.float32).reshape(2, 3))

        out = bt.cat([left, empty], dim=0)
        loss = (out * weight).sum()
        loss.backward()

        np.testing.assert_allclose(
            to_numpy(_require_grad(left)),
            np.arange(6, dtype=np.float32).reshape(2, 3),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            to_numpy(_require_grad(empty)),
            np.asarray([], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
