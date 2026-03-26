import unittest

import numpy as np

import bt
from tests.utils import to_numpy


def _require_grad(tensor: bt.Tensor) -> bt.Tensor:
    grad = tensor.grad
    assert grad is not None
    return grad


class StackTests(unittest.TestCase):
    def test_stack_along_dim_zero_matches_numpy(self) -> None:
        first = np.arange(6, dtype=np.float32).reshape(2, 3)
        second = np.arange(6, 12, dtype=np.float32).reshape(2, 3)

        out = bt.stack([bt.tensor(first), bt.tensor(second)], dim=0)

        expected = np.stack([first, second], axis=0)
        self.assertEqual(out.shape, [2, 2, 3])
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_stack_along_middle_dim_matches_numpy(self) -> None:
        first = np.arange(6, dtype=np.float32).reshape(2, 3)
        second = np.arange(6, 12, dtype=np.float32).reshape(2, 3)

        out = bt.stack([bt.tensor(first), bt.tensor(second)], dim=1)

        expected = np.stack([first, second], axis=1)
        self.assertEqual(out.shape, [2, 2, 3])
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_stack_accepts_negative_dim(self) -> None:
        first = np.arange(6, dtype=np.float32).reshape(2, 3)
        second = np.arange(6, 12, dtype=np.float32).reshape(2, 3)

        out = bt.stack([bt.tensor(first), bt.tensor(second)], dim=-1)

        expected = np.stack([first, second], axis=-1)
        self.assertEqual(out.shape, [2, 3, 2])
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_stack_supports_scalar_tensors(self) -> None:
        out = bt.stack(
            [
                bt.tensor(np.asarray(1.5, dtype=np.float32)),
                bt.tensor(np.asarray(2.5, dtype=np.float32)),
            ],
            dim=0,
        )

        expected = np.stack(
            [
                np.asarray(1.5, dtype=np.float32),
                np.asarray(2.5, dtype=np.float32),
            ],
            axis=0,
        )
        self.assertEqual(out.shape, [2])
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_stack_supports_non_contiguous_inputs(self) -> None:
        first_source = np.arange(12, dtype=np.float32).reshape(3, 4)
        second_source = np.arange(12, 24, dtype=np.float32).reshape(3, 4)
        first = bt.tensor(first_source).transpose(0, 1)
        second = bt.tensor(second_source).transpose(0, 1)

        out = bt.stack([first, second], dim=1)

        expected = np.stack([first_source.T, second_source.T], axis=1)
        self.assertEqual(out.shape, [4, 2, 3])
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_stack_supports_empty_tensors_with_matching_shape(self) -> None:
        first = np.asarray(np.zeros((0, 3), dtype=np.float32), dtype=np.float32)
        second = np.asarray(np.ones((0, 3), dtype=np.float32), dtype=np.float32)

        out = bt.stack([bt.tensor(first), bt.tensor(second)], dim=0)

        expected = np.stack([first, second], axis=0)
        self.assertEqual(out.shape, [2, 0, 3])
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_stack_rejects_empty_sequence(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"stack\(\) expected a non-empty sequence of tensors",
        ):
            _ = bt.stack([])

    def test_stack_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"tensor at position 1 has shape \[3, 2\] but expected shape \[2, 3\]",
        ):
            _ = bt.stack(
                [
                    bt.tensor(np.zeros((2, 3), dtype=np.float32)),
                    bt.tensor(np.zeros((3, 2), dtype=np.float32)),
                ]
            )

    def test_stack_rejects_dtype_mismatch(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"tensor at position 1 has dtype int64 but expected dtype float32",
        ):
            _ = bt.stack(
                [
                    bt.tensor(np.zeros((2, 3), dtype=np.float32)),
                    bt.tensor(np.zeros((2, 3), dtype=np.int64)),
                ]
            )

    def test_stack_rejects_dim_out_of_range(self) -> None:
        tensor = bt.tensor(np.zeros((2, 3), dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"dim=3 is out of range for insertion rank 3",
        ):
            _ = bt.stack([tensor, tensor], dim=3)

        with self.assertRaisesRegex(
            ValueError,
            r"dim=-4 is out of range for insertion rank 3",
        ):
            _ = bt.stack([tensor, tensor], dim=-4)

    def test_stack_backward_routes_gradients_by_position(self) -> None:
        first = bt.tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)
        second = bt.tensor(np.arange(6, 12, dtype=np.float32).reshape(2, 3), requires_grad=True)
        weight_np = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        weight = bt.tensor(weight_np)

        out = bt.stack([first, second], dim=1)
        loss = (out * weight).sum()
        loss.backward()

        np.testing.assert_allclose(
            to_numpy(_require_grad(first)),
            weight_np[:, 0, :],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            to_numpy(_require_grad(second)),
            weight_np[:, 1, :],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_stack_backward_supports_scalar_inputs(self) -> None:
        first = bt.tensor(np.asarray(1.0, dtype=np.float32), requires_grad=True)
        second = bt.tensor(np.asarray(2.0, dtype=np.float32), requires_grad=True)
        weight = bt.tensor(np.asarray([3.0, 4.0], dtype=np.float32))

        out = bt.stack([first, second])
        loss = (out * weight).sum()
        loss.backward()

        np.testing.assert_allclose(
            to_numpy(_require_grad(first)),
            np.asarray(3.0, dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            to_numpy(_require_grad(second)),
            np.asarray(4.0, dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
