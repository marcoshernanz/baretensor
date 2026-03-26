import unittest

import numpy as np

import bt
from tests.utils import to_numpy


class ReshapeTests(unittest.TestCase):
    def test_view_allows_scalar_to_numel_one_shape(self) -> None:
        tensor = bt.tensor(np.asarray(3.5, dtype=np.float32))

        viewed = tensor.view([1, 1])

        self.assertEqual(viewed.shape, [1, 1])
        np.testing.assert_allclose(to_numpy(viewed), np.asarray([[3.5]], dtype=np.float32))

    def test_reshape_allows_scalar_to_numel_one_shape(self) -> None:
        tensor = bt.tensor(np.asarray(3.5, dtype=np.float32))

        reshaped = tensor.reshape([1, 1])

        self.assertEqual(reshaped.shape, [1, 1])
        np.testing.assert_allclose(to_numpy(reshaped), np.asarray([[3.5]], dtype=np.float32))

    def test_view_preserves_values(self) -> None:
        source = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor = bt.tensor(source)

        viewed = tensor.view([2, 6])

        self.assertEqual(viewed.shape, [2, 6])
        np.testing.assert_allclose(to_numpy(viewed), source.reshape(2, 6))

    def test_view_accepts_tuple_shape(self) -> None:
        source = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor = bt.tensor(source)

        viewed_tuple = tensor.view((2, 6))

        np.testing.assert_allclose(to_numpy(viewed_tuple), source.reshape(2, 6))

    def test_view_infers_single_negative_dimension(self) -> None:
        source = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor = bt.tensor(source)

        viewed = tensor.view([-1, 3])

        self.assertEqual(viewed.shape, [4, 3])
        np.testing.assert_allclose(to_numpy(viewed), source.reshape(4, 3))

    def test_view_rejects_mismatched_numel(self) -> None:
        tensor = bt.tensor(np.arange(6, dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"element counts differ",
        ):
            _ = tensor.view([4, 2])

    def test_reshape_matches_view_for_viewable_shape(self) -> None:
        source = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor = bt.tensor(source)

        reshaped = tensor.reshape([2, 6])
        viewed = tensor.view([2, 6])

        np.testing.assert_allclose(to_numpy(reshaped), to_numpy(viewed))

    def test_reshape_preserves_values(self) -> None:
        source = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor = bt.tensor(source)

        reshaped = tensor.reshape([2, 6])

        self.assertEqual(reshaped.shape, [2, 6])
        np.testing.assert_allclose(to_numpy(reshaped), source.reshape(2, 6))

    def test_reshape_accepts_tuple_shape(self) -> None:
        source = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor = bt.tensor(source)

        reshaped_tuple = tensor.reshape((2, 6))

        np.testing.assert_allclose(to_numpy(reshaped_tuple), source.reshape(2, 6))

    def test_reshape_infers_single_negative_dimension(self) -> None:
        source = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor = bt.tensor(source)

        reshaped = tensor.reshape([-1, 2])

        self.assertEqual(reshaped.shape, [6, 2])
        np.testing.assert_allclose(to_numpy(reshaped), source.reshape(6, 2))

    def test_reshape_rejects_multiple_negative_one_dimensions(self) -> None:
        tensor = bt.tensor(np.arange(6, dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"at most one '-1' dimension is allowed",
        ):
            _ = tensor.reshape([-1, -1])

    def test_reshape_rejects_invalid_negative_dimension(self) -> None:
        tensor = bt.tensor(np.arange(6, dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"has invalid size -2",
        ):
            _ = tensor.reshape([3, -2])

    def test_reshape_rejects_mismatched_numel(self) -> None:
        tensor = bt.tensor(np.arange(6, dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"element counts differ",
        ):
            _ = tensor.reshape([4, 2])

    def test_reshape_rejects_ambiguous_zero_product_inference(self) -> None:
        tensor = bt.zeros([0, 4])

        with self.assertRaisesRegex(
            ValueError,
            r"cannot infer '-1' when known dimensions multiply to zero",
        ):
            _ = tensor.reshape([-1, 0])

    def test_flatten_default_preserves_values(self) -> None:
        source = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor = bt.tensor(source)

        flattened = tensor.flatten()

        self.assertEqual(flattened.shape, [12])
        np.testing.assert_allclose(to_numpy(flattened), source.reshape(12))

    def test_flatten_over_subrange_preserves_outer_dimensions(self) -> None:
        source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        tensor = bt.tensor(source)

        flattened = tensor.flatten(1, 2)

        self.assertEqual(flattened.shape, [2, 12])
        np.testing.assert_allclose(to_numpy(flattened), source.reshape(2, 12))

    def test_flatten_on_scalar_returns_length_one_tensor(self) -> None:
        tensor = bt.tensor(np.asarray(3.5, dtype=np.float32))

        flattened = tensor.flatten()

        self.assertEqual(flattened.shape, [1])
        np.testing.assert_allclose(to_numpy(flattened), np.asarray([3.5], dtype=np.float32))

    def test_flatten_accepts_negative_dimensions(self) -> None:
        source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        tensor = bt.tensor(source)

        flattened = tensor.flatten(-2, -1)

        self.assertEqual(flattened.shape, [2, 12])
        np.testing.assert_allclose(to_numpy(flattened), source.reshape(2, 12))

    def test_flatten_rejects_start_dim_after_end_dim(self) -> None:
        tensor = bt.tensor(np.arange(12, dtype=np.float32).reshape(3, 4))

        with self.assertRaisesRegex(
            ValueError,
            r"start_dim=1 cannot come after end_dim=0",
        ):
            _ = tensor.flatten(1, 0)

    def test_unsqueeze_inserts_leading_dimension(self) -> None:
        source = np.arange(6, dtype=np.float32).reshape(2, 3)
        tensor = bt.tensor(source)

        expanded = tensor.unsqueeze(0)

        self.assertEqual(expanded.shape, [1, 2, 3])
        np.testing.assert_allclose(to_numpy(expanded), np.expand_dims(source, axis=0))

    def test_unsqueeze_inserts_middle_dimension(self) -> None:
        source = np.arange(6, dtype=np.float32).reshape(2, 3)
        tensor = bt.tensor(source)

        expanded = tensor.unsqueeze(1)

        self.assertEqual(expanded.shape, [2, 1, 3])
        np.testing.assert_allclose(to_numpy(expanded), np.expand_dims(source, axis=1))

    def test_unsqueeze_accepts_negative_dimension(self) -> None:
        source = np.arange(6, dtype=np.float32).reshape(2, 3)
        tensor = bt.tensor(source)

        expanded = tensor.unsqueeze(-1)

        self.assertEqual(expanded.shape, [2, 3, 1])
        np.testing.assert_allclose(to_numpy(expanded), np.expand_dims(source, axis=-1))

    def test_unsqueeze_supports_scalar_tensor(self) -> None:
        tensor = bt.tensor(np.asarray(3.5, dtype=np.float32))

        expanded = tensor.unsqueeze(0)

        self.assertEqual(expanded.shape, [1])
        np.testing.assert_allclose(to_numpy(expanded), np.asarray([3.5], dtype=np.float32))

    def test_unsqueeze_supports_non_contiguous_input(self) -> None:
        source = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor = bt.tensor(source).transpose(0, 1)

        expanded = tensor.unsqueeze(1)

        self.assertEqual(expanded.shape, [4, 1, 3])
        np.testing.assert_allclose(to_numpy(expanded), np.expand_dims(source.T, axis=1))

    def test_unsqueeze_rejects_out_of_range_dimension(self) -> None:
        tensor = bt.tensor(np.arange(6, dtype=np.float32).reshape(2, 3))

        with self.assertRaisesRegex(
            ValueError,
            r"dim=3 is out of range for insertion rank 3",
        ):
            _ = tensor.unsqueeze(3)

        with self.assertRaisesRegex(
            ValueError,
            r"dim=-4 is out of range for insertion rank 3",
        ):
            _ = tensor.unsqueeze(-4)


if __name__ == "__main__":
    unittest.main()
