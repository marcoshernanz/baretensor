import unittest
from typing import Any, cast

import numpy as np

import bt
from tests.utils import to_numpy


class ConstructorsAndMetadataTests(unittest.TestCase):
    def test_full_zeros_ones_values(self) -> None:
        shape = [2, 3]

        full = bt.full(shape, 7.5)
        zeros = bt.zeros(shape)
        ones = bt.ones(shape)

        np.testing.assert_allclose(to_numpy(full), np.full(shape, 7.5, dtype=np.float32))
        np.testing.assert_allclose(to_numpy(zeros), np.zeros(shape, dtype=np.float32))
        np.testing.assert_allclose(to_numpy(ones), np.ones(shape, dtype=np.float32))
        self.assertEqual(full.dtype, bt.float32)
        self.assertEqual(zeros.dtype, bt.float32)
        self.assertEqual(ones.dtype, bt.float32)

    def test_tensor_from_numpy_copies_data(self) -> None:
        source = np.arange(6, dtype=np.float32).reshape(2, 3)
        tensor = bt.tensor(source)

        source[0, 0] = 999.0
        expected = np.arange(6, dtype=np.float32).reshape(2, 3)
        np.testing.assert_allclose(to_numpy(tensor), expected)

    def test_tensor_accepts_python_lists(self) -> None:
        tensor = bt.tensor([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(tensor.shape, [2, 3])
        self.assertEqual(tensor.dtype, bt.int64)
        np.testing.assert_array_equal(
            to_numpy(tensor),
            np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
        )

    def test_tensor_accepts_python_scalar(self) -> None:
        tensor = bt.tensor(3.5)

        self.assertEqual(tensor.shape, [])
        self.assertEqual(tensor.dtype, bt.float32)
        np.testing.assert_allclose(to_numpy(tensor), np.asarray(3.5, dtype=np.float32))

    def test_tensor_accepts_python_int_scalar_as_int64(self) -> None:
        tensor = bt.tensor(7)

        self.assertEqual(tensor.shape, [])
        self.assertEqual(tensor.dtype, bt.int64)
        self.assertEqual(to_numpy(tensor).dtype, np.int64)
        self.assertEqual(cast(int, tensor.item()), 7)

    def test_scalar_tensor_metadata(self) -> None:
        tensor = bt.tensor(np.asarray(3.5, dtype=np.float32))

        self.assertEqual(tensor.shape, [])
        self.assertEqual(tensor.strides, [])
        self.assertEqual(tensor.ndim(), 0)
        self.assertEqual(tensor.numel(), 1)
        self.assertTrue(tensor.is_contiguous())
        self.assertEqual(tensor.dtype, bt.float32)
        np.testing.assert_allclose(to_numpy(tensor), np.asarray(3.5, dtype=np.float32))
        self.assertAlmostEqual(cast(float, tensor.item()), 3.5)

    def test_item_accepts_any_tensor_with_single_element(self) -> None:
        tensor = bt.full([1, 1], 2.25)

        self.assertAlmostEqual(cast(float, tensor.item()), 2.25)

    def test_item_rejects_tensors_with_more_than_one_element(self) -> None:
        tensor = bt.zeros([2, 3])

        with self.assertRaisesRegex(
            ValueError,
            r"item\(\) can only be called on tensors with exactly one element",
        ):
            _ = tensor.item()

    def test_zero_dim_extent_metadata(self) -> None:
        tensor = bt.zeros([0, 3])

        self.assertEqual(tensor.shape, [0, 3])
        self.assertEqual(tensor.strides, [3, 1])
        self.assertEqual(tensor.ndim(), 2)
        self.assertEqual(tensor.numel(), 0)
        self.assertTrue(tensor.is_contiguous())
        np.testing.assert_allclose(to_numpy(tensor), np.zeros((0, 3), dtype=np.float32))

    def test_factory_accepts_explicit_int64_dtype(self) -> None:
        tensor = bt.full([2, 2], 3, dtype=bt.int64)

        self.assertEqual(tensor.dtype, bt.int64)
        np.testing.assert_array_equal(to_numpy(tensor), np.full((2, 2), 3, dtype=np.int64))

    def test_factory_rejects_requires_grad_on_int64(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"set_requires_grad\(true\) is only supported for floating-point tensors",
        ):
            _ = bt.zeros([2, 2], dtype=bt.int64, requires_grad=True)

    def test_negative_shape_raises(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"Invalid tensor shape \[2, -1\]: dimension 1 has negative size -1\.",
        ):
            _ = bt.zeros([2, -1])

        with self.assertRaisesRegex(
            ValueError,
            r"Invalid tensor shape \[-3\]: dimension 0 has negative size -3\.",
        ):
            _ = bt.full([-3], 1.0)

    def test_tensor_rejects_unsupported_numpy_dtype_without_explicit_dtype(self) -> None:
        float64 = np.arange(6, dtype=np.float64).reshape(2, 3)

        with self.assertRaisesRegex(
            TypeError,
            r"Unsupported NumPy dtype float64",
        ):
            _ = bt.tensor(cast(Any, float64))

    def test_tensor_accepts_non_contiguous_supported_numpy(self) -> None:
        non_contiguous = np.arange(6, dtype=np.int64).reshape(3, 2).T
        self.assertFalse(non_contiguous.flags["C_CONTIGUOUS"])

        tensor = bt.tensor(cast(Any, non_contiguous))

        self.assertEqual(tensor.dtype, bt.int64)
        np.testing.assert_array_equal(to_numpy(tensor), np.asarray(non_contiguous, dtype=np.int64))

    def test_tensor_explicit_dtype_casts_supported_numpy(self) -> None:
        float64 = np.arange(6, dtype=np.float64).reshape(2, 3)

        tensor = bt.tensor(cast(Any, float64), dtype=bt.float32)

        self.assertEqual(tensor.dtype, bt.float32)
        np.testing.assert_allclose(to_numpy(tensor), np.asarray(float64, dtype=np.float32))

    def test_tensor_list_respects_requires_grad(self) -> None:
        tensor = bt.tensor([1.0, 2.0, 3.0], requires_grad=True)

        self.assertTrue(tensor.requires_grad)
        self.assertEqual(tensor.dtype, bt.float32)

    def test_requires_grad_rejects_int64_tensor(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"set_requires_grad\(true\) is only supported for floating-point tensors",
        ):
            _ = bt.tensor([1, 2, 3], requires_grad=True)

    def test_explicit_int64_cast_is_strict(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"requires integer-valued floats",
        ):
            _ = bt.tensor([1.0, 2.5], dtype=bt.int64)

    def test_tensor_to_preserves_and_converts_dtype(self) -> None:
        source = bt.tensor([1, 2, 3])
        same = source.to(bt.int64)
        converted = source.to(bt.float32)

        self.assertEqual(same.dtype, bt.int64)
        self.assertEqual(converted.dtype, bt.float32)
        np.testing.assert_array_equal(to_numpy(same), np.asarray([1, 2, 3], dtype=np.int64))
        np.testing.assert_allclose(
            to_numpy(converted), np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        )


if __name__ == "__main__":
    unittest.main()
