import unittest

import numpy as np

import bt
from tests.utils import to_numpy


class TriangularTests(unittest.TestCase):
    def test_triu_matches_numpy_for_square_matrix(self) -> None:
        source = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
        tensor = bt.tensor(source)

        for diagonal in (0, 1, -1):
            with self.subTest(diagonal=diagonal):
                out = tensor.triu(diagonal)

                self.assertEqual(out.shape, [3, 3])
                self.assertTrue(out.is_contiguous())
                np.testing.assert_allclose(to_numpy(out), np.triu(source, k=diagonal))

    def test_tril_matches_numpy_for_rectangular_matrix(self) -> None:
        source = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
        tensor = bt.tensor(source)

        for diagonal in (0, 1, -1):
            with self.subTest(diagonal=diagonal):
                out = tensor.tril(diagonal)

                self.assertEqual(out.shape, [3, 4])
                self.assertTrue(out.is_contiguous())
                np.testing.assert_allclose(to_numpy(out), np.tril(source, k=diagonal))

    def test_batched_triu_and_tril_match_numpy_on_last_two_dims(self) -> None:
        source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        tensor = bt.tensor(source)

        upper = bt.triu(tensor, diagonal=1)
        lower = bt.tril(tensor, diagonal=-1)

        self.assertEqual(upper.shape, [2, 3, 4])
        self.assertEqual(lower.shape, [2, 3, 4])
        np.testing.assert_allclose(to_numpy(upper), np.triu(source, k=1))
        np.testing.assert_allclose(to_numpy(lower), np.tril(source, k=-1))

    def test_non_contiguous_input_produces_contiguous_output(self) -> None:
        source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        transposed_source = np.transpose(source, (0, 2, 1))
        tensor = bt.tensor(source).transpose(1, 2)

        self.assertFalse(tensor.is_contiguous())

        upper = tensor.triu(diagonal=-1)
        lower = tensor.tril(diagonal=1)

        self.assertTrue(upper.is_contiguous())
        self.assertTrue(lower.is_contiguous())
        np.testing.assert_allclose(to_numpy(upper), np.triu(transposed_source, k=-1))
        np.testing.assert_allclose(to_numpy(lower), np.tril(transposed_source, k=1))

    def test_zero_sized_matrix_dimensions_are_preserved(self) -> None:
        empty_rows = np.empty((0, 3), dtype=np.float32)
        empty_cols = np.empty((2, 0), dtype=np.float32)

        upper = bt.triu(bt.tensor(empty_rows), diagonal=1)
        lower = bt.tril(bt.tensor(empty_cols), diagonal=-1)

        self.assertEqual(upper.shape, [0, 3])
        self.assertEqual(lower.shape, [2, 0])
        np.testing.assert_allclose(to_numpy(upper), np.triu(empty_rows, k=1))
        np.testing.assert_allclose(to_numpy(lower), np.tril(empty_cols, k=-1))

    def test_int64_input_preserves_dtype_and_values(self) -> None:
        source = np.arange(12, dtype=np.int64).reshape(3, 4)

        upper = bt.triu(bt.tensor(source), diagonal=1)
        lower = bt.tril(bt.tensor(source), diagonal=-1)

        self.assertEqual(upper.dtype, bt.int64)
        self.assertEqual(lower.dtype, bt.int64)
        np.testing.assert_array_equal(to_numpy(upper), np.triu(source, k=1))
        np.testing.assert_array_equal(to_numpy(lower), np.tril(source, k=-1))

    def test_top_level_and_method_surfaces_match(self) -> None:
        source = np.arange(20, dtype=np.float32).reshape(4, 5)
        tensor = bt.tensor(source)

        from_function_upper = bt.triu(tensor, diagonal=-1)
        from_method_upper = tensor.triu(diagonal=-1)
        from_function_lower = bt.tril(tensor, diagonal=2)
        from_method_lower = tensor.tril(diagonal=2)

        np.testing.assert_allclose(to_numpy(from_function_upper), to_numpy(from_method_upper))
        np.testing.assert_allclose(to_numpy(from_function_lower), to_numpy(from_method_lower))

    def test_triu_and_tril_reject_rank_less_than_two(self) -> None:
        vector = bt.tensor(np.arange(5, dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"triu failed for tensor with shape \[5\]: expected ndim\(\) >= 2, but got 1\.",
        ):
            _ = vector.triu()

        with self.assertRaisesRegex(
            ValueError,
            r"tril failed for tensor with shape \[5\]: expected ndim\(\) >= 2, but got 1\.",
        ):
            _ = vector.tril()


if __name__ == "__main__":
    unittest.main()
