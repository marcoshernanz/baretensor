import unittest
from typing import Any, cast

import numpy as np

import bt
import bt.nn.functional as F
from tests.utils import to_numpy


def _layer_norm_expected(
    source: np.ndarray,
    normalized_shape: tuple[int, ...],
    weight: np.ndarray | None = None,
    bias: np.ndarray | None = None,
    eps: float = 1e-5,
) -> np.ndarray:
    axis = tuple(range(source.ndim - len(normalized_shape), source.ndim))
    mean = np.mean(source, axis=axis, keepdims=True)
    variance = np.mean(np.square(source - mean), axis=axis, keepdims=True)
    out = (source - mean) / np.sqrt(variance + eps)

    if weight is not None:
        affine_shape = (1,) * (source.ndim - len(normalized_shape)) + normalized_shape
        out = out * weight.reshape(affine_shape)
    if bias is not None:
        affine_shape = (1,) * (source.ndim - len(normalized_shape)) + normalized_shape
        out = out + bias.reshape(affine_shape)

    return np.asarray(out, dtype=np.float32)


class LayerNormTests(unittest.TestCase):
    def test_layer_norm_matches_numpy_without_affine(self) -> None:
        source = np.asarray(np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4), dtype=np.float32)

        out = F.layer_norm(bt.tensor(source), normalized_shape=4)

        expected = _layer_norm_expected(source, normalized_shape=(4,))
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-5, atol=1e-6)

    def test_layer_norm_matches_numpy_with_affine(self) -> None:
        source = np.asarray(
            np.linspace(-3.0, 2.0, num=2 * 5 * 6, dtype=np.float32).reshape(2, 5, 6),
            dtype=np.float32,
        )
        weight = np.asarray(np.linspace(0.5, 1.5, num=6, dtype=np.float32), dtype=np.float32)
        bias = np.asarray(np.linspace(-1.0, 1.0, num=6, dtype=np.float32), dtype=np.float32)

        out = F.layer_norm(
            bt.tensor(source),
            normalized_shape=(6,),
            weight=bt.tensor(weight),
            bias=bt.tensor(bias),
            eps=1e-4,
        )

        expected = _layer_norm_expected(source, normalized_shape=(6,), weight=weight, bias=bias, eps=1e-4)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-5, atol=1e-6)

    def test_layer_norm_matches_numpy_for_multi_dim_normalized_shape(self) -> None:
        source = np.asarray(
            np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4),
            dtype=np.float32,
        )
        weight = np.asarray(np.linspace(0.8, 1.2, num=2 * 4, dtype=np.float32).reshape(2, 4), dtype=np.float32)
        bias = np.asarray(np.linspace(-0.2, 0.2, num=2 * 4, dtype=np.float32).reshape(2, 4), dtype=np.float32)

        out = F.layer_norm(
            bt.tensor(source),
            normalized_shape=(2, 4),
            weight=bt.tensor(weight),
            bias=bt.tensor(bias),
        )

        expected = _layer_norm_expected(source, normalized_shape=(2, 4), weight=weight, bias=bias)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-5, atol=1e-6)

    def test_layer_norm_supports_non_contiguous_input_and_affine(self) -> None:
        source = np.asarray(np.arange(5 * 3 * 2, dtype=np.float32).reshape(5, 3, 2), dtype=np.float32)
        input_tensor = bt.tensor(source).transpose(1, 2)

        weight_source = np.asarray(np.linspace(0.5, 1.5, num=3 * 2, dtype=np.float32).reshape(3, 2), dtype=np.float32)
        bias_source = np.asarray(np.linspace(-0.3, 0.3, num=3 * 2, dtype=np.float32).reshape(3, 2), dtype=np.float32)
        weight = bt.tensor(weight_source).transpose(0, 1)
        bias = bt.tensor(bias_source).transpose(0, 1)

        out = F.layer_norm(input_tensor, normalized_shape=(2, 3), weight=weight, bias=bias)

        expected_input = np.transpose(source, (0, 2, 1))
        expected_weight = np.transpose(weight_source, (1, 0))
        expected_bias = np.transpose(bias_source, (1, 0))
        expected = _layer_norm_expected(
            expected_input,
            normalized_shape=(2, 3),
            weight=expected_weight,
            bias=expected_bias,
        )
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-5, atol=1e-6)

    def test_layer_norm_empty_batch_dimension_returns_empty_output(self) -> None:
        source = np.asarray(np.zeros((0, 4, 6), dtype=np.float32), dtype=np.float32)

        out = F.layer_norm(bt.tensor(source), normalized_shape=6)

        self.assertEqual(out.shape, [0, 4, 6])
        expected = _layer_norm_expected(source, normalized_shape=(6,))
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-5, atol=1e-6)

    def test_layer_norm_rejects_invalid_normalized_shape_type(self) -> None:
        tensor = bt.tensor(np.asarray(np.zeros((2, 3), dtype=np.float32), dtype=np.float32))

        with self.assertRaisesRegex(
            TypeError,
            r"layer_norm\(\) expected 'normalized_shape' to be an int or a sequence of ints\.",
        ):
            _ = F.layer_norm(tensor, normalized_shape=cast(Any, 1.5))

    def test_layer_norm_rejects_normalized_shape_rank_larger_than_input_rank(self) -> None:
        tensor = bt.tensor(np.asarray(np.zeros((2, 3), dtype=np.float32), dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"layer_norm failed for input shape \[2, 3\] and normalized_shape \[2, 3, 4\]: "
            r"normalized_shape rank must be <= input rank, got 3 and 2\.",
        ):
            _ = F.layer_norm(tensor, normalized_shape=(2, 3, 4))

    def test_layer_norm_rejects_trailing_shape_mismatch(self) -> None:
        tensor = bt.tensor(np.asarray(np.zeros((2, 3, 4), dtype=np.float32), dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"layer_norm failed for input shape \[2, 3, 4\] and normalized_shape \[3, 5\]: "
            r"input tail dimensions \[3, 4\] must match normalized_shape\.",
        ):
            _ = F.layer_norm(tensor, normalized_shape=(3, 5))

    def test_layer_norm_rejects_weight_shape_mismatch(self) -> None:
        tensor = bt.tensor(np.asarray(np.zeros((2, 3, 4), dtype=np.float32), dtype=np.float32))
        weight = bt.tensor(np.asarray(np.zeros((4, 3), dtype=np.float32), dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"layer_norm failed for input shape \[2, 3, 4\] and normalized_shape \[3, 4\]: "
            r"weight shape \[4, 3\] must match normalized_shape\.",
        ):
            _ = F.layer_norm(tensor, normalized_shape=(3, 4), weight=weight)

    def test_layer_norm_rejects_non_positive_eps(self) -> None:
        tensor = bt.tensor(np.asarray(np.zeros((2, 3), dtype=np.float32), dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"layer_norm failed for input shape \[2, 3\] and normalized_shape \[3\]: "
            r"eps must be a finite value > 0, got 0\.",
        ):
            _ = F.layer_norm(tensor, normalized_shape=3, eps=0.0)


if __name__ == "__main__":
    unittest.main()
