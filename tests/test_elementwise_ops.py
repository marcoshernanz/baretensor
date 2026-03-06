import unittest
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

import bt
from tests.utils import to_numpy

ArrayF32: TypeAlias = NDArray[np.float32]
Shape: TypeAlias = tuple[int, ...]


def _op_names() -> tuple[str, str, str, str]:
    return ("add", "sub", "mul", "div")


def _apply_bt_op(name: str, a: bt.Tensor, b: bt.Tensor) -> bt.Tensor:
    if name == "add":
        return a + b
    if name == "sub":
        return a - b
    if name == "mul":
        return a * b
    if name == "div":
        return a / b
    raise ValueError(f"Unknown op: {name}")


def _apply_bt_scalar_op(name: str, a: bt.Tensor, scalar: float) -> bt.Tensor:
    if name == "add":
        return a + scalar
    if name == "sub":
        return a - scalar
    if name == "mul":
        return a * scalar
    if name == "div":
        return a / scalar
    raise ValueError(f"Unknown op: {name}")


def _apply_bt_reverse_scalar_op(name: str, scalar: float, a: bt.Tensor) -> bt.Tensor:
    if name == "add":
        return scalar + a
    if name == "sub":
        return scalar - a
    if name == "mul":
        return scalar * a
    if name == "div":
        return scalar / a
    raise ValueError(f"Unknown op: {name}")


def _apply_np_op(name: str, a: ArrayF32, b: ArrayF32) -> ArrayF32:
    if name == "add":
        return np.asarray(a + b, dtype=np.float32)
    if name == "sub":
        return np.asarray(a - b, dtype=np.float32)
    if name == "mul":
        return np.asarray(a * b, dtype=np.float32)
    if name == "div":
        return np.asarray(a / b, dtype=np.float32)
    raise ValueError(f"Unknown op: {name}")


def _make_arange(shape: Shape, *, start: float = 0.0) -> ArrayF32:
    n = int(np.prod(shape, dtype=np.int64)) if shape else 1
    return np.asarray(
        np.arange(start, start + n, dtype=np.float32).reshape(shape), dtype=np.float32
    )


class ElementwiseOpsTests(unittest.TestCase):
    def test_unary_neg_matches_numpy(self) -> None:
        a_np: ArrayF32 = np.asarray(np.arange(12, dtype=np.float32).reshape(3, 4), dtype=np.float32)
        a = bt.tensor(a_np)

        out = -a

        np.testing.assert_allclose(
            to_numpy(out), np.asarray(-a_np, dtype=np.float32), rtol=1e-6, atol=1e-6
        )

    def test_tensor_tensor_same_shape(self) -> None:
        a_np: ArrayF32 = np.asarray(np.arange(12, dtype=np.float32).reshape(3, 4), dtype=np.float32)
        b_np: ArrayF32 = np.asarray(
            np.linspace(1.0, 2.0, num=12, dtype=np.float32).reshape(3, 4), dtype=np.float32
        )
        a = bt.tensor(a_np)
        b = bt.tensor(b_np)

        for name in _op_names():
            with self.subTest(op=name):
                out = _apply_bt_op(name, a, b)
                np.testing.assert_allclose(
                    to_numpy(out), _apply_np_op(name, a_np, b_np), rtol=1e-6, atol=1e-6
                )

    def test_tensor_scalar(self) -> None:
        a_np: ArrayF32 = np.asarray(np.arange(12, dtype=np.float32).reshape(3, 4), dtype=np.float32)
        a = bt.tensor(a_np)
        scalar = 2.5

        for name in _op_names():
            with self.subTest(op=name):
                out = _apply_bt_scalar_op(name, a, scalar)
                scalar_array: ArrayF32 = np.asarray(np.float32(scalar))
                np.testing.assert_allclose(
                    to_numpy(out),
                    _apply_np_op(name, a_np, scalar_array),
                    rtol=1e-6,
                    atol=1e-6,
                )

    def test_reverse_scalar_tensor(self) -> None:
        a_np: ArrayF32 = np.asarray(
            np.linspace(1.0, 12.0, num=12, dtype=np.float32).reshape(3, 4), dtype=np.float32
        )
        a = bt.tensor(a_np)
        scalar = 2.5

        for name in _op_names():
            with self.subTest(op=name):
                out = _apply_bt_reverse_scalar_op(name, scalar, a)
                scalar_array: ArrayF32 = np.asarray(np.float32(scalar))
                np.testing.assert_allclose(
                    to_numpy(out),
                    _apply_np_op(name, scalar_array, a_np),
                    rtol=1e-6,
                    atol=1e-6,
                )

    def test_tensor_tensor_broadcast(self) -> None:
        cases: list[tuple[Shape, Shape]] = [
            ((3, 1, 5), (1, 4, 5)),
            ((5,), (2, 3, 5)),
            ((), (2, 3)),
            ((0, 3), (1, 3)),
            ((4, 1, 2), (2,)),
        ]

        for shape_a, shape_b in cases:
            a_np = _make_arange(shape_a)
            b_np = _make_arange(shape_b, start=1.0)
            a = bt.tensor(a_np)
            b = bt.tensor(b_np)

            for name in _op_names():
                with self.subTest(op=name, shape_a=shape_a, shape_b=shape_b):
                    out = _apply_bt_op(name, a, b)
                    np.testing.assert_allclose(
                        to_numpy(out), _apply_np_op(name, a_np, b_np), rtol=1e-6, atol=1e-6
                    )

    def test_incompatible_shapes_raise(self) -> None:
        bad_cases: list[tuple[Shape, Shape]] = [
            ((2, 3), (4, 3)),
            ((0, 3), (2, 3)),
            ((3, 2, 5), (4, 5)),
        ]

        for shape_a, shape_b in bad_cases:
            a_np: ArrayF32 = np.asarray(np.zeros(shape_a, dtype=np.float32), dtype=np.float32)
            b_np: ArrayF32 = np.asarray(np.zeros(shape_b, dtype=np.float32), dtype=np.float32)
            a = bt.tensor(a_np)
            b = bt.tensor(b_np)

            for name in _op_names():
                with self.subTest(op=name, shape_a=shape_a, shape_b=shape_b):
                    with self.assertRaises(ValueError):
                        _ = _apply_bt_op(name, a, b)

    def test_incompatible_shapes_error_message_contains_context(self) -> None:
        a = bt.tensor(np.asarray(np.zeros((2, 3), dtype=np.float32), dtype=np.float32))
        b = bt.tensor(np.asarray(np.zeros((4, 3), dtype=np.float32), dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            r"Cannot broadcast shapes \[2, 3\] and \[4, 3\]: incompatible dimension",
        ):
            _ = a + b


if __name__ == "__main__":
    unittest.main()
