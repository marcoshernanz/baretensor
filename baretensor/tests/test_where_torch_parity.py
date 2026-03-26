import unittest
from typing import Any, cast

import numpy as np
import torch

import bt
from tests.utils import to_numpy


def _assert_allclose(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def _torch_grad_numpy(tensor: torch.Tensor) -> np.ndarray:
    grad = cast(Any, tensor).grad
    assert grad is not None
    return cast(np.ndarray, grad.detach().cpu().numpy().astype(np.float32))


class WhereTorchParityTests(unittest.TestCase):
    def test_randomized_forward_parity_across_dtypes(self) -> None:
        shape_cases = (
            ((2, 3, 1), (1, 3, 4), (2, 1, 4)),
            ((3, 1), (3, 4), (1, 4)),
            ((), (2, 3), (1, 3)),
            ((1,), (), (5,)),
        )
        dtype_cases = (
            (np.float32, torch.float32),
            (np.int64, torch.int64),
            (np.bool_, torch.bool),
        )

        for seed in range(18):
            rng = np.random.default_rng(900 + seed)
            condition_shape, input_shape, other_shape = shape_cases[seed % len(shape_cases)]
            np_dtype, torch_dtype = dtype_cases[seed % len(dtype_cases)]

            condition_np = rng.integers(0, 2, size=condition_shape, dtype=np.int64).astype(np.bool_)
            if np_dtype == np.float32:
                input_np = rng.normal(size=input_shape).astype(np.float32)
                other_np = rng.normal(size=other_shape).astype(np.float32)
            elif np_dtype == np.int64:
                input_np = rng.integers(-8, 9, size=input_shape, dtype=np.int64)
                other_np = rng.integers(-8, 9, size=other_shape, dtype=np.int64)
            else:
                input_np = rng.integers(0, 2, size=input_shape, dtype=np.int64).astype(np.bool_)
                other_np = rng.integers(0, 2, size=other_shape, dtype=np.int64).astype(np.bool_)

            bt_out = bt.where(bt.tensor(condition_np), bt.tensor(input_np), bt.tensor(other_np))
            torch_out = torch.where(
                torch.tensor(condition_np, dtype=torch.bool),
                torch.tensor(input_np, dtype=torch_dtype),
                torch.tensor(other_np, dtype=torch_dtype),
            )

            np.testing.assert_array_equal(to_numpy(bt_out), torch_out.detach().cpu().numpy())

    def test_non_contiguous_forward_parity(self) -> None:
        condition_base = np.asarray(
            [
                [[True, False], [False, True], [True, False]],
                [[False, True], [True, False], [False, True]],
                [[True, True], [False, False], [True, False]],
                [[False, False], [True, True], [False, True]],
            ],
            dtype=np.bool_,
        )
        input_base = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
        other_base = (np.arange(8, dtype=np.float32).reshape(4, 1, 2) - 5.0).astype(np.float32)

        condition_bt = bt.tensor(condition_base).permute([2, 1, 0])
        input_bt = bt.tensor(input_base).transpose(0, 1)
        other_bt = bt.tensor(other_base).permute([2, 1, 0])

        condition_torch = torch.tensor(condition_base, dtype=torch.bool).permute(2, 1, 0)
        input_torch = torch.tensor(input_base, dtype=torch.float32).transpose(0, 1)
        other_torch = torch.tensor(other_base, dtype=torch.float32).permute(2, 1, 0)

        bt_out = bt.where(condition_bt, input_bt, other_bt)
        torch_out = torch.where(condition_torch, input_torch, other_torch)

        _assert_allclose(
            to_numpy(bt_out), torch_out.detach().cpu().numpy().astype(np.float32)
        )

    def test_backward_parity_with_broadcast_and_non_contiguous_inputs(self) -> None:
        for seed in range(8):
            rng = np.random.default_rng(1000 + seed)

            condition_base = rng.integers(0, 2, size=(4, 3, 2), dtype=np.int64).astype(np.bool_)
            input_base = rng.normal(size=(3, 2, 4)).astype(np.float32)
            other_base = rng.normal(size=(4, 2, 1)).astype(np.float32)
            weight_np = rng.normal(size=(2, 3, 4)).astype(np.float32)

            condition_bt = bt.tensor(condition_base).permute([2, 1, 0])
            input_bt_base = bt.tensor(input_base, requires_grad=True)
            other_bt_base = bt.tensor(other_base, requires_grad=True)
            input_bt = input_bt_base.transpose(0, 1)
            other_bt = other_bt_base.transpose(0, 1).transpose(1, 2)
            weight_bt = bt.tensor(weight_np)

            condition_torch = torch.tensor(condition_base, dtype=torch.bool).permute(2, 1, 0)
            input_torch_base = torch.tensor(input_base, dtype=torch.float32, requires_grad=True)
            other_torch_base = torch.tensor(other_base, dtype=torch.float32, requires_grad=True)
            input_torch = input_torch_base.transpose(0, 1)
            other_torch = other_torch_base.transpose(0, 1).transpose(1, 2)
            weight_torch = torch.tensor(weight_np, dtype=torch.float32)

            bt_out = bt.where(condition_bt, input_bt, other_bt)
            torch_out = torch.where(condition_torch, input_torch, other_torch)
            bt_loss = (bt_out * weight_bt).sum()
            torch_loss = (torch_out * weight_torch).sum()

            bt_loss.backward()
            cast(Any, torch_loss).backward()

            _assert_allclose(
                to_numpy(bt_out), torch_out.detach().cpu().numpy().astype(np.float32)
            )
            _assert_allclose(to_numpy(cast(bt.Tensor, input_bt_base.grad)), _torch_grad_numpy(input_torch_base))
            _assert_allclose(to_numpy(cast(bt.Tensor, other_bt_base.grad)), _torch_grad_numpy(other_torch_base))

    def test_backward_parity_with_scalar_other(self) -> None:
        for seed in range(6):
            rng = np.random.default_rng(1100 + seed)
            input_np = rng.normal(size=(2, 3, 4)).astype(np.float32)
            condition_np = rng.integers(0, 2, size=(2, 3, 4), dtype=np.int64).astype(np.bool_)
            weight_np = rng.normal(size=(2, 3, 4)).astype(np.float32)

            input_bt = bt.tensor(input_np, requires_grad=True)
            input_torch = torch.tensor(input_np, dtype=torch.float32, requires_grad=True)
            condition_bt = bt.tensor(condition_np)
            condition_torch = torch.tensor(condition_np, dtype=torch.bool)
            weight_bt = bt.tensor(weight_np)
            weight_torch = torch.tensor(weight_np, dtype=torch.float32)

            bt_loss = (bt.where(condition_bt, input_bt, 0.0) * weight_bt).sum()
            torch_loss = (torch.where(condition_torch, input_torch, 0.0) * weight_torch).sum()

            bt_loss.backward()
            cast(Any, torch_loss).backward()

            _assert_allclose(to_numpy(cast(bt.Tensor, input_bt.grad)), _torch_grad_numpy(input_torch))


class ComparisonTorchParityTests(unittest.TestCase):
    def test_randomized_comparison_forward_parity(self) -> None:
        ops = (
            ("eq", lambda a, b: a == b, lambda a, b: a == b),
            ("ne", lambda a, b: a != b, lambda a, b: a != b),
            ("lt", lambda a, b: a < b, lambda a, b: a < b),
            ("le", lambda a, b: a <= b, lambda a, b: a <= b),
            ("gt", lambda a, b: a > b, lambda a, b: a > b),
            ("ge", lambda a, b: a >= b, lambda a, b: a >= b),
        )
        shape_cases = (((2, 3, 1), (1, 3, 4)), ((3, 1), (1, 4)), ((), (2, 3)))

        for seed in range(24):
            rng = np.random.default_rng(1200 + seed)
            _, bt_op, torch_op = ops[seed % len(ops)]
            lhs_shape, rhs_shape = shape_cases[seed % len(shape_cases)]

            if seed % 2 == 0:
                lhs_np = rng.normal(size=lhs_shape).astype(np.float32)
                rhs_np = rng.normal(size=rhs_shape).astype(np.float32)
                lhs_torch = torch.tensor(lhs_np, dtype=torch.float32)
                rhs_torch = torch.tensor(rhs_np, dtype=torch.float32)
            else:
                lhs_np = rng.integers(-9, 10, size=lhs_shape, dtype=np.int64)
                rhs_np = rng.integers(-9, 10, size=rhs_shape, dtype=np.int64)
                lhs_torch = torch.tensor(lhs_np, dtype=torch.int64)
                rhs_torch = torch.tensor(rhs_np, dtype=torch.int64)

            lhs_bt = bt.tensor(lhs_np)
            rhs_bt = bt.tensor(rhs_np)

            bt_out = bt_op(lhs_bt, rhs_bt)
            torch_out = torch_op(lhs_torch, rhs_torch)

            self.assertEqual(bt_out.dtype, bt.bool)
            np.testing.assert_array_equal(to_numpy(bt_out), torch_out.detach().cpu().numpy())

        lhs_bool = bt.tensor(np.asarray([True, False, True], dtype=np.bool_))
        rhs_bool = bt.tensor(np.asarray([False, False, True], dtype=np.bool_))
        np.testing.assert_array_equal(
            to_numpy(lhs_bool == rhs_bool),
            (torch.tensor([True, False, True]) == torch.tensor([False, False, True]))
            .detach()
            .cpu()
            .numpy(),
        )

    def test_non_contiguous_comparison_parity(self) -> None:
        lhs_base = np.arange(12, dtype=np.float32).reshape(3, 4)
        rhs_base = np.linspace(-1.0, 10.0, num=12, dtype=np.float32).reshape(3, 4)

        lhs_bt = bt.tensor(lhs_base).transpose(0, 1)
        rhs_bt = bt.tensor(rhs_base).transpose(0, 1)
        lhs_torch = torch.tensor(lhs_base, dtype=torch.float32).transpose(0, 1)
        rhs_torch = torch.tensor(rhs_base, dtype=torch.float32).transpose(0, 1)

        np.testing.assert_array_equal(
            to_numpy(lhs_bt <= rhs_bt),
            (lhs_torch <= rhs_torch).detach().cpu().numpy(),
        )
        np.testing.assert_array_equal(
            to_numpy(lhs_bt > rhs_bt),
            (lhs_torch > rhs_torch).detach().cpu().numpy(),
        )


if __name__ == "__main__":
    unittest.main()
