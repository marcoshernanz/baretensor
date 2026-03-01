import unittest
from typing import Any, cast

import numpy as np
import torch

import bt
from tests.utils import to_numpy


def _require_grad(tensor: bt.Tensor) -> bt.Tensor:
    grad = tensor.grad
    assert grad is not None
    return grad


def _assert_allclose(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def _torch_grad_numpy(tensor: torch.Tensor) -> np.ndarray:
    grad = cast(Any, tensor).grad
    assert grad is not None
    return cast(np.ndarray, grad.detach().cpu().numpy().astype(np.float32))


class AutogradTorchParityTests(unittest.TestCase):
    def test_randomized_elementwise_and_reduction_parity(self) -> None:
        ops = ("add", "sub", "mul", "div")
        reductions = ("sum", "mean", "max")

        for seed in range(8):
            rng = np.random.default_rng(seed)
            op = ops[seed % len(ops)]
            reduction = reductions[seed % len(reductions)]

            a_np = rng.normal(size=(2, 3, 4)).astype(np.float32)
            b_np = rng.normal(size=(1, 3, 1)).astype(np.float32)
            if op == "div":
                b_np = b_np + np.where(b_np >= 0.0, 1.5, -1.5).astype(np.float32)

            a_bt = bt.tensor(a_np, requires_grad=True)
            b_bt = bt.tensor(b_np, requires_grad=True)
            a_torch = torch.tensor(a_np, dtype=torch.float32, requires_grad=True)
            b_torch = torch.tensor(b_np, dtype=torch.float32, requires_grad=True)

            if op == "add":
                out_bt = a_bt + b_bt
                out_torch = a_torch + b_torch
            elif op == "sub":
                out_bt = a_bt - b_bt
                out_torch = a_torch - b_torch
            elif op == "mul":
                out_bt = a_bt * b_bt
                out_torch = a_torch * b_torch
            else:
                out_bt = a_bt / b_bt
                out_torch = a_torch / b_torch

            if reduction == "sum":
                reduced_bt = out_bt.sum(2, keepdim=False)
                reduced_torch = out_torch.sum(dim=2, keepdim=False)
            elif reduction == "mean":
                reduced_bt = out_bt.mean(0, keepdim=False)
                reduced_torch = out_torch.mean(dim=0, keepdim=False)
            else:
                # Add tiny deterministic offsets to avoid ties.
                tie_break = np.linspace(0.0, 1e-4, num=out_torch.numel(), dtype=np.float32).reshape(
                    tuple(out_torch.shape)
                )
                out_bt = out_bt + bt.tensor(tie_break)
                out_torch = out_torch + torch.tensor(tie_break, dtype=torch.float32)
                reduced_bt = out_bt.max(1, keepdim=False)
                reduced_torch = out_torch.max(dim=1, keepdim=False).values

            w_np = rng.normal(size=tuple(reduced_torch.shape)).astype(np.float32)
            w_bt = bt.tensor(w_np)
            w_torch = torch.tensor(w_np, dtype=torch.float32)

            loss_bt = (reduced_bt * w_bt).sum()
            loss_torch = (reduced_torch * w_torch).sum()

            loss_bt.backward()
            cast(Any, loss_torch).backward()

            _assert_allclose(to_numpy(loss_bt), loss_torch.detach().cpu().numpy().astype(np.float32))
            _assert_allclose(to_numpy(_require_grad(a_bt)), _torch_grad_numpy(a_torch))
            _assert_allclose(to_numpy(_require_grad(b_bt)), _torch_grad_numpy(b_torch))

    def test_randomized_matmul_softmax_log_softmax_parity(self) -> None:
        matmul_shapes = (
            ((2, 3), (3, 4)),
            ((4, 2, 3), (3, 5)),
            ((1, 7, 3, 4), (7, 4, 6)),
            ((3,), (3, 2)),
            ((2, 3), (3,)),
        )

        for seed, (a_shape, b_shape) in enumerate(matmul_shapes):
            rng = np.random.default_rng(100 + seed)
            a_np = rng.normal(size=a_shape).astype(np.float32)
            b_np = rng.normal(size=b_shape).astype(np.float32)

            a_bt = bt.tensor(a_np, requires_grad=True)
            b_bt = bt.tensor(b_np, requires_grad=True)
            a_torch = torch.tensor(a_np, dtype=torch.float32, requires_grad=True)
            b_torch = torch.tensor(b_np, dtype=torch.float32, requires_grad=True)

            out_bt = a_bt.matmul(b_bt)
            out_torch = a_torch @ b_torch
            g_np = rng.normal(size=tuple(out_torch.shape)).astype(np.float32)
            g_bt = bt.tensor(g_np)
            g_torch = torch.tensor(g_np, dtype=torch.float32)

            loss_bt = (out_bt * g_bt).sum()
            loss_torch = (out_torch * g_torch).sum()
            loss_bt.backward()
            cast(Any, loss_torch).backward()

            _assert_allclose(to_numpy(out_bt), out_torch.detach().cpu().numpy().astype(np.float32))
            _assert_allclose(to_numpy(_require_grad(a_bt)), _torch_grad_numpy(a_torch))
            _assert_allclose(to_numpy(_require_grad(b_bt)), _torch_grad_numpy(b_torch))

        for seed in range(6):
            rng = np.random.default_rng(200 + seed)
            x_np = rng.normal(size=(2, 3, 4)).astype(np.float32)
            dim = seed % 3

            x_soft_bt = bt.tensor(x_np, requires_grad=True)
            x_soft_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
            w_soft_np = rng.normal(size=(2, 3, 4)).astype(np.float32)
            w_soft_bt = bt.tensor(w_soft_np)
            w_soft_torch = torch.tensor(w_soft_np, dtype=torch.float32)

            loss_soft_bt = (x_soft_bt.softmax(dim) * w_soft_bt).sum()
            loss_soft_torch = (torch.softmax(x_soft_torch, dim=dim) * w_soft_torch).sum()
            loss_soft_bt.backward()
            cast(Any, loss_soft_torch).backward()

            _assert_allclose(
                to_numpy(_require_grad(x_soft_bt)),
                _torch_grad_numpy(x_soft_torch),
            )

            x_log_bt = bt.tensor(x_np, requires_grad=True)
            x_log_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
            w_log_np = rng.normal(size=(2, 3, 4)).astype(np.float32)
            w_log_bt = bt.tensor(w_log_np)
            w_log_torch = torch.tensor(w_log_np, dtype=torch.float32)

            loss_log_bt = (x_log_bt.log_softmax(dim) * w_log_bt).sum()
            loss_log_torch = (torch.log_softmax(x_log_torch, dim=dim) * w_log_torch).sum()
            loss_log_bt.backward()
            cast(Any, loss_log_torch).backward()

            _assert_allclose(
                to_numpy(_require_grad(x_log_bt)),
                _torch_grad_numpy(x_log_torch),
            )


if __name__ == "__main__":
    unittest.main()
