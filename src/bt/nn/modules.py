"""Minimal module and layer abstractions for BareTensor."""

from __future__ import annotations

import builtins
import math
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

import bt
from bt._C import Tensor

from . import functional


def _is_parameter_tensor(value: object) -> builtins.bool:
    return isinstance(value, Tensor) and value.requires_grad


def Parameter(data: Tensor | ArrayLike) -> Tensor:
    """Create a trainable floating-point tensor detached from prior history."""
    if isinstance(data, Tensor):
        parameter = data.detach()
        if parameter.dtype != bt.float32:
            parameter = parameter.to(bt.float32)
        parameter.requires_grad = True
        return parameter

    return bt.tensor(data, dtype=bt.float32, requires_grad=True)


class Module:
    """Base class for composable neural-network modules."""

    training: builtins.bool

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", dict[str, Tensor]())
        object.__setattr__(self, "_modules", dict[str, Module]())
        object.__setattr__(self, "training", True)

    def __setattr__(self, name: str, value: object) -> None:
        if name in {"_parameters", "_modules", "training"}:
            object.__setattr__(self, name, value)
            return

        parameters = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")
        if parameters is None or modules is None:
            object.__setattr__(self, name, value)
            return

        parameters.pop(name, None)
        modules.pop(name, None)

        if isinstance(value, Module):
            modules[name] = value
        elif _is_parameter_tensor(value):
            parameters[name] = value

        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        parameters = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")
        if parameters is not None:
            parameters.pop(name, None)
        if modules is not None:
            modules.pop(name, None)
        object.__delattr__(self, name)

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__}.forward() must be implemented.")

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.forward(*args, **kwargs)

    def parameters(self) -> Iterator[Tensor]:
        for parameter in self._parameters.values():
            yield parameter
        for module in self._modules.values():
            yield from module.parameters()

    def train(self, mode: builtins.bool = True) -> Module:
        object.__setattr__(self, "training", builtins.bool(mode))
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> Module:
        return self.train(False)


class Linear(Module):
    """Applies an affine transformation to the last input dimension."""

    in_features: int
    out_features: int
    weight: Tensor
    bias: Tensor | None

    def __init__(self, in_features: int, out_features: int, bias: builtins.bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(np.zeros((out_features, in_features), dtype=np.float32))
        self.bias = None
        if bias:
            self.bias = Parameter(np.zeros((out_features,), dtype=np.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight = Parameter(
            np.random.uniform(
                low=-bound,
                high=bound,
                size=(self.out_features, self.in_features),
            ).astype(np.float32)
        )
        if self.bias is not None:
            self.bias = Parameter(
                np.random.uniform(
                    low=-bound,
                    high=bound,
                    size=(self.out_features,),
                ).astype(np.float32)
            )

    def forward(self, input: Tensor) -> Tensor:
        output = input @ self.weight.transpose(0, 1)
        if self.bias is not None:
            output = output + self.bias
        return output


class Embedding(Module):
    """Embedding lookup table."""

    num_embeddings: int
    embedding_dim: int
    weight: Tensor

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(np.zeros((num_embeddings, embedding_dim), dtype=np.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight = Parameter(
            np.random.normal(
                loc=0.0,
                scale=1.0,
                size=(self.num_embeddings, self.embedding_dim),
            ).astype(np.float32)
        )

    def forward(self, input: Tensor) -> Tensor:
        return functional.embedding(input, self.weight)


class LayerNorm(Module):
    """Layer normalization over the trailing dimensions."""

    normalized_shape: tuple[int, ...]
    eps: float
    elementwise_affine: builtins.bool
    weight: Tensor | None
    bias: Tensor | None

    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        eps: float = 1e-5,
        elementwise_affine: builtins.bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = self._normalize_shape(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = None
        self.bias = None
        if self.elementwise_affine:
            self.weight = Parameter(np.ones(self.normalized_shape, dtype=np.float32))
            self.bias = Parameter(np.zeros(self.normalized_shape, dtype=np.float32))
        self.reset_parameters()

    @staticmethod
    def _normalize_shape(normalized_shape: int | Sequence[int]) -> tuple[int, ...]:
        if isinstance(normalized_shape, int):
            return (normalized_shape,)
        if not isinstance(normalized_shape, Sequence):
            raise TypeError("normalized_shape must be an int or a sequence of ints.")
        return tuple(int(dim) for dim in normalized_shape)

    def reset_parameters(self) -> None:
        if self.weight is not None:
            self.weight = Parameter(np.ones(self.normalized_shape, dtype=np.float32))
        if self.bias is not None:
            self.bias = Parameter(np.zeros(self.normalized_shape, dtype=np.float32))

    def forward(self, input: Tensor) -> Tensor:
        return functional.layer_norm(
            input,
            normalized_shape=self.normalized_shape,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )
