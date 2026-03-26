from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flax import nnx
from flax import serialization
import jax
import numpy as np


@dataclass(frozen=True, slots=True)
class CheckpointState:
    next_step: int
    best_validation_loss: float | None
    ema_train_loss: float | None
    batch_rng: jax.Array
    status: str


def save_checkpoint(
    path: Path,
    *,
    model: nnx.Module,
    optimizer: nnx.Optimizer[Any],
    state: CheckpointState,
) -> None:
    payload = {
        "runner": {
            "next_step": state.next_step,
            "best_validation_loss": state.best_validation_loss,
            "ema_train_loss": state.ema_train_loss,
            "batch_rng": np.asarray(jax.device_get(jax.random.key_data(state.batch_rng))),
            "status": state.status,
        },
        "model": _stringify_int_keys(nnx.to_pure_dict(nnx.state(model))),
        "optimizer": _stringify_int_keys(nnx.to_pure_dict(nnx.state(optimizer))),
    }
    path.write_bytes(serialization.msgpack_serialize(payload))


def load_checkpoint(
    path: Path,
    *,
    model: nnx.Module,
    optimizer: nnx.Optimizer[Any],
) -> CheckpointState:
    payload = serialization.msgpack_restore(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint at {path} is not a valid mapping payload.")

    model_state = nnx.state(model)
    optimizer_state = nnx.state(optimizer)
    nnx.replace_by_pure_dict(model_state, nnx.restore_int_paths(_require_dict(payload, "model")))
    nnx.replace_by_pure_dict(
        optimizer_state,
        nnx.restore_int_paths(_require_dict(payload, "optimizer")),
    )

    runner_payload = _require_dict(payload, "runner")
    batch_rng = jax.random.wrap_key_data(
        jax.numpy.asarray(runner_payload["batch_rng"], dtype=jax.numpy.uint32)
    )
    return CheckpointState(
        next_step=int(runner_payload["next_step"]),
        best_validation_loss=(
            None
            if runner_payload["best_validation_loss"] is None
            else float(runner_payload["best_validation_loss"])
        ),
        ema_train_loss=(
            None if runner_payload["ema_train_loss"] is None else float(runner_payload["ema_train_loss"])
        ),
        batch_rng=batch_rng,
        status=str(runner_payload["status"]),
    )


def _require_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint payload is missing mapping field {key!r}.")
    return value


def _stringify_int_keys(value: Any) -> Any:
    if isinstance(value, dict):
        converted: dict[str, Any] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str | int):
                raise TypeError(f"Unsupported checkpoint key type: {type(key)!r}")
            converted[str(key)] = _stringify_int_keys(nested_value)
        return converted
    if isinstance(value, list):
        return [_stringify_int_keys(item) for item in value]
    return value
