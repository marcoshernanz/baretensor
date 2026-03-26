from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True, slots=True)
class RunConfig:
    experiment_name: str
    seed: int
    output_root: Path
    log_interval: int
    eval_interval: int
    checkpoint_interval: int
    sample_interval: int


@dataclass(frozen=True, slots=True)
class DataConfig:
    dataset_path: Path
    tokenizer_path: Path
    train_split_ratio: float
    context_tokens: int
    text_limit: int | None


@dataclass(frozen=True, slots=True)
class ModelConfig:
    embedding_dim: int
    num_heads: int
    num_decoder_blocks: int
    hidden_dim: int
    layer_norm_eps: float


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    name: str
    learning_rate: float


@dataclass(frozen=True, slots=True)
class TrainConfig:
    steps: int
    batch_size: int
    eval_batch_size: int
    sample_tokens: int
    loss_ema_decay: float


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    run: RunConfig
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    train: TrainConfig


def load_config(path: Path) -> TrainingConfig:
    config_path = path.resolve()
    base_dir = config_path.parent
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a TOML table.")

    _expect_exact_keys(
        "root",
        payload,
        {"run", "data", "model", "optimizer", "train"},
    )

    run_payload = _require_table("run", payload["run"])
    data_payload = _require_table("data", payload["data"])
    model_payload = _require_table("model", payload["model"])
    optimizer_payload = _require_table("optimizer", payload["optimizer"])
    train_payload = _require_table("train", payload["train"])

    run_config = _parse_run_config(run_payload, base_dir=base_dir)
    data_config = _parse_data_config(data_payload, base_dir=base_dir)
    model_config = _parse_model_config(model_payload)
    optimizer_config = _parse_optimizer_config(optimizer_payload)
    train_config = _parse_train_config(train_payload)

    _validate_cross_section_constraints(data_config, model_config, train_config, optimizer_config)

    return TrainingConfig(
        run=run_config,
        data=data_config,
        model=model_config,
        optimizer=optimizer_config,
        train=train_config,
    )


def render_config_toml(config: TrainingConfig) -> str:
    sections: list[str] = []
    sections.append(
        _render_section(
            "run",
            [
                ("experiment_name", config.run.experiment_name),
                ("seed", config.run.seed),
                ("output_root", str(config.run.output_root)),
                ("log_interval", config.run.log_interval),
                ("eval_interval", config.run.eval_interval),
                ("checkpoint_interval", config.run.checkpoint_interval),
                ("sample_interval", config.run.sample_interval),
            ],
        )
    )
    sections.append(
        _render_section(
            "data",
            [
                ("dataset_path", str(config.data.dataset_path)),
                ("tokenizer_path", str(config.data.tokenizer_path)),
                ("train_split_ratio", config.data.train_split_ratio),
                ("context_tokens", config.data.context_tokens),
                ("text_limit", config.data.text_limit),
            ],
        )
    )
    sections.append(
        _render_section(
            "model",
            [
                ("embedding_dim", config.model.embedding_dim),
                ("num_heads", config.model.num_heads),
                ("num_decoder_blocks", config.model.num_decoder_blocks),
                ("hidden_dim", config.model.hidden_dim),
                ("layer_norm_eps", config.model.layer_norm_eps),
            ],
        )
    )
    sections.append(
        _render_section(
            "optimizer",
            [
                ("name", config.optimizer.name),
                ("learning_rate", config.optimizer.learning_rate),
            ],
        )
    )
    sections.append(
        _render_section(
            "train",
            [
                ("steps", config.train.steps),
                ("batch_size", config.train.batch_size),
                ("eval_batch_size", config.train.eval_batch_size),
                ("sample_tokens", config.train.sample_tokens),
                ("loss_ema_decay", config.train.loss_ema_decay),
            ],
        )
    )
    return "\n\n".join(sections) + "\n"


def _parse_run_config(payload: dict[str, Any], *, base_dir: Path) -> RunConfig:
    _expect_exact_keys(
        "run",
        payload,
        {
            "experiment_name",
            "seed",
            "output_root",
            "log_interval",
            "eval_interval",
            "checkpoint_interval",
            "sample_interval",
        },
    )
    return RunConfig(
        experiment_name=_require_string("run.experiment_name", payload["experiment_name"]),
        seed=_require_positive_int("run.seed", payload["seed"]),
        output_root=_require_path("run.output_root", payload["output_root"], base_dir=base_dir),
        log_interval=_require_positive_int("run.log_interval", payload["log_interval"]),
        eval_interval=_require_positive_int("run.eval_interval", payload["eval_interval"]),
        checkpoint_interval=_require_positive_int(
            "run.checkpoint_interval", payload["checkpoint_interval"]
        ),
        sample_interval=_require_positive_int("run.sample_interval", payload["sample_interval"]),
    )


def _parse_data_config(payload: dict[str, Any], *, base_dir: Path) -> DataConfig:
    _expect_exact_keys(
        "data",
        payload,
        {
            "dataset_path",
            "tokenizer_path",
            "train_split_ratio",
            "context_tokens",
        },
        optional_keys={"text_limit"},
    )
    text_limit = payload.get("text_limit")
    if text_limit is not None:
        text_limit = _require_positive_int("data.text_limit", text_limit)
    dataset_path = _require_path("data.dataset_path", payload["dataset_path"], base_dir=base_dir)
    tokenizer_path = _require_path(
        "data.tokenizer_path",
        payload["tokenizer_path"],
        base_dir=base_dir,
    )
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    if not tokenizer_path.exists():
        raise ValueError(f"Tokenizer path does not exist: {tokenizer_path}")
    return DataConfig(
        dataset_path=dataset_path,
        tokenizer_path=tokenizer_path,
        train_split_ratio=_require_probability(
            "data.train_split_ratio", payload["train_split_ratio"]
        ),
        context_tokens=_require_positive_int("data.context_tokens", payload["context_tokens"]),
        text_limit=text_limit,
    )


def _parse_model_config(payload: dict[str, Any]) -> ModelConfig:
    _expect_exact_keys(
        "model",
        payload,
        {
            "embedding_dim",
            "num_heads",
            "num_decoder_blocks",
            "hidden_dim",
            "layer_norm_eps",
        },
    )
    return ModelConfig(
        embedding_dim=_require_positive_int("model.embedding_dim", payload["embedding_dim"]),
        num_heads=_require_positive_int("model.num_heads", payload["num_heads"]),
        num_decoder_blocks=_require_positive_int(
            "model.num_decoder_blocks", payload["num_decoder_blocks"]
        ),
        hidden_dim=_require_positive_int("model.hidden_dim", payload["hidden_dim"]),
        layer_norm_eps=_require_positive_float("model.layer_norm_eps", payload["layer_norm_eps"]),
    )


def _parse_optimizer_config(payload: dict[str, Any]) -> OptimizerConfig:
    _expect_exact_keys("optimizer", payload, {"name", "learning_rate"})
    name = _require_string("optimizer.name", payload["name"])
    if name != "sgd":
        raise ValueError(f"Unsupported optimizer name {name!r}. Break C v1 only supports 'sgd'.")
    return OptimizerConfig(
        name=name,
        learning_rate=_require_positive_float("optimizer.learning_rate", payload["learning_rate"]),
    )


def _parse_train_config(payload: dict[str, Any]) -> TrainConfig:
    _expect_exact_keys(
        "train",
        payload,
        {"steps", "batch_size", "eval_batch_size", "sample_tokens", "loss_ema_decay"},
    )
    loss_ema_decay = _require_float("train.loss_ema_decay", payload["loss_ema_decay"])
    if not 0.0 <= loss_ema_decay < 1.0:
        raise ValueError("train.loss_ema_decay must satisfy 0 <= value < 1.")
    return TrainConfig(
        steps=_require_positive_int("train.steps", payload["steps"]),
        batch_size=_require_positive_int("train.batch_size", payload["batch_size"]),
        eval_batch_size=_require_positive_int("train.eval_batch_size", payload["eval_batch_size"]),
        sample_tokens=_require_positive_int("train.sample_tokens", payload["sample_tokens"]),
        loss_ema_decay=loss_ema_decay,
    )


def _validate_cross_section_constraints(
    data: DataConfig,
    model: ModelConfig,
    train: TrainConfig,
    optimizer: OptimizerConfig,
) -> None:
    if model.embedding_dim % model.num_heads != 0:
        raise ValueError("model.embedding_dim must be divisible by model.num_heads.")
    if data.context_tokens <= 0:
        raise ValueError("data.context_tokens must be positive.")
    if train.batch_size <= 0 or train.eval_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if optimizer.learning_rate <= 0.0:
        raise ValueError("optimizer.learning_rate must be positive.")


def _require_table(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a TOML table.")
    return value


def _expect_exact_keys(
    name: str,
    payload: dict[str, Any],
    required_keys: set[str],
    *,
    optional_keys: set[str] | None = None,
) -> None:
    optional_keys = optional_keys or set()
    allowed_keys = required_keys | optional_keys
    unknown_keys = sorted(set(payload) - allowed_keys)
    missing_keys = sorted(required_keys - set(payload))
    if unknown_keys:
        raise ValueError(f"{name} contains unknown keys: {', '.join(unknown_keys)}")
    if missing_keys:
        raise ValueError(f"{name} is missing required keys: {', '.join(missing_keys)}")


def _require_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _require_path(name: str, value: Any, *, base_dir: Path) -> Path:
    path = Path(_require_string(name, value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def _require_positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _require_float(name: str, value: Any) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number.")
    return float(value)


def _require_positive_float(name: str, value: Any) -> float:
    result = _require_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _require_probability(name: str, value: Any) -> float:
    result = _require_float(name, value)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must satisfy 0 < value < 1.")
    return result


def _render_section(name: str, rows: list[tuple[str, Any]]) -> str:
    rendered = [f"[{name}]"]
    for key, value in rows:
        if value is None:
            rendered.append(f"# {key} = null")
            continue
        rendered.append(f"{key} = {_format_toml_value(value)}")
    return "\n".join(rendered)


def _format_toml_value(value: Any) -> str:
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    raise TypeError(f"Unsupported TOML value type: {type(value)!r}")
