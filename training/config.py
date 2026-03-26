from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError
from pydantic import ValidationInfo
from pydantic import field_validator
from pydantic import model_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(ConfigModel):
    seed: int = Field(gt=0)


class DataConfig(ConfigModel):
    dataset_path: Path
    tokenizer_path: Path
    context_tokens: int = Field(gt=0)
    text_limit: int | None = Field(default=None, gt=0)

    @field_validator("dataset_path", mode="before")
    @classmethod
    def resolve_dataset_path(cls, value: object, info: ValidationInfo) -> Path:
        return _resolve_path("data.dataset_path", value, info)

    @field_validator("tokenizer_path", mode="before")
    @classmethod
    def resolve_tokenizer_path(cls, value: object, info: ValidationInfo) -> Path:
        return _resolve_path("data.tokenizer_path", value, info)

    @model_validator(mode="after")
    def validate_paths_and_ratio(self) -> DataConfig:
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")
        if not self.tokenizer_path.exists():
            raise ValueError(f"Tokenizer path does not exist: {self.tokenizer_path}")
        return self


class ModelConfig(ConfigModel):
    embedding_dim: int = Field(gt=0)
    num_heads: int = Field(gt=0)
    num_decoder_blocks: int = Field(gt=0)
    hidden_dim: int = Field(gt=0)


class OptimizerConfig(ConfigModel):
    learning_rate: float = Field(gt=0)


class TrainConfig(ConfigModel):
    steps: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    eval_batch_size: int = Field(gt=0)
    sample_tokens: int = Field(gt=0)


class TrainingConfig(ConfigModel):
    run: RunConfig
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    train: TrainConfig

    @model_validator(mode="after")
    def validate_cross_section_constraints(self) -> TrainingConfig:
        if self.model.embedding_dim % self.model.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        return self


def load_config(path: Path) -> TrainingConfig:
    config_path = path.resolve()
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    try:
        return TrainingConfig.model_validate(payload, context={"base_dir": config_path.parent})
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def render_config_toml(config: TrainingConfig) -> str:
    sections: list[str] = []
    sections.append(
        _render_section(
            "run",
            [
                ("seed", config.run.seed),
            ],
        )
    )
    sections.append(
        _render_section(
            "data",
            [
                ("dataset_path", str(config.data.dataset_path)),
                ("tokenizer_path", str(config.data.tokenizer_path)),
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
            ],
        )
    )
    sections.append(
        _render_section(
            "optimizer",
            [
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
            ],
        )
    )
    return "\n\n".join(sections) + "\n"


def _resolve_path(name: str, value: object, info: ValidationInfo) -> Path:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{name} must be a non-empty string.")
    path = Path(value).expanduser()
    base_dir = info.context.get("base_dir") if isinstance(info.context, dict) else None
    if not isinstance(base_dir, Path):
        raise ValueError("Config validation context is missing base_dir.")
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


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
