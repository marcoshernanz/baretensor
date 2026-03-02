#!/usr/bin/env python3
"""Phase 0 training entrypoint scaffold for BareTensor LLM experiments.

This script intentionally focuses on reproducible run setup first:
- Parse a typed config object.
- Apply profile overrides (`dev-cpu`, `local-2060`, `cloud`, etc.).
- Materialize a run directory with metadata and resolved config.

The full training loop is a subsequent milestone in the roadmap.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import subprocess
import sys
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


@dataclass(slots=True)
class RunConfig:
    name: str
    output_dir: str
    seed: int
    device: str
    dry_run: bool


@dataclass(slots=True)
class DatasetConfig:
    name: str
    train_manifest: str
    val_manifest: str
    test_manifest: str
    context_length: int


@dataclass(slots=True)
class TokenizerConfig:
    kind: str
    vocab_size: int
    artifact_path: str


@dataclass(slots=True)
class ModelConfig:
    family: str
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int
    dropout_p: float


@dataclass(slots=True)
class OptimizerConfig:
    kind: str
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    eps: float
    grad_clip_norm: float


@dataclass(slots=True)
class TrainingConfig:
    max_steps: int
    micro_batch_size: int
    grad_accum_steps: int
    validate_every_steps: int
    checkpoint_every_steps: int
    log_every_steps: int


@dataclass(slots=True)
class ExperimentConfig:
    run: RunConfig
    dataset: DatasetConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base`, returning a new dictionary."""
    merged: dict[str, Any] = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    """Load YAML; fall back to JSON for zero-dependency config support.

    YAML is the preferred format. If `PyYAML` is not installed, a JSON-formatted
    file with `.yaml`/`.yml` extension is also accepted because JSON is valid
    YAML syntax.
    """
    text = path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore[import-not-found]

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"Expected top-level mapping in {path}.")
        return data
    except ModuleNotFoundError:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as err:
            raise RuntimeError(
                "Failed to parse YAML config because PyYAML is not installed, and "
                "the file is not valid JSON fallback syntax. Install PyYAML or "
                "convert the file to JSON/TOML."
            ) from err
        if not isinstance(data, dict):
            raise ValueError(f"Expected top-level object in {path}.")
        return data


def load_config_dict(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return _load_yaml_or_json(path)
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Expected top-level object in {path}.")
        return raw
    if suffix == ".toml":
        if tomllib is None:  # pragma: no cover
            raise RuntimeError("Python tomllib is unavailable in this environment.")
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Expected top-level table in {path}.")
        return raw
    raise ValueError(f"Unsupported config extension: {suffix}.")


def resolve_profile(config: dict[str, Any], profile: str | None) -> dict[str, Any]:
    profiles = config.get("profiles", {})
    if profiles is None:
        profiles = {}
    if not isinstance(profiles, dict):
        raise ValueError("`profiles` must be a mapping of profile_name -> override.")

    resolved = {k: v for k, v in config.items() if k != "profiles"}

    if profile is None:
        return resolved
    if profile not in profiles:
        available = ", ".join(sorted(str(k) for k in profiles.keys()))
        raise ValueError(
            f"Unknown profile '{profile}'. Available profiles: [{available}]."
        )

    override = profiles[profile]
    if not isinstance(override, dict):
        raise ValueError(f"Profile '{profile}' must map to an object override.")

    return _deep_merge(resolved, override)


def _require_section(raw: dict[str, Any], section: str) -> dict[str, Any]:
    value = raw.get(section)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid section `{section}` in config.")
    return value


def build_experiment_config(raw: dict[str, Any]) -> ExperimentConfig:
    run = RunConfig(**_require_section(raw, "run"))
    dataset = DatasetConfig(**_require_section(raw, "dataset"))
    tokenizer = TokenizerConfig(**_require_section(raw, "tokenizer"))
    model = ModelConfig(**_require_section(raw, "model"))
    optimizer = OptimizerConfig(**_require_section(raw, "optimizer"))
    training = TrainingConfig(**_require_section(raw, "training"))

    _validate_config(run, dataset, tokenizer, model, optimizer, training)
    return ExperimentConfig(
        run=run,
        dataset=dataset,
        tokenizer=tokenizer,
        model=model,
        optimizer=optimizer,
        training=training,
    )


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"`{name}` must be > 0, got {value}.")


def _validate_config(
    run: RunConfig,
    dataset: DatasetConfig,
    tokenizer: TokenizerConfig,
    model: ModelConfig,
    optimizer: OptimizerConfig,
    training: TrainingConfig,
) -> None:
    _validate_positive_int("run.seed", run.seed)
    if run.device not in {"cpu", "cuda"}:
        raise ValueError("`run.device` must be one of {'cpu', 'cuda'}.")

    _validate_positive_int("dataset.context_length", dataset.context_length)
    _validate_positive_int("tokenizer.vocab_size", tokenizer.vocab_size)

    _validate_positive_int("model.n_layers", model.n_layers)
    _validate_positive_int("model.n_heads", model.n_heads)
    _validate_positive_int("model.d_model", model.d_model)
    _validate_positive_int("model.d_ff", model.d_ff)
    if not (0.0 <= model.dropout_p < 1.0):
        raise ValueError("`model.dropout_p` must be in [0.0, 1.0).")

    if optimizer.lr <= 0.0:
        raise ValueError("`optimizer.lr` must be > 0.")
    if optimizer.weight_decay < 0.0:
        raise ValueError("`optimizer.weight_decay` must be >= 0.")
    if not (0.0 <= optimizer.beta1 < 1.0 and 0.0 <= optimizer.beta2 < 1.0):
        raise ValueError("`optimizer.beta1` and `optimizer.beta2` must be in [0.0, 1.0).")
    if optimizer.eps <= 0.0:
        raise ValueError("`optimizer.eps` must be > 0.")
    if optimizer.grad_clip_norm <= 0.0:
        raise ValueError("`optimizer.grad_clip_norm` must be > 0.")

    _validate_positive_int("training.max_steps", training.max_steps)
    _validate_positive_int("training.micro_batch_size", training.micro_batch_size)
    _validate_positive_int("training.grad_accum_steps", training.grad_accum_steps)
    _validate_positive_int("training.validate_every_steps", training.validate_every_steps)
    _validate_positive_int(
        "training.checkpoint_every_steps", training.checkpoint_every_steps
    )
    _validate_positive_int("training.log_every_steps", training.log_every_steps)


def _timestamp_utc() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%d-%H%M%SZ")


def _safe_slug(value: str) -> str:
    kept = [ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.strip()]
    slug = "".join(kept).strip("-")
    return slug or "run"


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def create_run_dir(config: ExperimentConfig, run_name_override: str | None) -> Path:
    output_dir = Path(config.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = run_name_override if run_name_override else config.run.name
    run_name = _safe_slug(run_name)
    run_dir = output_dir / f"{_timestamp_utc()}-{run_name}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def dump_run_metadata(
    run_dir: Path,
    config: ExperimentConfig,
    config_path: Path,
    profile: str | None,
) -> None:
    resolved_config = asdict(config)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(resolved_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "config_path": str(config_path),
        "profile": profile,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _print_summary(config: ExperimentConfig, run_dir: Path, profile: str | None) -> None:
    summary = textwrap.dedent(
        f"""
        BareTensor train.py bootstrap complete.
          run_dir: {run_dir}
          profile: {profile or '<none>'}
          device: {config.run.device}
          dataset: {config.dataset.name}
          tokenizer: {config.tokenizer.kind} (vocab={config.tokenizer.vocab_size})
          model: {config.model.family} layers={config.model.n_layers} heads={config.model.n_heads} d_model={config.model.d_model}
          train: steps={config.training.max_steps} micro_batch={config.training.micro_batch_size} grad_accum={config.training.grad_accum_steps}
        """
    ).strip()
    print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BareTensor training entrypoint scaffold (Phase 0)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment config (.yaml/.yml/.json/.toml).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Optional config profile override (e.g., dev-cpu/local-2060/cloud).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run-name override (slug-safe).",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Disable dry-run mode (currently raises NotImplementedError).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config_path: Path = args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = load_config_dict(config_path)
    resolved = resolve_profile(raw, args.profile)
    config = build_experiment_config(resolved)

    if args.no_dry_run:
        config.run.dry_run = False

    run_dir = create_run_dir(config, args.run_name)
    dump_run_metadata(run_dir, config, config_path, args.profile)
    _print_summary(config, run_dir, args.profile)

    if config.run.dry_run:
        print(
            "Dry-run mode enabled; train/validation loop is intentionally deferred "
            "to the next roadmap milestone."
        )
        return 0

    raise NotImplementedError(
        "Training loop is not implemented yet. This milestone adds the unified "
        "entrypoint + typed config + profile resolution + run bootstrap only."
    )


if __name__ == "__main__":
    raise SystemExit(main())
