from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
import platform
import socket
import subprocess
from time import perf_counter

import flax
import jax
import numpy as np
import optax  # pyright: ignore

from training.artifacts import MetricRow
from training.artifacts import RunPaths
from training.artifacts import append_metric_row
from training.artifacts import create_run_paths
from training.artifacts import regenerate_loss_curve
from training.artifacts import write_json
from training.artifacts import write_sample
from training.artifacts import write_text
from training.config import TrainingConfig
from training.config import load_config
from training.config import render_config_toml
from training.recipes import TokenizedDecoderJaxRecipe


RECIPE_NAME = "tokenized_decoder_jax"


@dataclass(frozen=True, slots=True)
class RunResult:
    run_dir: Path
    train_loss: float
    validation_loss: float
    train_seconds: float
    total_seconds: float
    interrupted: bool


def run_from_config(config_path: Path) -> RunResult:
    config = load_config(config_path)
    paths = create_run_paths(config.run.output_root, config.run.experiment_name)
    metadata = _build_metadata(config, paths.run_dir, status="running")
    write_text(paths.resolved_config_path, render_config_toml(config))
    write_json(paths.metadata_path, metadata)
    return _execute_training(config, paths, metadata)


def _execute_training(
    config: TrainingConfig,
    paths: RunPaths,
    metadata: dict[str, object],
) -> RunResult:
    total_start = perf_counter()
    recipe = TokenizedDecoderJaxRecipe.create(config)
    train_start = perf_counter()

    batch_rng = jax.random.key(config.run.seed)
    ema_train_loss: float | None = None

    sample_rng = jax.random.key(config.run.seed)
    _, sample_text = recipe.generate_sample(sample_rng)
    write_sample(paths.samples_dir, 0, sample_text)

    interrupted = False

    try:
        for step in range(config.train.steps):
            batch_rng, raw_train_loss = recipe.train_batch(batch_rng)
            ema_train_loss = (
                raw_train_loss
                if ema_train_loss is None
                else config.train.loss_ema_decay * ema_train_loss
                + (1.0 - config.train.loss_ema_decay) * raw_train_loss
            )

            completed_steps = step + 1
            is_eval_step = (
                completed_steps % config.run.eval_interval == 0
                or completed_steps == config.train.steps
            )
            is_sample_step = (
                completed_steps % config.run.sample_interval == 0
                or completed_steps == config.train.steps
            )
            is_log_step = (
                completed_steps % config.run.log_interval == 0
                or completed_steps == config.train.steps
            )

            validation_loss: float | None = None
            if is_eval_step:
                validation_loss = recipe.evaluate_validation_loss()

            append_metric_row(
                paths.metrics_path,
                MetricRow(
                    step=step,
                    raw_train_loss=raw_train_loss,
                    ema_train_loss=ema_train_loss,
                    validation_loss=validation_loss,
                ),
            )

            if is_sample_step:
                batch_rng, sample_text = recipe.generate_sample(batch_rng)
                write_sample(paths.samples_dir, completed_steps, sample_text)

            if is_log_step:
                message = (
                    f"step={completed_steps} raw_train_loss={raw_train_loss:.6f} "
                    f"ema_train_loss={ema_train_loss:.6f}"
                )
                if validation_loss is not None:
                    message += f" validation_loss={validation_loss:.6f}"
                print(message)
    except KeyboardInterrupt:
        interrupted = True
        metadata["status"] = "interrupted"
        metadata["last_update_time"] = _utcnow()
        metadata["end_time"] = _utcnow()
        write_json(paths.metadata_path, metadata)
        regenerate_loss_curve(paths.metrics_path, paths.loss_curve_path)
        raise SystemExit(130) from None

    train_seconds = perf_counter() - train_start
    train_loss = recipe.evaluate_train_loss()
    validation_loss = recipe.evaluate_validation_loss()
    batch_rng, sample_text = recipe.generate_sample(batch_rng)
    write_sample(paths.samples_dir, config.train.steps, sample_text)
    regenerate_loss_curve(paths.metrics_path, paths.loss_curve_path)

    metadata["status"] = "completed"
    metadata["last_update_time"] = _utcnow()
    metadata["end_time"] = _utcnow()
    metadata["train_summary"] = {
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "train_seconds": train_seconds,
        "steps_per_second": config.train.steps / train_seconds,
        "final_ema_train_loss": ema_train_loss,
        "dataset_stats": asdict(recipe.stats),
    }
    write_json(paths.metadata_path, metadata)

    total_seconds = perf_counter() - total_start
    _print_final_summary(
        config=config,
        paths=paths,
        recipe=recipe,
        train_loss=train_loss,
        validation_loss=validation_loss,
        train_seconds=train_seconds,
        total_seconds=total_seconds,
    )

    return RunResult(
        run_dir=paths.run_dir,
        train_loss=train_loss,
        validation_loss=validation_loss,
        train_seconds=train_seconds,
        total_seconds=total_seconds,
        interrupted=interrupted,
    )


def _print_final_summary(
    *,
    config: TrainingConfig,
    paths: RunPaths,
    recipe: TokenizedDecoderJaxRecipe,
    train_loss: float,
    validation_loss: float,
    train_seconds: float,
    total_seconds: float,
) -> None:
    print(f"recipe={RECIPE_NAME}")
    print(f"run_dir={paths.run_dir}")
    print(f"tokenizer_path={config.data.tokenizer_path}")
    print(f"vocab_size={recipe.stats.vocab_size}")
    print(f"train_chars={recipe.stats.train_chars}")
    print(f"validation_chars={recipe.stats.validation_chars}")
    print(f"train_tokens={recipe.stats.train_tokens}")
    print(f"validation_tokens={recipe.stats.validation_tokens}")
    print(f"train_chars_per_token={recipe.stats.train_chars_per_token:.4f}")
    print(f"validation_chars_per_token={recipe.stats.validation_chars_per_token:.4f}")
    print(f"train_loss={train_loss:.6f}")
    print(f"validation_loss={validation_loss:.6f}")
    print(f"metrics_csv={paths.metrics_path}")
    print(f"loss_curve_svg={paths.loss_curve_path}")
    print(f"train_seconds={train_seconds:.3f}")
    print(f"steps_per_second={config.train.steps / train_seconds:.3f}")
    print(f"total_seconds={total_seconds:.3f}")


def _build_metadata(
    config: TrainingConfig,
    run_dir: Path,
    *,
    status: str,
) -> dict[str, object]:
    git_commit, dirty = _git_state(run_dir)
    return {
        "recipe_name": RECIPE_NAME,
        "status": status,
        "start_time": _utcnow(),
        "last_update_time": _utcnow(),
        "end_time": None,
        "dataset_path": str(config.data.dataset_path),
        "tokenizer_path": str(config.data.tokenizer_path),
        "dataset_sha256": _sha256(config.data.dataset_path),
        "tokenizer_sha256": _sha256(config.data.tokenizer_path),
        "git_commit": git_commit,
        "git_dirty": dirty,
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "flax_version": flax.__version__,
        "optax_version": optax.__version__,
        "numpy_version": np.__version__,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(cwd: Path) -> tuple[str | None, bool | None]:
    repo_root = cwd.resolve()
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if commit_result.returncode != 0:
        return None, None

    status_result = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = None if status_result.returncode != 0 else status_result.stdout.strip() != ""
    return commit_result.stdout.strip(), dirty


def _utcnow() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
