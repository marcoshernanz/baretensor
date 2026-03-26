from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import jax

from tokenizer.bpe import BYTE_VOCAB_SIZE
from tokenizer.bpe import train_bpe
from training.checkpoints import CheckpointState
from training.checkpoints import load_checkpoint
from training.checkpoints import save_checkpoint
from training.config import load_config
from training.recipes import TokenizedDecoderJaxRecipe


class TrainingRunnerTests(unittest.TestCase):
    def test_cli_run_creates_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = _write_tiny_training_fixture(temp_path)

            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve().parent.parent / "train.py"), "--config", str(config_path)],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            run_dir = _extract_run_dir(result.stdout)
            self.assertTrue((run_dir / "resolved_config.toml").exists())
            self.assertTrue((run_dir / "run_metadata.json").exists())
            self.assertTrue((run_dir / "metrics.csv").exists())
            self.assertTrue((run_dir / "loss_curve.svg").exists())
            self.assertTrue((run_dir / "samples" / "step_000000.txt").exists())
            self.assertTrue((run_dir / "checkpoints" / "latest.msgpack").exists())
            self.assertTrue((run_dir / "checkpoints" / "best.msgpack").exists())

            metrics_rows = _read_metrics(run_dir / "metrics.csv")
            self.assertEqual(len(metrics_rows), 3)
            self.assertEqual(metrics_rows[-1]["step"], "2")
            self.assertNotEqual(metrics_rows[-1]["validation_loss"], "")

            metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")
            self.assertIn("dataset_sha256", metadata)
            self.assertIn("tokenizer_sha256", metadata)

    def test_checkpoint_round_trip_restores_runner_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = _write_tiny_training_fixture(temp_path)
            config = load_config(config_path)
            recipe = TokenizedDecoderJaxRecipe.create(config)
            checkpoint_path = temp_path / "checkpoint.msgpack"
            checkpoint_state = CheckpointState(
                next_step=2,
                best_validation_loss=1.25,
                ema_train_loss=1.5,
                batch_rng=jax.random.key(17),
                status="running",
            )

            save_checkpoint(
                checkpoint_path,
                model=recipe.model,
                optimizer=recipe.optimizer,
                state=checkpoint_state,
            )
            restored = load_checkpoint(
                checkpoint_path,
                model=recipe.model,
                optimizer=recipe.optimizer,
            )

            self.assertEqual(restored.next_step, 2)
            self.assertAlmostEqual(restored.best_validation_loss or 0.0, 1.25)
            self.assertAlmostEqual(restored.ema_train_loss or 0.0, 1.5)
            self.assertEqual(restored.status, "running")

    def test_resume_appends_metrics_and_advances_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            initial_config = _write_tiny_training_fixture(temp_path, steps=2)
            initial_result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve().parent.parent / "train.py"), "--config", str(initial_config)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(initial_result.returncode, 0, msg=initial_result.stderr)
            run_dir = _extract_run_dir(initial_result.stdout)

            resolved_config_path = run_dir / "resolved_config.toml"
            config_text = resolved_config_path.read_text(encoding="utf-8")
            resolved_config_path.write_text(
                config_text.replace("steps = 2", "steps = 4"),
                encoding="utf-8",
            )

            resume_result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve().parent.parent / "train.py"), "--resume", str(run_dir)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(resume_result.returncode, 0, msg=resume_result.stderr)

            metrics_rows = _read_metrics(run_dir / "metrics.csv")
            self.assertEqual(len(metrics_rows), 4)
            self.assertEqual(metrics_rows[-1]["step"], "3")

            metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")


def _write_tiny_training_fixture(root: Path, *, steps: int = 3) -> Path:
    dataset_path = root / "dataset.txt"
    tokenizer_path = root / "tokenizer.json"
    dataset_path.write_text(
        "To be, or not to be, that is the question.\n" * 50,
        encoding="utf-8",
    )
    train_bpe(dataset_path.read_text(encoding="utf-8"), BYTE_VOCAB_SIZE + 8).save(tokenizer_path)
    config_path = root / "config.toml"
    config_path.write_text(
        f"""
[run]
experiment_name = "tiny_break_c"
seed = 7
output_root = "{root / "runs"}"
log_interval = 1
eval_interval = 1
checkpoint_interval = 1
sample_interval = 1

[data]
dataset_path = "{dataset_path}"
tokenizer_path = "{tokenizer_path}"
train_split_ratio = 0.8
context_tokens = 8
# text_limit = null

[model]
embedding_dim = 8
num_heads = 2
num_decoder_blocks = 1
hidden_dim = 16
layer_norm_eps = 1e-5

[optimizer]
name = "sgd"
learning_rate = 0.05

[train]
steps = {steps}
batch_size = 2
eval_batch_size = 4
sample_tokens = 12
loss_ema_decay = 0.9
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _extract_run_dir(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("run_dir="):
            return Path(line.removeprefix("run_dir="))
    raise AssertionError(f"Could not find run_dir in stdout:\n{stdout}")


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
