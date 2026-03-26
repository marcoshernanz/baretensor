from __future__ import annotations

import argparse
from pathlib import Path

from training.runner import run_from_config
from training.runner import run_from_resume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Break C tokenized decoder runner.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        type=Path,
        help="Absolute path to a TOML config that starts a new run.",
    )
    group.add_argument(
        "--resume",
        type=Path,
        help="Absolute path to an existing run directory to resume from latest checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config is not None:
        run_from_config(args.config.resolve())
        return
    if args.resume is not None:
        run_from_resume(args.resume.resolve())
        return
    raise RuntimeError("Expected either --config or --resume.")


if __name__ == "__main__":
    main()
