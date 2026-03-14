#!/usr/bin/env bash
set -euo pipefail

brew install uv cmake ninja

uv python install 3.13

if [[ -d ".venv" ]]; then
  echo ".venv already exists; leaving it as-is."
else
  uv venv --python 3.13
fi

uv sync --dev

cmake --preset dev
cmake --build --preset dev

PYTHONPATH=src uv run python -c "import bt, numpy as np; print(bt.tensor(np.arange(4, dtype=np.float32)).shape)"
PYTHONPATH=src uv run python -m unittest discover -v -s tests -p "test_*.py"
