# baretensor

Bootstrap for the BareTensor project (tensor engine + autograd + CUDA), starting with a minimal nanobind extension.

## Quickstart (A1)

```bash
brew install uv cmake ninja

uv python install 3.13
uv venv --python 3.13
uv sync --dev

cmake -S . -B build -G Ninja -DPython_EXECUTABLE="$(pwd)/.venv/bin/python"
cmake --build build

PYTHONPATH=src .venv/bin/python -c "import bt; print(bt.add(1, 2))"
PYTHONPATH=src .venv/bin/python -m unittest discover -v -s tests -p 'test_*.py'
```

Docs:

- Setup: `SETUP.md`
- Project spec: `PROJECT.md`
