# baretensor

Bootstrap for the BareTensor project (tensor engine + autograd + CUDA), starting with a minimal nanobind extension.

## Quickstart (A1)

```bash
make bootstrap

# or run the steps manually:

brew install uv cmake ninja

uv python install 3.13
uv venv --python 3.13
uv sync --dev

cmake -S . -B build -G Ninja -DPython_EXECUTABLE="$(pwd)/.venv/bin/python"
cmake --build build

PYTHONPATH=src .venv/bin/python -c "import bt; print(bt.add(1, 2))"
PYTHONPATH=src .venv/bin/python -m unittest discover -v -s tests -p 'test_*.py'

# Stubs are generated automatically by the CMake build into:
#   src/bt/_C.pyi
# and a marker file:
#   src/bt/py.typed
```

Common commands:

- `make build`
- `make run`
- `make dog`
- `make test`
- `make check`

Docs:

- Setup: `SETUP.md`
- Project spec: `PROJECT.md`
