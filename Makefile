# BareTensor convenience commands.
#
# Usage:
#   make <target>
#
# Notes:
# - Most Python commands are run via `uv run`.
# - Use `PYTHONPATH=src` so `import bt` finds the `src/` layout package.
# - The native extension is built into `src/bt/` so imports work without install.

SHELL := /bin/bash

.DEFAULT_GOAL := help

UV ?= uv
CMAKE ?= cmake
PYTHONPATH ?= src
BUILD_DIR ?= build

.PHONY: help
help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "%-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: bootstrap
bootstrap: ## Install deps + venv + build + run tests (macOS: uses Homebrew)
	./scripts/bootstrap.sh

.PHONY: sync
sync: ## Install/update dev tools (ruff/pyright/nanobind) in .venv
	$(UV) sync --dev

.PHONY: configure
configure: ## Configure CMake (preset: dev)
	$(CMAKE) --preset dev

.PHONY: build
build: ## Build native extension (preset: dev)
	$(CMAKE) --build --preset dev

.PHONY: rebuild
rebuild: ## Clean build dir, then configure + build
	rm -rf "$(BUILD_DIR)"
	$(CMAKE) --preset dev
	$(CMAKE) --build --preset dev

.PHONY: run
run: ## Run the example script
	PYTHONPATH="$(PYTHONPATH)" $(UV) run python examples/example.py

.PHONY: repl
repl: ## Start an interactive Python REPL with bt importable
	PYTHONPATH="$(PYTHONPATH)" $(UV) run python

.PHONY: test
test: ## Run unit tests (unittest discovery)
	PYTHONPATH="$(PYTHONPATH)" $(UV) run python -m unittest discover -v -s tests -p "test_*.py"

.PHONY: fmt
fmt: ## Format Python (ruff format)
	PYTHONPATH="$(PYTHONPATH)" $(UV) run ruff format src tests examples

.PHONY: lint
lint: ## Lint Python (ruff check)
	PYTHONPATH="$(PYTHONPATH)" $(UV) run ruff check src tests examples

.PHONY: typecheck
typecheck: ## Typecheck Python (pyright)
	PYTHONPATH="$(PYTHONPATH)" $(UV) run pyright

.PHONY: check
check: lint typecheck test ## Run lint + typecheck + tests

.PHONY: stub
stub: ## Generate/update nanobind stubs via CMake
	$(CMAKE) --build --preset dev --target bt_stub

.PHONY: clean
clean: ## Remove CMake build directory
	rm -rf "$(BUILD_DIR)"

.PHONY: distclean
distclean: clean ## Remove build dir + venv + compiled extension
	rm -rf .venv
	rm -f src/bt/_C*.so src/bt/_C*.dylib src/bt/_C*.pyd
