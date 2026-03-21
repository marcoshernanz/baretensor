#!/usr/bin/env python3
"""Normalize nanobind-generated stubs for BareTensor's public API."""

from __future__ import annotations

from pathlib import Path
import re
import sys


def _replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Expected snippet not found while normalizing stub:\n{old}")
    return text.replace(old, new, 1)


def normalize_stub(text: str) -> str:
    if "import builtins" not in text:
        text = _replace_once(
            text,
            "from collections.abc import Sequence\n",
            "from collections.abc import Sequence\nimport builtins\n",
        )

    text = text.replace("-> bool:", "-> builtins.bool:")
    text = text.replace("arg: bool, /)", "arg: builtins.bool, /)")
    text = text.replace("requires_grad: bool)", "requires_grad: builtins.bool)")
    text = text.replace("keepdim: bool = False", "keepdim: builtins.bool = False")
    text = text.replace("requires_grad: bool = False", "requires_grad: builtins.bool = False")

    eq_pattern = re.compile(
        r"""
    \s*@overload\n
    \s*def\ __eq__\(self,\ arg:\ Tensor,\ /\)\ ->\ Tensor:\ \.\.\.\n
    \n
    \s*@overload\n
    \s*def\ __eq__\(self,\ arg:\ float,\ /\)\ ->\ Tensor:\ \.\.\.\n
    """,
        re.VERBOSE,
    )
    ne_pattern = re.compile(
        r"""
    \s*@overload\n
    \s*def\ __ne__\(self,\ arg:\ Tensor,\ /\)\ ->\ Tensor:\ \.\.\.\n
    \n
    \s*@overload\n
    \s*def\ __ne__\(self,\ arg:\ float,\ /\)\ ->\ Tensor:\ \.\.\.\n
    """,
        re.VERBOSE,
    )

    text, eq_count = eq_pattern.subn("\n    def __eq__(self, arg: object, /) -> Any: ...\n", text)
    text, ne_count = ne_pattern.subn("\n    def __ne__(self, arg: object, /) -> Any: ...\n", text)
    if eq_count != 1 or ne_count != 1:
        raise RuntimeError("Failed to normalize __eq__/__ne__ overloads in nanobind stub.")

    if "from typing import Any, overload" not in text:
        text = _replace_once(
            text,
            "from typing import overload\n",
            "from typing import Any, overload\n",
        )

    return text


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: normalize_nanobind_stub.py <stub-path>")

    stub_path = Path(sys.argv[1])
    original = stub_path.read_text(encoding="utf-8")
    normalized = normalize_stub(original)
    if normalized != original:
        stub_path.write_text(normalized, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
