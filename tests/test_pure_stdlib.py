"""The magform package must depend only on the Python standard library."""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

PKG = pathlib.Path(__file__).resolve().parents[1] / "magform"
STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}


def _modules():
    return sorted(PKG.glob("*.py"))


@pytest.mark.skipif(not STDLIB, reason="sys.stdlib_module_names requires Python 3.10+")
@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_only_stdlib_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top in STDLIB, f"{path.name} imports non-stdlib module {alias.name!r}"
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative import inside the package
            top = (node.module or "").split(".")[0]
            assert top in STDLIB, f"{path.name} imports non-stdlib module {node.module!r}"


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_no_absolute_self_imports(path):
    """Inside the package, siblings are imported relatively (avoids partial-import cycles)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and not node.level:
            assert (node.module or "").split(".")[0] != "magform", f"{path.name}: use relative imports"
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] != "magform", f"{path.name}: use relative imports"
