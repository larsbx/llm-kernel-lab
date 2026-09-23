#!/usr/bin/env python3
"""Fail closed on analytic primitives in the authoritative tensor source layer."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = (ROOT / "tensor_lab", ROOT / "mojo" / "tensor_frontier")

BANNED_CALLS = {
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh",
    "exp", "expm1", "log", "log1p", "log2", "log10",
    "sqrt", "complex",
}
BANNED_ATTRS = BANNED_CALLS | {"pi", "e", "tau"}
BANNED_MODULES = {"math", "cmath"}
MOJO_CALL = re.compile(
    r"\b(?:sin|cos|tan|asin|acos|atan|atan2|sinh|cosh|tanh|"
    r"exp|expm1|log|log1p|log2|log10|sqrt|complex)\s*\("
)
MOJO_CONST = re.compile(r"\b(?:pi|tau)\b")


def audit_python(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in BANNED_MODULES:
                    errors.append(f"{path}:{node.lineno}: banned module {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in BANNED_MODULES:
                errors.append(f"{path}:{node.lineno}: banned module {node.module}")
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in BANNED_CALLS:
                errors.append(f"{path}:{node.lineno}: banned call {fn.id}")
            elif isinstance(fn, ast.Attribute) and fn.attr in BANNED_CALLS:
                errors.append(f"{path}:{node.lineno}: banned call .{fn.attr}")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            errors.append(f"{path}:{node.lineno}: banned attribute .{node.attr}")
    return errors


def audit_mojo(path: Path) -> list[str]:
    errors: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        code = line.split("#", 1)[0]
        if MOJO_CALL.search(code):
            errors.append(f"{path}:{lineno}: banned analytic call")
        if MOJO_CONST.search(code):
            errors.append(f"{path}:{lineno}: banned analytic constant")
    return errors


def audit_paths(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for root in paths:
        if root.is_file():
            candidates = [root]
        elif root.exists():
            candidates = sorted(p for p in root.rglob("*") if p.suffix in {".py", ".mojo"})
        else:
            continue
        for path in candidates:
            if path.resolve() == Path(__file__).resolve():
                continue
            if path.suffix == ".py":
                errors.extend(audit_python(path))
            else:
                errors.extend(audit_mojo(path))
    return errors


def main(argv: list[str]) -> int:
    roots = [Path(arg).resolve() for arg in argv] if argv else list(DEFAULT_ROOTS)
    errors = audit_paths(roots)
    if errors:
        print("transcendental firewall: FAIL", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("transcendental firewall: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
