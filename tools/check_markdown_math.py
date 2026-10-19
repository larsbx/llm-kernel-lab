#!/usr/bin/env python3
"""Audit tensor-frontier Markdown math delimiters and control characters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "research" / "tensor_frontier"
BACKTICK_FENCE = chr(96) * 3
TILDE_FENCE = "~" * 3


def _markdown_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.md")))
        elif path.suffix.lower() == ".md":
            files.append(path)
    return files


def check_text(path: Path, text: str) -> list[str]:
    """Return human-readable Markdown math violations."""
    errors: list[str] = []

    for index, char in enumerate(text):
        code = ord(char)
        if (code < 32 and char != "\n") or code == 127:
            line = text.count("\n", 0, index) + 1
            errors.append(f"{path}:{line}: forbidden control character U+{code:04X}")

    in_code = False
    code_fence_char: str | None = None
    code_fence_length = 0
    in_display = False
    display_start = 0

    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        fence_char: str | None = None
        fence_length = 0

        if stripped:
            candidate = stripped[0]
            if candidate in (chr(96), "~"):
                while (
                    fence_length < len(stripped)
                    and stripped[fence_length] == candidate
                ):
                    fence_length += 1
                if fence_length >= 3:
                    fence_char = candidate

        if fence_char is not None:
            if not in_code:
                in_code = True
                code_fence_char = fence_char
                code_fence_length = fence_length
            elif (
                fence_char == code_fence_char
                and fence_length >= code_fence_length
                and stripped[fence_length:] == ""
            ):
                in_code = False
                code_fence_char = None
                code_fence_length = 0
            continue

        if in_code:
            continue

        if stripped == "$":
            errors.append(
                f"{path}:{line_number}: standalone '$' display fence; use '$$'"
            )
            continue

        if stripped == "$$":
            if in_display:
                in_display = False
            else:
                in_display = True
                display_start = line_number
            continue

        if in_display and "$" in line:
            errors.append(
                f"{path}:{line_number}: '$' delimiter inside '$$' display block"
            )

    if in_display:
        errors.append(f"{path}:{display_start}: unclosed '$$' display block")

    return errors


def check_path(path: Path) -> list[str]:
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        return [f"{path}: invalid UTF-8: {exc}"]
    return check_text(path, text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Markdown files or directories; defaults to research/tensor_frontier",
    )
    args = parser.parse_args()

    roots = args.paths or [DEFAULT_ROOT]
    files = _markdown_files(roots)
    errors: list[str] = []
    for path in files:
        errors.extend(check_path(path))

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
