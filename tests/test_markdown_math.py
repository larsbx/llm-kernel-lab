import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "tools" / "check_markdown_math.py"


def run_audit(path: Path | None = None):
    command = [sys.executable, str(AUDIT)]
    if path is not None:
        command.append(str(path))
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_tensor_frontier_markdown_math_passes():
    run = run_audit()
    assert run.returncode == 0, run.stderr


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$\nx+y\n$\n", "standalone '$' display fence"),
        ("$$\n\\lambda_c$A$\n$$\n", "inside '$$' display block"),
        ("value\x0bhere\n", "U+000B"),
        ("$$\nx+y\n", "unclosed '$$' display block"),
    ],
)
def test_markdown_math_audit_rejects_known_corruptions(tmp_path, text, expected):
    bad = tmp_path / "bad.md"
    bad.write_bytes(text.encode("utf-8"))
    run = run_audit(bad)
    assert run.returncode == 1
    assert expected in run.stderr
