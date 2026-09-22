"""The verifier accepts a proof only on the kernel's word, and fails closed."""

import os
import shutil
from pathlib import Path

import pytest

from prover_loop.verify import LEAN_DIR, check, judge, run_bounded

NAME = "t"
CLEAN = "'t' depends on axioms: [propext, Classical.choice, Quot.sound]\n"


def test_standard_axioms_are_accepted():
    assert judge(CLEAN, 0, NAME).verified


def test_no_axioms_at_all_is_accepted():
    assert judge("'t' does not depend on any axioms\n", 0, NAME).verified


@pytest.mark.parametrize("output, code, reason", [
    ("'t' depends on axioms: [sorryAx]\n", 0, "disallowed axioms: sorryAx"),
    ("'t' depends on axioms: [propext, Lean.ofReduceBool]\n", 0, "disallowed axioms: Lean.ofReduceBool"),
    ("f.lean:3:2: error: linarith failed\n" + CLEAN, 1, "lean error"),
    (CLEAN, 1, "lean error"),
    ("", 0, "no axiom report"),
    ("'other' depends on axioms: [propext]\n", 0, "no axiom report"),
])
def test_everything_else_is_rejected_with_its_reason(output, code, reason):
    verdict = judge(output, code, NAME)
    assert not verdict.verified and verdict.reason == reason


def alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ")[1][0] not in "ZX"
    except (FileNotFoundError, IndexError):
        return False


def test_a_timeout_kills_the_whole_process_group_not_just_the_child(tmp_path):
    """`lake env lean` is lake with lean beneath it; killing only lake leaves a
    runaway lean holding gigabytes, which is how a verifier timeout OOM-killed the server."""
    import time
    pidfile = tmp_path / "grandchild.pid"
    start = time.monotonic()
    code, _ = run_bounded(["sh", "-c", f"sleep 60 & echo $! > {pidfile}; wait"], tmp_path, timeout=1)
    assert code is None and time.monotonic() - start < 10, "returns at the timeout, not when the grandchild ends"
    time.sleep(0.2)
    assert not alive(int(pidfile.read_text()))


def test_a_finished_command_returns_its_code_and_output(tmp_path):
    assert run_bounded(["sh", "-c", "echo out; echo err >&2; exit 3"], tmp_path, timeout=10) == (3, "out\nerr\n")


LEAN = shutil.which("lake") is not None and (LEAN_DIR / ".lake" / "packages" / "mathlib").is_dir()
needs_lean = pytest.mark.skipif(not LEAN and not os.environ.get("REQUIRE_LEAN"), reason="Lean + Mathlib not set up")

HEADER = "import Mathlib\n\n"


@needs_lean
@pytest.mark.parametrize("body, verified", [
    ("theorem t (x : ℕ) (h : x = 2) : x + x = 4 := by\n  subst h\n  rfl\n", True),
    ("theorem t : (2 : ℕ) + 2 = 5 := by\n  sorry\n", False),
    ("theorem t : (2 : ℕ) ^ 10 = 1024 := by\n  native_decide\n", False),
    ("theorem t : (2 : ℕ) + 2 = 5 := by\n  norm_num\n", False),
    ("theorem t : (2 : ℕ) + 2 = 4 := by\n  rfl\n#exit\n", False),
])
def test_real_lean(body, verified):
    assert check(HEADER + body, NAME, timeout=600).verified is verified
