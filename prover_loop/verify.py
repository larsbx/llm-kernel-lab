"""Lean 4 + Mathlib verification, on the kernel's word.

A candidate is compiled with `#print axioms <theorem>` appended, and accepted
only if Lean exits cleanly, reports no error, and the theorem's axioms are a
subset of Lean's standard three. That one check rejects `sorry` (`sorryAx`),
`native_decide` (`Lean.ofReduceBool`, which trusts the compiler), `admit`, and
any helper lemma that smuggles either in, because axioms are collected
transitively. A missing report (a `#exit`, a renamed theorem, a crash) is a
rejection, never a pass.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

LEAN_DIR = Path(__file__).resolve().parents[1] / "lean"
STANDARD_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
ERROR = re.compile(r":\d+:\d+: error")


@dataclass(frozen=True)
class Verdict:
    verified: bool
    reason: str
    seconds: float = 0.0


def judge(output: str, returncode: int, theorem: str) -> Verdict:
    """The verdict on one Lean run's combined output. Pure, so it is tested without Lean."""
    if returncode != 0 or ERROR.search(output):
        return Verdict(False, "lean error")
    if f"'{theorem}' does not depend on any axioms" in output:
        return Verdict(True, "verified")
    report = re.search(rf"^'{re.escape(theorem)}' depends on axioms: \[(.*)\]$", output, re.MULTILINE)
    if not report:
        return Verdict(False, "no axiom report")
    extra = sorted({a.strip() for a in report.group(1).split(",")} - STANDARD_AXIOMS)
    return Verdict(False, "disallowed axioms: " + ", ".join(extra)) if extra else Verdict(True, "verified")


def check(source: str, theorem: str, timeout: float = 300, lean_dir: Path = LEAN_DIR) -> Verdict:
    """Compile `source` in the Mathlib project and judge it."""
    scratch = lean_dir / ".verify"
    scratch.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".lean", dir=scratch, delete=False, encoding="utf-8") as f:
        f.write(f"{source}\n\n#print axioms {theorem}\n")
    start = time.monotonic()
    try:
        done = subprocess.run(["lake", "env", "lean", f.name], cwd=lean_dir, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return Verdict(False, "timeout", time.monotonic() - start)
    finally:
        Path(f.name).unlink(missing_ok=True)
    verdict = judge(done.stdout + done.stderr, done.returncode, theorem)
    return Verdict(verdict.verified, verdict.reason, time.monotonic() - start)
