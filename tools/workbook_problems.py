#!/usr/bin/env python3
"""Draw a training problem set from Lean Workbook.

Only statements that Lean Workbook ships a proof for are drawn, so every
problem is true and no sample is spent on a mis-formalization. The draw is
deterministic (ascending SHA-256 of the theorem name), excludes every
statement that matches an evaluation problem (`statement_key`, so renamed
duplicates count), and, with --lean, drops statements that no longer
elaborate under this repository's Mathlib: Lean Workbook was formalized
against Lean v4.8, and one batch Lean run checks them all.

Usage:
    workbook_problems.py WORKBOOK_JSON --n 400 [--exclude problems/minif2f_test.jsonl] [--lean] [--out FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prover_loop.problems import Problem, load, statement_key  # noqa: E402
from prover_loop.verify import LEAN_DIR  # noqa: E402

HEADER = ("import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 400000\n\n"
          "open BigOperators Real Nat Topology Rat\n\n")
SORRY = re.compile(r"\s*:=\s*by\s+sorry\s*$")
LOCATED_ERROR = re.compile(r":(\d+):\d+: error", re.MULTILINE)


def to_problem(row: dict) -> Problem:
    statement = SORRY.sub(" := by\n", row["formal_statement"].strip())
    name = re.match(r"theorem\s+(\S+)", statement).group(1)
    return Problem(name, row.get("natural_language_statement", ""), HEADER + statement)


def select(rows: list[dict], n: int, exclude: set[str]) -> list[Problem]:
    proved = (to_problem(r) for r in rows if r.get("proof"))
    eligible = sorted((p for p in proved if statement_key(p.formal_statement) not in exclude),
                      key=lambda p: hashlib.sha256(p.name.encode()).hexdigest())
    return eligible[:n]


def batch_source(problems: list[Problem]) -> tuple[str, dict[str, tuple[int, int]]]:
    """One file stating every theorem with `sorry`, and each theorem's 1-based line span."""
    source, spans = HEADER, {}
    for p in problems:
        start = source.count("\n") + 1
        source += p.formal_statement.removeprefix(HEADER) + "  sorry\n\n"
        spans[p.name] = (start, source.count("\n") - 1)
    return source, spans


def uncompiled(output: str, spans: dict[str, tuple[int, int]]) -> set[str]:
    lines = {int(m) for m in LOCATED_ERROR.findall(output)}
    return {name for name, (first, last) in spans.items() if any(first <= line <= last for line in lines)}


def compiling(problems: list[Problem], lean_dir: Path = LEAN_DIR, timeout: float = 3600) -> list[Problem]:
    source, spans = batch_source(problems)
    path = lean_dir / ".verify" / "workbook_statements.lean"
    path.parent.mkdir(exist_ok=True)
    path.write_text(source, encoding="utf-8")
    done = subprocess.run(["lake", "env", "lean", str(path)], cwd=lean_dir, capture_output=True, text=True, timeout=timeout)
    bad = uncompiled(done.stdout + done.stderr, spans)
    return [p for p in problems if p.name not in bad]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("workbook", type=Path, help="lean_workbook.json from internlm/Lean-Workbook")
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--exclude", type=Path, nargs="*", default=[Path("problems/minif2f_test.jsonl")])
    parser.add_argument("--lean", action="store_true", help="drop statements that do not elaborate here")
    parser.add_argument("--out", type=Path, default=Path("problems/lean_workbook_train.jsonl"))
    args = parser.parse_args()
    rows = json.loads(args.workbook.read_text(encoding="utf-8"))
    exclude = {statement_key(p.formal_statement) for path in args.exclude for p in load(path)}
    oversample = args.n * 5 // 4 if args.lean else args.n
    problems = select(rows, oversample, exclude)
    if args.lean:
        kept = compiling(problems)
        print(f"{len(problems) - len(kept)} of {len(problems)} statements do not elaborate under this Mathlib")
        problems = kept[:args.n]
    args.out.write_text("".join(json.dumps(p.__dict__, ensure_ascii=False) + "\n" for p in problems), encoding="utf-8")
    print(f"wrote {len(problems)} problems to {args.out} (excluded {len(exclude)} eval statements)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
