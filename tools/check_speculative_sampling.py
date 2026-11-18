#!/usr/bin/env python3
"""Fresh Lean proof/type/axiom acceptance and exact Python mass conformance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sampling_lab.speculative import prepare

STANDARD_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
MODEL_NAMES = (
    "accepted_add_residual", "accepted_add_rejected", "scaled_target_total",
    "scaled_draft_total", "accepted_residual_total", "accepted_rejected_total",
    "residual_equals_rejection", "output_mass_closed_form",
    "output_mass_target_identity", "output_mass_total", "output_space_positive",
    "exact_target_distribution",
)
CONTRACT_NAMES = (
    "target_identity", "mass_total", "rejection_identity", "positive_space",
    "common_denominator", "complete_contract",
)
INPUTS = (
    "lean/lean-toolchain", "lean/SpeculativeSampling.lean",
    "lean/SpeculativeSamplingContract.lean", "sampling_lab/__init__.py",
    "sampling_lab/speculative.py", "tests/test_speculative_sampling.py",
    "tests/test_speculative_acceptance.py", "tools/check_speculative_sampling.py",
    ".github/workflows/speculative-sampling.yml", "ESTATE.toml", "pixi.toml",
)


def audit_report(output: str, name: str) -> list[str]:
    """Require exactly one anchored report, never a missing or conflicting one."""
    pattern = re.compile(
        rf"^'{re.escape(name)}' (?:does not depend on any axioms|"
        r"depends on axioms: \[([^\]\r\n]*)\])$", re.MULTILINE
    )
    matches = list(pattern.finditer(output))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one axiom report for {name}")
    axioms = sorted(filter(None, (a.strip() for a in (matches[0].group(1) or "").split(","))))
    if not set(axioms) <= STANDARD_AXIOMS:
        raise ValueError(f"disallowed axioms for {name}: {axioms}")
    return axioms


def snapshot() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in INPUTS}


def source_identity(require_clean: bool) -> tuple[str, str]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if require_clean:
        if status:
            raise ValueError("acceptance requires a clean committed checkout")
        # Direct blob comparison also rejects assume-unchanged/skip-worktree edits.
        for name in INPUTS:
            committed = subprocess.check_output(["git", "show", f"{head}:{name}"], cwd=ROOT)
            if committed != (ROOT / name).read_bytes():
                raise ValueError(f"acceptance source differs from commit: {name}")
    return head, status


def cases() -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    result = []
    for size in (1, 2, 3):
        weights = [w for w in product(range(3), repeat=size) if sum(w)]
        result.extend(product(weights, repeat=2))
    # Additional cases use unequal totals, duplicate rows, and > machine-word weights.
    result.extend([
        ((1, 4, 2), (3, 1, 1)),
        ((0, 2, 0, 3), (4, 0, 1, 0)),
        ((2**257 + 123, 1), (1, 2**257 + 123)),
    ])
    return result


def conformance_source(inputs) -> str:
    literals = ["[" + ",".join(f"({p},{q})" for p, q in zip(ps, qs)) + "]" for ps, qs in inputs]
    return """import SpeculativeSampling
open SpeculativeSampling
def encode (xs : List Nat) : String :=
  "[" ++ String.intercalate "," (xs.map toString) ++ "]"
def main : IO Unit := do
  let cases : List Weights := [""" + ",".join(literals) + """]
  for ws in cases do
    let totals := [targetTotal ws, draftTotal ws, denominator ws,
      totalAccepted ws, totalResidual ws, totalRejected ws, outputSpace ws]
    let rows := ws.flatMap (fun w =>
      [scaledTarget ws w, scaledDraft ws w, accepted ws w,
       residual ws w, rejected ws w, outputMass ws w])
    IO.println ("MASS " ++ encode (totals ++ rows))
"""


def expected_mass(ps, qs) -> list[int]:
    plan = prepare(ps, qs)
    r = plan.rejection_total
    rejected = tuple(q * plan.target_total - a for q, a in zip(qs, plan.accepted))
    space = plan.denominator if r == 0 else plan.denominator * r
    output = [plan.target_total, plan.draft_total, plan.denominator,
              sum(plan.accepted), r, sum(rejected), space]
    for p, q, a, residual, reject in zip(ps, qs, plan.accepted, plan.residual, rejected):
        mass = a if r == 0 else a * r + sum(rejected) * residual
        output.extend([p * plan.draft_total, q * plan.target_total, a, residual, reject, mass])
    return output


def accept(lean: str, require_clean: bool = False) -> dict:
    before = snapshot()
    head, status = source_identity(require_clean)
    pin = (ROOT / "lean/lean-toolchain").read_text().strip().rsplit(":v", 1)[1]
    version = subprocess.check_output([lean, "--version"], text=True, timeout=20).strip()
    if not re.search(rf"\bversion {re.escape(pin)}(?:,|\s|\))", version):
        raise ValueError(f"expected pinned Lean {pin}; got {version}")
    axioms = {}
    controls = []
    with tempfile.TemporaryDirectory(prefix="speculative-acceptance-") as temporary:
        tmp = Path(temporary)
        env = dict(os.environ, LEAN_PATH=str(tmp))

        def run(path: Path, *, compile_to: Path | None = None, execute=False):
            command = [lean]
            if compile_to is not None:
                command += ["-o", str(compile_to)]
            if execute:
                command += ["--run"]
            command += [str(path)]
            process = subprocess.run(command, cwd=path.parent, env=env, text=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            return process.returncode, process.stdout

        for module, names in (("SpeculativeSampling", MODEL_NAMES),
                              ("SpeculativeSamplingContract", CONTRACT_NAMES)):
            code, output = run(ROOT / "lean" / f"{module}.lean", compile_to=tmp / f"{module}.olean")
            if code != 0:
                raise ValueError(f"{module} compilation failed:\n{output}")
            for name in names:
                full_name = f"{module}.{name}"
                axioms[full_name] = audit_report(output, full_name)

        inputs = cases()
        conformance = tmp / "Conformance.lean"
        conformance.write_text(conformance_source(inputs))
        code, output = run(conformance, execute=True)
        if code != 0:
            raise ValueError(f"Lean conformance execution failed:\n{output}")
        actual = [json.loads(line[5:]) for line in output.splitlines() if line.startswith("MASS ")]
        expected = [expected_mass(ps, qs) for ps, qs in inputs]
        if len(actual) != len(expected) or actual != expected:
            raise ValueError("Lean/Python mass conformance mismatch or missing output")

        # Real compiler/audit negative controls, never statistical stand-ins.
        for label, source, require_compile in (
            ("false_proposition", "theorem sentinel : (2 : Nat) + 2 = 5 := by decide", False),
            ("sorry_axiom", "theorem sentinel : False := by sorry", True),
            ("compiler_trust", "theorem sentinel : (2 : Nat) + 2 = 4 := by native_decide", True),
            ("missing_report", "theorem other : True := by trivial\n#print axioms other", True),
        ):
            path = tmp / "Negative.lean"
            suffix = "" if label == "missing_report" else "\n#print axioms sentinel\n"
            path.write_text("import Std\n" + source + suffix)
            code, output = run(path)
            if require_compile and code != 0:
                raise ValueError(f"negative control did not compile as intended: {label}\n{output}")
            if not require_compile and code == 0:
                raise ValueError(f"negative control unexpectedly compiled: {label}")
            if require_compile:
                try:
                    audit_report(output, "sentinel")
                except ValueError:
                    pass
                else:
                    raise ValueError(f"negative control unexpectedly accepted: {label}")
            controls.append(label)

        # Mutate the actual exported theorem, leaving its name and proof report.
        # The six independent type wrappers must reject this weaker proposition.
        weak_dir = tmp / "weakened"
        weak_dir.mkdir()
        weakened = weak_dir / "SpeculativeSampling.lean"
        original = (ROOT / "lean/SpeculativeSampling.lean").read_text()
        start = original.index("theorem exact_target_distribution")
        end = original.index("\n-- Kernel-reduced", start)
        weakened.write_text(original[:start] +
                            "theorem exact_target_distribution (weights : Weights) "
                            "(hvalid : Valid weights) : True := by trivial\n" + original[end:])
        code, output = run(weakened, compile_to=weak_dir / "SpeculativeSampling.olean")
        if code != 0:
            raise ValueError(f"weakening setup failed:\n{output}")
        env["LEAN_PATH"] = str(weak_dir)
        code, output = run(ROOT / "lean/SpeculativeSamplingContract.lean")
        if code == 0 or "type mismatch" not in output:
            raise ValueError(f"weakening did not produce the expected type rejection:\n{output}")
        controls.append("weakened_statement")

    if snapshot() != before:
        raise ValueError("acceptance inputs changed during the run")
    if source_identity(require_clean)[0] != head:
        raise ValueError("source revision changed during the run")
    return {
        "schema": "llm-kernel-lab.speculative-acceptance.v1",
        "source_commit": head, "working_tree_clean": not bool(status),
        "source_sha256": before, "lean_version": version,
        "lean_mass_theorems_accepted": True, "axioms": axioms,
        "lean_python_mass_cases": len(inputs), "negative_controls": controls,
        "python_universal_refinement": False, "rng_verified": False,
        "model_integration_verified": False, "acceleration_measured": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean", default="lean")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        if args.out:
            if args.out.resolve() in {(ROOT / name).resolve() for name in INPUTS}:
                raise ValueError("receipt output cannot overwrite an acceptance input")
            # A failed rerun must not leave an older successful receipt at this path.
            args.out.unlink(missing_ok=True)
        report = accept(args.lean, args.require_clean)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"speculative acceptance FAILED: {exc}", file=sys.stderr)
        return 1
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
