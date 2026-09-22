"""One round of the loop: sample k proofs per problem from MAX, verify each in Lean.

Outputs, in --out:
  attempts.jsonl  every sample, with the kernel's verdict and its reason
  verified.jsonl  (prompt, completion) for each distinct verified proof: the
                  training set for the next round of expert iteration
  summary.json    solved problems and pass@k, rewritten after every problem
                  ("complete": false until the last one)

Only the verifier decides what is solved. The model's text is a candidate.

Usage:
    python -m prover_loop.run [--problems FILE] [--limit N] [--k K] [--out DIR] ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

from .client import Sampler, sampler
from .problems import Problem, load
from .prompt import prompt, proof
from .verify import Verdict, check

Verifier = Callable[[str, str], Verdict]


def attempt_rows(problem: Problem, sample: Sampler, verify: Verifier, k: int) -> tuple[list[dict], list[dict]]:
    """Every sample's record, and one training row per distinct verified proof."""
    text = prompt(problem)
    verdicts: dict[str, Verdict] = {}
    attempts, verified = [], []
    for i, completion in enumerate(sample(text, k)):
        candidate = proof(problem, completion.text)
        fresh = candidate not in verdicts
        if fresh:
            verdicts[candidate] = verify(candidate, problem.theorem)
        verdict = verdicts[candidate]
        attempts.append({
            "problem": problem.name, "sample": i, "completion": completion.text,
            "proof_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
            "verified": verdict.verified, "reason": verdict.reason, "lean_s": round(verdict.seconds, 2),
            "gen_s": round(completion.seconds, 2), "completion_tokens": completion.tokens,
        })
        if verdict.verified and fresh:
            verified.append({"problem": problem.name, "prompt": text, "completion": completion.text})
    return attempts, verified


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def append_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def run(problems: list[Problem], sample: Sampler, verify: Verifier, k: int, out: Path, meta: dict | None = None,
        resume: bool = False) -> dict:
    """A problem's rows are appended only once it is finished, so after a crash
    `resume` skips exactly the problems whose results are on disk."""
    out.mkdir(parents=True, exist_ok=True)
    log, train = out / "attempts.jsonl", out / "verified.jsonl"
    if not resume:
        log.write_text("", encoding="utf-8")
        train.write_text("", encoding="utf-8")
    recorded = read_rows(log)
    done = {r["problem"] for r in recorded}
    solved = {r["problem"] for r in recorded if r["verified"]}
    attempts = len(recorded)

    def summary(complete: bool) -> dict:
        s = (meta or {}) | {
            "k": k, "problems": len(problems), "problems_done": len(done), "attempts": attempts,
            "solved": [p.name for p in problems if p.name in solved],
            "pass_at_k": len(solved) / len(problems) if problems else 0.0, "complete": complete,
        }
        (out / "summary.json").write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
        return s

    for problem in problems:
        if problem.name in done:
            continue
        rows, proofs = attempt_rows(problem, sample, verify, k)
        append_rows(log, rows)
        append_rows(train, proofs)
        done.add(problem.name)
        attempts += len(rows)
        if proofs:
            solved.add(problem.name)
        print(f"{problem.name}: {'SOLVED' if proofs else 'unsolved'} ({len(proofs)} distinct proofs verified) "
              f"[{len(done)}/{len(problems)}]", flush=True)
        summary(complete=False)
    return summary(complete=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--problems", type=Path, default=Path("problems/minif2f_test.jsonl"))
    parser.add_argument("--names", nargs="*", help="only these problems")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="deepseek-prover-v1.5-rl")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lean-timeout", type=float, default=300)
    parser.add_argument("--resume", action="store_true", help="skip problems already recorded in --out")
    parser.add_argument("--out", type=Path, default=Path("runs") / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    args = parser.parse_args(argv)
    problems = [p for p in load(args.problems) if not args.names or p.name in args.names][:args.limit]
    sample = sampler(args.url, args.model, temperature=args.temperature, top_p=args.top_p,
                     max_tokens=args.max_tokens, seed=args.seed)
    meta = {"model": args.model, "temperature": args.temperature, "top_p": args.top_p,
            "max_tokens": args.max_tokens, "seed": args.seed, "problem_file": str(args.problems)}
    summary = run(problems, sample, lambda src, thm: check(src, thm, args.lean_timeout), args.k, args.out, meta, args.resume)
    print(f"pass@{args.k}: {len(summary['solved'])}/{summary['problems']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
