"""Supervised fine-tuning rows from verified proofs.

Rows are (prompt, completion) pairs exactly as the loop sampled them, so the
tuned model is trained on the prompt format it is served with. Duplicates are
dropped, each problem contributes at most `max_per_problem` proofs (file
order), and a row whose statement is an evaluation problem refuses the whole
build: training on the test set is an error, not a filter setting.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .prompt import FENCE
from .problems import load, statement_key

OPEN = f"{FENCE}lean4\n"


class Contamination(ValueError):
    """A training row states an evaluation problem."""


def statement_of(prompt_text: str) -> str:
    return prompt_text.split(OPEN, 1)[1]


def sft_rows(rows: list[dict], eval_keys: set[str], max_per_problem: int) -> list[dict]:
    leaked = sorted({r["problem"] for r in rows if statement_key(statement_of(r["prompt"])) in eval_keys})
    if leaked:
        raise Contamination(f"{len(leaked)} training problem(s) state an evaluation problem: {', '.join(leaked)}")
    seen, per_problem, out = set(), Counter(), []
    for r in rows:
        pair = (r["prompt"], r["completion"])
        if pair in seen or per_problem[r["problem"]] >= max_per_problem:
            continue
        seen.add(pair)
        per_problem[r["problem"]] += 1
        out.append({"prompt": r["prompt"], "completion": r["completion"]})
    return out


def load_sft(verified: list[Path], eval_problems: list[Path], max_per_problem: int) -> list[dict]:
    rows = [json.loads(line) for path in verified for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    eval_keys = {statement_key(p.formal_statement) for path in eval_problems for p in load(path)}
    return sft_rows(rows, eval_keys, max_per_problem)
