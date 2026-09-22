"""DeepSeek-Prover-V1.5's whole-proof format (non-CoT).

The prompt ends inside an open ```lean4 block right after the statement's
`:= by`, so the completion is the proof body and nothing else. The proof that
gets checked is the untouched statement followed by the completion up to the
first closing fence: the model can never restate, weaken, or rename the goal.
"""

from __future__ import annotations

from .problems import Problem

FENCE = "```"


def prompt(problem: Problem) -> str:
    return f"Complete the following Lean 4 code:\n\n{FENCE}lean4\n{problem.formal_statement}"


def proof(problem: Problem, completion: str) -> str:
    return problem.formal_statement + completion.split(FENCE, 1)[0]
