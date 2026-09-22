"""SFT rows from verified proofs: capped, deduplicated, and never from the eval set."""

import pytest

from prover_loop.problems import Problem, statement_key
from prover_loop.prompt import prompt
from prover_loop.sft import Contamination, sft_rows, statement_of

TRAIN = Problem("w1", "", "import Mathlib\n\ntheorem w1 (n : ℕ) : n + 0 = n := by\n")
EVAL = Problem("m1", "", "import Mathlib\n\ntheorem m1 (x : ℝ) (h : x = 2) : x * 2 = 4 := by\n")


def verified(problem: Problem, completion: str) -> dict:
    return {"problem": problem.name, "prompt": prompt(problem), "completion": completion}


def test_the_statement_is_recovered_from_the_prompt():
    assert statement_of(prompt(TRAIN)) == TRAIN.formal_statement


def test_rows_are_deduplicated_and_capped_per_problem_in_file_order():
    rows = [verified(TRAIN, c) for c in ["  simp\n```", "  simp\n```", "  omega\n```", "  rfl\n```"]]
    out = sft_rows(rows, eval_keys=set(), max_per_problem=2)
    assert [r["completion"] for r in out] == ["  simp\n```", "  omega\n```"]
    assert set(out[0]) == {"prompt", "completion"}


def test_a_row_from_the_eval_set_refuses_the_whole_build():
    renamed = Problem("w9", "", EVAL.formal_statement.replace("m1", "w9"))
    rows = [verified(TRAIN, "  simp\n```"), verified(renamed, "  subst h\n  norm_num\n```")]
    with pytest.raises(Contamination, match="w9"):
        sft_rows(rows, eval_keys={statement_key(EVAL.formal_statement)}, max_per_problem=2)
