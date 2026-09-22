"""Prompting and proof assembly: the statement is never the model's to change."""

from prover_loop.problems import Problem
from prover_loop.prompt import proof, prompt

STATEMENT = "import Mathlib\n\ntheorem t (x : ℕ) (h : x = 2) : x + x = 4 := by\n"
P = Problem(name="t", informal="", formal_statement=STATEMENT)


def test_prompt_ends_inside_an_open_lean_block_after_the_statement():
    assert prompt(P) == "Complete the following Lean 4 code:\n\n```lean4\n" + STATEMENT


def test_proof_is_the_statement_followed_by_the_completion_up_to_the_fence():
    assert proof(P, "  subst h\n  rfl\n```\ntheorem extra : False := sorry") == STATEMENT + "  subst h\n  rfl\n"


def test_proof_without_a_fence_keeps_the_whole_completion():
    assert proof(P, "  omega") == STATEMENT + "  omega"


def test_theorem_name_is_read_from_the_statement():
    assert P.theorem == "t"
