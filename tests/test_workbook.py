"""Lean Workbook -> training problems: deterministic, true, and disjoint from the eval set."""

from prover_loop.problems import Problem, statement_key
from tools.workbook_problems import HEADER, batch_source, select, to_problem, uncompiled

ROW = {"formal_statement": "theorem lean_workbook_7 (x : ℝ) (h : 0 < x) : 0 < x ^ 2  :=  by sorry",
       "natural_language_statement": "Show x^2 > 0.", "proof": ["positivity"], "split": "lean_workbook"}


def row(name: str, body: str = "(n : ℕ) : n + 0 = n", proof=("simp",)) -> dict:
    return {"formal_statement": f"theorem {name} {body} := by sorry", "natural_language_statement": "",
            "proof": list(proof), "split": "lean_workbook"}


def test_a_row_becomes_a_problem_whose_statement_ends_at_by():
    p = to_problem(ROW)
    assert p == Problem("lean_workbook_7", "Show x^2 > 0.",
                        HEADER + "theorem lean_workbook_7 (x : ℝ) (h : 0 < x) : 0 < x ^ 2 := by\n")
    assert p.theorem == "lean_workbook_7"


def test_statement_key_ignores_the_name_the_header_and_whitespace():
    a = Problem("a", "", "import Mathlib\n\ntheorem a (x : ℕ) :\n  x = x := by\n")
    b = Problem("b", "", "theorem   b (x : ℕ) : x = x := by\n")
    assert statement_key(a.formal_statement) == statement_key(b.formal_statement)


def test_selection_is_deterministic_proved_only_and_excludes_the_eval_set():
    rows = [row(f"w{i}", f"(n : ℕ) : n + {i} = {i} + n") for i in range(50)]
    rows.append(row("unproved", proof=()))
    held_out = {statement_key(to_problem(rows[3]).formal_statement)}
    chosen = select(rows, 10, held_out)
    assert chosen == select(list(reversed(rows)), 10, held_out), "order of the source does not matter"
    assert len(chosen) == 10 and "unproved" not in {p.name for p in chosen} and "w3" not in {p.name for p in chosen}


def test_batch_source_states_every_theorem_once_under_one_header_with_line_spans():
    problems = [to_problem(row("ok")), to_problem(row("bad", "(n : ℕ) : n + 0 = m"))]
    source, spans = batch_source(problems)
    lines = source.splitlines()
    assert source.startswith(HEADER) and source.count("import Mathlib") == 1
    for name, (first, last) in spans.items():
        assert lines[first - 1].startswith(f"theorem {name} ") and "sorry" in lines[last - 1]


def test_uncompiled_names_the_theorems_whose_statements_have_errors():
    spans = {"ok": (8, 9), "bad": (11, 12), "ok2": (14, 15)}
    output = ("x.lean:11:30: error: unknown identifier 'm'\n"
              "x.lean:8:8: warning: declaration uses 'sorry'\nx.lean:14:8: warning: declaration uses 'sorry'\n")
    assert uncompiled(output, spans) == {"bad"}
