"""LoRA training end to end on a tiny Llama: data in, adapter and manifest out.

Needs the `train` environment (`pixi run -e train pytest tests/test_train_lora.py`);
elsewhere torch is absent and the test is skipped.
"""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")

from prover_loop.problems import Problem  # noqa: E402
from prover_loop.prompt import prompt  # noqa: E402
from prover_loop.sft import Contamination  # noqa: E402
from tools.train_lora import main  # noqa: E402

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def write_verified(path, problems):
    rows = [{"problem": p.name, "prompt": prompt(p), "completion": "  simp\n```"} for p in problems]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def problem(i: int) -> Problem:
    return Problem(f"w{i}", "", f"import Mathlib\n\ntheorem w{i} (n : ℕ) : n + {i} = {i} + n := by\n")


def test_trains_and_saves_an_adapter_with_its_manifest(tmp_path):
    data, evals, out = tmp_path / "verified.jsonl", tmp_path / "eval.jsonl", tmp_path / "adapter"
    write_verified(data, [problem(i) for i in range(4)])
    evals.write_text(json.dumps(problem(99).__dict__) + "\n", encoding="utf-8")
    main(["--data", str(data), "--eval-problems", str(evals), "--base", TINY, "--tokenizer", TINY,
          "--out", str(out), "--max-steps", "2", "--rank", "4"])
    assert (out / "adapter_config.json").exists() and (out / "adapter_model.safetensors").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["base"] == TINY and manifest["rows"] == 4 and manifest["lora"]["r"] == 4
    assert manifest["data"][0]["sha256"] and manifest["final_loss"] == manifest["final_loss"]  # not NaN


def test_refuses_to_train_on_the_eval_set(tmp_path):
    data, evals = tmp_path / "verified.jsonl", tmp_path / "eval.jsonl"
    write_verified(data, [problem(1)])
    evals.write_text(json.dumps(problem(1).__dict__) + "\n", encoding="utf-8")
    with pytest.raises(Contamination):
        main(["--data", str(data), "--eval-problems", str(evals), "--base", TINY, "--tokenizer", TINY,
              "--out", str(tmp_path / "adapter"), "--max-steps", "1"])
