#!/usr/bin/env python3
"""Assemble a local model directory MAX can serve on CPU.

DeepSeek-Prover-V1.5's tokenizer_config.json names ``LlamaTokenizerFast``,
and current transformers then loads its byte-level BPE ``tokenizer.json`` as
a SentencePiece tokenizer: spaces and every non-ASCII symbol (ℕ, ℝ, ≤, ...)
are lost on the round trip, which silently corrupts Lean. The fix is one
field: name the generic ``PreTrainedTokenizerFast`` instead. Everything else
is the upstream file, and the weights are a Q4_K GGUF so the 7B model fits a
CPU box.

Usage:
    prepare_model.py [--out models/deepseek-prover-v1.5-rl]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL = "deepseek-ai/DeepSeek-Prover-V1.5-RL"
GGUF_REPO = "mradermacher/DeepSeek-Prover-V1.5-RL-GGUF"
GGUF_FILE = "DeepSeek-Prover-V1.5-RL.Q4_K_M.gguf"
PROBE = "theorem two : (1:ℕ) + 1 ≤ 2 := by"


def prepare(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer.json"):
        shutil.copy(hf_hub_download(MODEL, name), out / name)
    config = json.loads(Path(hf_hub_download(MODEL, "tokenizer_config.json")).read_text(encoding="utf-8"))
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    (out / "tokenizer_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    weights = out / GGUF_FILE
    if not weights.exists():
        weights.symlink_to(hf_hub_download(GGUF_REPO, GGUF_FILE))
    return out


def round_trips(model_dir: Path) -> bool:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    return tok.decode(tok.encode(PROBE, add_special_tokens=False)) == PROBE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("models/deepseek-prover-v1.5-rl"))
    out = prepare(parser.parse_args().out)
    if not round_trips(out):
        print(f"FAIL: tokenizer in {out} does not round-trip {PROBE!r}")
        return 1
    print(f"OK: {out} (weights {GGUF_FILE}); tokenizer round-trips Lean symbols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
