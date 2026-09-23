#!/usr/bin/env python3
"""Assemble a local model directory MAX can serve on CPU.

DeepSeek-Prover-V1.5's tokenizer_config.json names ``LlamaTokenizerFast``,
and current transformers then loads its byte-level BPE ``tokenizer.json`` as
a SentencePiece tokenizer: spaces and every non-ASCII symbol (ℕ, ℝ, ≤, ...)
are lost on the round trip, which silently corrupts Lean. The fix is one
field: name the generic ``PreTrainedTokenizerFast`` instead. Everything else
is the upstream file. Weights are either a Q4_K GGUF, so the 7B model fits a
CPU box, or the upstream bf16 safetensors for a GPU (and for LoRA, which MAX
serves only on GPU).

Usage:
    prepare_model.py [--weights gguf|safetensors] [--out DIR]
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
DEFAULT_OUT = {"gguf": Path("models/deepseek-prover-v1.5-rl"), "safetensors": Path("models/deepseek-prover-v1.5-rl-bf16")}


def weight_files(repo_files: list[str], kind: str) -> list[str]:
    """The upstream weight files a `kind` directory links: every safetensors shard and its index."""
    chosen = sorted(f for f in repo_files if f.endswith(".safetensors") or f == "model.safetensors.index.json")
    if not chosen:
        raise ValueError(f"no {kind} weights among {sorted(repo_files)}")
    return chosen


def link(out: Path, name: str, source: str) -> None:
    target = out / name
    if not target.exists():
        target.symlink_to(source)


def prepare(out: Path, weights: str = "gguf") -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer.json"):
        shutil.copy(hf_hub_download(MODEL, name), out / name)
    config = json.loads(Path(hf_hub_download(MODEL, "tokenizer_config.json")).read_text(encoding="utf-8"))
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    (out / "tokenizer_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    if weights == "gguf":
        link(out, GGUF_FILE, hf_hub_download(GGUF_REPO, GGUF_FILE))
    else:
        from huggingface_hub import list_repo_files
        for name in weight_files(list_repo_files(MODEL), weights):
            link(out, name, hf_hub_download(MODEL, name))
    return out


def round_trips(model_dir: Path) -> bool:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    return tok.decode(tok.encode(PROBE, add_special_tokens=False)) == PROBE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", choices=sorted(DEFAULT_OUT), default="gguf")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = prepare(args.out or DEFAULT_OUT[args.weights], args.weights)
    if not round_trips(out):
        print(f"FAIL: tokenizer in {out} does not round-trip {PROBE!r}")
        return 1
    print(f"OK: {out} ({args.weights} weights); tokenizer round-trips Lean symbols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
