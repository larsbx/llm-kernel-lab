#!/usr/bin/env python3
"""Fold a LoRA adapter into its base and save a plain model MAX can serve.

MAX serves adapters natively only on GPU (its LoRA kernel, SGMV, is GPU-only,
and only on ModuleV3 architectures). A merged model needs neither: it is
served like any base model, on CPU or GPU, with prefix caching left on.

Usage:
    merge_lora.py ADAPTER_DIR OUT_DIR [--dtype bfloat16|float32]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("adapter", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers

    manifest = json.loads((args.adapter / "manifest.json").read_text(encoding="utf-8"))
    base = transformers.AutoModelForCausalLM.from_pretrained(manifest["base"], dtype=getattr(torch, args.dtype))
    merged = peft.PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out, safe_serialization=True)
    transformers.AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    (args.out / "merge.json").write_text(json.dumps({"adapter": str(args.adapter), "dtype": args.dtype} | manifest,
                                                    indent=2) + "\n", encoding="utf-8")
    print(f"merged {args.adapter} into {manifest['base']} -> {args.out}")
    return args.out


if __name__ == "__main__":
    main()
