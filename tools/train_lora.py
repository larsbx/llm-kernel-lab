#!/usr/bin/env python3
"""Fine-tune a LoRA adapter on verified proofs (one round of expert iteration).

Rows are the loop's (prompt, completion) pairs, loss on the completion only,
so the adapter learns to continue the exact prompt MAX serves. The data is
refused outright if any row states an evaluation problem. The adapter is
saved with a manifest naming the base, the data files and their SHA-256, the
hyperparameters and the final training loss.

Train on the full-precision base, not the GGUF the CPU server uses:
    pixi run -e train-cuda train-lora --data results/<run>/verified.jsonl --out adapters/round1

Usage:
    train_lora.py --data FILE [FILE ...] --out DIR [--base HF_ID] [--tokenizer DIR] ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prover_loop.sft import load_sft  # noqa: E402

BASE = "deepseek-ai/DeepSeek-Prover-V1.5-RL"
TOKENIZER = "models/deepseek-prover-v1.5-rl"  # prepare_model.py's fixed tokenizer
# What MAX serves: its Llama LoRA path (ModuleV3) accepts attention projections only
# and rejects an adapter that also touches the MLP (LOAD_INVALID_ADAPTER).
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, nargs="+", required=True, help="verified.jsonl file(s)")
    parser.add_argument("--eval-problems", type=Path, nargs="*", default=[Path("problems/minif2f_test.jsonl")])
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--tokenizer", default=TOKENIZER)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-per-problem", type=int, default=2)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rows = load_sft(args.data, args.eval_problems, args.max_per_problem)  # raises Contamination first

    import peft
    import torch
    import transformers
    import trl
    from datasets import Dataset

    cuda = torch.cuda.is_available()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16 if cuda else torch.float32)
    lora = peft.LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                           target_modules=TARGETS, task_type="CAUSAL_LM")
    config = trl.SFTConfig(
        output_dir=str(args.out / "checkpoints"), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=0.03, bf16=cuda,
        max_length=args.max_length, completion_only_loss=True, logging_steps=1, save_strategy="no",
        report_to=[], seed=args.seed, gradient_checkpointing=cuda,
    )
    trainer = trl.SFTTrainer(model=model, args=config, train_dataset=Dataset.from_list(rows),
                             processing_class=tokenizer, peft_config=lora)
    result = trainer.train()
    trainer.model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    manifest = {
        "base": args.base, "tokenizer": str(args.tokenizer), "rows": len(rows),
        "data": [{"path": str(p), "sha256": sha256(p)} for p in args.data],
        "eval_problems_excluded": [str(p) for p in args.eval_problems],
        "lora": {"r": args.rank, "alpha": args.alpha, "dropout": args.dropout, "targets": TARGETS},
        "training": {"lr": args.lr, "epochs": args.epochs, "max_steps": args.max_steps,
                     "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                     "max_length": args.max_length, "seed": args.seed, "device": "cuda" if cuda else "cpu"},
        "final_loss": result.training_loss, "steps": result.global_step,
        "versions": {m.__name__: m.__version__ for m in (torch, transformers, peft, trl)},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"adapter -> {args.out} ({len(rows)} rows, {result.global_step} steps, loss {result.training_loss:.4f})")
    return manifest


if __name__ == "__main__":
    main()
