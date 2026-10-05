"""micro-train: train a micro causal LM or text classifier from scratch.

    micro-train lm       --data corpus.txt --out runs/lm
    micro-train classify --train train.jsonl --eval eval.jsonl --out runs/clf

Each run writes model.safetensors, config.json and metrics.json to --out.
Seeded: the same arguments on the same device give the same metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from .data import ByteTokenizer, load_labelled, split_text
from .model import MicroConfig, MicroTransformer

TOKENIZER = ByteTokenizer()


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def save(model: MicroTransformer, out: Path, mode: str, metrics: dict, extra: dict | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, out / "model.safetensors")
    (out / "config.json").write_text(json.dumps({"mode": mode, "tokenizer": "bytes", **model.config.to_dict(),
                                                 **(extra or {})}, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def windows(ids: torch.Tensor, context: int, batch: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(ids) - context - 1, (batch,), generator=gen)
    x = torch.stack([ids[s:s + context] for s in starts])
    return x, torch.stack([ids[s + 1:s + context + 1] for s in starts])


@torch.no_grad()
def bits_per_byte(model: MicroTransformer, ids: torch.Tensor, context: int, dev: torch.device) -> float:
    """Mean next-byte cross-entropy over non-overlapping windows, in bits."""
    model.eval()
    total, count = 0.0, 0
    for s in range(0, len(ids) - 1, context):
        x, y = ids[s:s + context], ids[s + 1:s + context + 1]
        x = x[:len(y)]
        if len(y) == 0:
            break
        total += F.cross_entropy(model(x[None].to(dev))[0], y.to(dev), reduction="sum").item()
        count += len(y)
    model.train()
    return total / count / math.log(2)


def train_lm(text: str, config: MicroConfig, steps: int, batch_size: int, lr: float, seed: int, out: Path,
             held_out: float = 0.1) -> dict:
    train_text, held_text = split_text(text, held_out)
    train_ids, held_ids = (torch.tensor(TOKENIZER.encode(t), dtype=torch.long) for t in (train_text, held_text))
    if len(train_ids) <= config.context + 1 or len(held_ids) < 2:
        raise ValueError(f"corpus too small for context {config.context}: {len(train_ids)} training bytes")
    gen, dev = seed_everything(seed), device()
    model = MicroTransformer(config, n_classes=None).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        x, y = windows(train_ids, config.context, batch_size, gen)
        loss = F.cross_entropy(model(x.to(dev)).flatten(0, 1), y.to(dev).flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    metrics = {"mode": "lm", "steps": steps, "seed": seed, "train_bytes": len(train_ids), "held_out_bytes": len(held_ids),
               "final_train_loss": loss.item(), "held_out_bits_per_byte": bits_per_byte(model, held_ids, config.context, dev),
               "device": dev.type}
    return save(model, out, "lm", metrics)


def batch_of(texts: list[str], context: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Byte ids truncated to `context`, right-padded with 0, and the padding mask."""
    encoded = [TOKENIZER.encode(t)[:context] or [0] for t in texts]
    width = max(map(len, encoded))
    ids = torch.zeros(len(encoded), width, dtype=torch.long)
    pad = torch.ones(len(encoded), width, dtype=torch.bool)
    for i, e in enumerate(encoded):
        ids[i, :len(e)], pad[i, :len(e)] = torch.tensor(e), False
    return ids, pad


@torch.no_grad()
def accuracy(model: MicroTransformer, texts: list[str], labels: list[int], context: int, dev: torch.device) -> float:
    model.eval()
    ids, pad = batch_of(texts, context)
    predicted = model(ids.to(dev), pad.to(dev)).argmax(-1).cpu()
    model.train()
    return (predicted == torch.tensor(labels)).float().mean().item()


def train_classifier(train: Path, held: Path, config: MicroConfig, steps: int, batch_size: int, lr: float, seed: int,
                     out: Path) -> dict:
    texts, labels, names = load_labelled(train)
    eval_texts, eval_labels, eval_names = load_labelled(held)
    unknown = set(eval_names) - set(names)
    if unknown:
        raise ValueError(f"eval labels never seen in training: {sorted(unknown)}")
    eval_labels = [names.index(eval_names[i]) for i in eval_labels]
    gen, dev = seed_everything(seed), device()
    model = MicroTransformer(config, n_classes=len(names)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    targets = torch.tensor(labels)
    for _ in range(steps):
        pick = torch.randint(0, len(texts), (batch_size,), generator=gen)
        ids, pad = batch_of([texts[i] for i in pick], config.context)
        loss = F.cross_entropy(model(ids.to(dev), pad.to(dev)), targets[pick].to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
    metrics = {"mode": "classify", "steps": steps, "seed": seed, "labels": names, "train_rows": len(texts),
               "eval_rows": len(eval_texts), "final_train_loss": loss.item(),
               "eval_accuracy": accuracy(model, eval_texts, eval_labels, config.context, dev), "device": dev.type}
    return save(model, out, "classify", metrics, {"labels": names})


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(prog="micro-train", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("lm", "classify"):
        p = sub.add_parser(name)
        if name == "lm":
            p.add_argument("--data", type=Path, required=True, help="UTF-8 text corpus")
        else:
            p.add_argument("--train", type=Path, required=True, help="JSONL rows {text, label}")
            p.add_argument("--eval", type=Path, required=True, help="JSONL rows {text, label}")
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--steps", type=int, default=1000)
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--lr", type=float, default=3e-4)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--d-model", type=int, default=128)
        p.add_argument("--layers", type=int, default=4)
        p.add_argument("--heads", type=int, default=4)
        p.add_argument("--context", type=int, default=256)
    args = parser.parse_args(argv)
    config = MicroConfig(args.d_model, args.layers, args.heads, args.context)
    try:
        if args.mode == "lm":
            metrics = train_lm(args.data.read_text(encoding="utf-8"), config, args.steps, args.batch_size, args.lr,
                               args.seed, args.out)
        else:
            metrics = train_classifier(args.train, args.eval, config, args.steps, args.batch_size, args.lr, args.seed,
                                       args.out)
    except ValueError as e:
        parser.exit(2, f"micro-train: {e}\n")
    print(json.dumps(metrics, indent=2))
    return metrics


def cli() -> None:
    """Console entry point: `main` returns the metrics, which must not become the exit status."""
    main()
