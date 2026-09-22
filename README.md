# llm-kernel-lab

A small expert-iteration loop for Lean 4 theorem proving, with a prover
served by [MAX](https://docs.modular.com/max/) and every proof checked by the
Lean kernel.

```text
problems ─▶ max serve (prover) ─▶ k candidate proofs ─▶ Lean + Mathlib ─▶ verified.jsonl
   ▲                                                   #print axioms            │
   └────────────── next round: fine-tune on verified.jsonl, re-serve ◀──────────┘
```

Only the kernel decides what is proved. A candidate is the problem's
statement, untouched, followed by the model's completion up to the first
closing fence. It is accepted only if Lean exits cleanly with no error and
`#print axioms` reports nothing beyond `propext`, `Classical.choice` and
`Quot.sound`. That rejects `sorry`, `admit` and `native_decide` (which trusts
the compiler), including when a helper lemma smuggles one in, since axioms are
collected transitively. A missing report (`#exit`, a crash, a timeout) is a
rejection too.

## Run it

```sh
pixi run prepare-model      # local model dir: upstream config + fixed tokenizer + Q4_K GGUF
pixi run lean-setup         # Mathlib v4.15.0 build cache (needs elan on PATH)
pixi run serve &            # MAX on CPU, OpenAI-compatible, port 8000
pixi run prove --limit 8 --k 4 --out runs/first
pixi run test               # 19 tests; the Lean ones need lean-setup (REQUIRE_LEAN=1 makes them mandatory)
```

A run writes `attempts.jsonl` (every sample with the kernel's verdict and
reason), `verified.jsonl` (one `(prompt, completion)` row per distinct verified
proof, the SFT data for the next round) and `summary.json` (solved, pass@k).

## Choices, and why

| Piece | Choice | Reason |
| --- | --- | --- |
| Prover | `deepseek-ai/DeepSeek-Prover-V1.5-RL`, Q4_K_M GGUF | Whole-proof output (hundreds of tokens, not thousands of CoT); Llama architecture, which MAX runs on CPU and which is where MAX's LoRA serving lives |
| Tokenizer | upstream `tokenizer.json`, `tokenizer_class` → `PreTrainedTokenizerFast` | Current transformers loads the upstream config as SentencePiece and silently drops spaces and every non-ASCII symbol (ℕ ℝ ≤ ...). `prepare_model.py` checks the round trip and fails otherwise |
| Problems | miniF2F-test, 244 problems (`AI-MO/minif2f_test`) | Lean 4 statements pinned to the Mathlib era the prover was trained on |
| Lean | `leanprover/lean4:v4.15.0`, Mathlib `v4.15.0` | Same era, so the model's lemma names still resolve |

Kimina-Prover-Distill (0.6B/1.7B) is the better small prover on paper, but it
is Qwen3, which MAX does not run on CPU (older nightlies refuse without a GPU
and 26.7 segfaults). The Qwen2-based Kimina-Prover-Preview-Distill-1.5B runs,
at ~4 tokens/s, but its long reasoning traces make each attempt several
minutes on CPU. Both are one `--model` away on a GPU box.

## CPU numbers (4 vCPU, 15 GB, no GPU)

- Generation: ~5 tokens/s per stream; one short proof ≈ 20–60 s.
- Verification: ≈ 10–25 s per candidate (Mathlib import dominates).
- Memory: MAX takes ~55% (weights + KV cache), Lean ~3 GB. At MAX's default
  90%, Lean thrashes: the serve task caps it with `--device-memory-utilization`.

## Next round (not run here)

The tuning step needs PyTorch and a GPU; MAX serves models but has no
training API. Fine-tune a LoRA adapter on `verified.jsonl` against the
full-precision base (PEFT/TRL SFT), then serve it with
`max serve ... --enable-lora --lora-paths <name>=<dir>` and point
`--model <name>` at it. Whether MAX's LoRA path accepts a GGUF-quantized base
is untested; on a GPU serve the bf16 base instead.
