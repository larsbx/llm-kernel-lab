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
pixi run test               # the Lean tests need lean-setup (REQUIRE_LEAN=1 makes them mandatory)
pixi run -e train pytest tests   # training tests (tiny model, CPU)
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

## Next round: fine-tuning

Training data never comes from the benchmark. The loop is run on a
**Lean Workbook** draw (`problems/lean_workbook_train.jsonl`, 400 problems)
and miniF2F-test is kept for evaluation only.

```sh
# 1. training problems: proved-only, deterministic, eval-disjoint, re-elaborated here
python tools/workbook_problems.py lean_workbook.json --n 400 --lean
# 2. collect verified proofs on them
pixi run prove --problems problems/lean_workbook_train.jsonl --k 4 --out runs/workbook-k4
# 3. LoRA on the verified proofs (GPU)
pixi run -e train-cuda train-lora --data runs/workbook-k4/verified.jsonl --out adapters/round1
# 4a. serve the adapter natively (GPU)
max serve --model deepseek-ai/DeepSeek-Prover-V1.5-RL --prefer-module-v3 \
  --enable-lora --no-enable-prefix-caching --lora-paths round1=adapters/round1
# 4b. or merge it and serve a plain model (any device)
pixi run -e train python tools/merge_lora.py adapters/round1 models/round1-merged
# 5. evaluate on miniF2F-test, which the adapter never saw (4a; after 4b, --model is the merged dir)
pixi run prove --model round1 --k 4 --out runs/minif2f-test-round1
```

What each step guarantees, and what was measured here:

- **Selection.** Only statements Lean Workbook ships a proof for (so true),
  ordered by SHA-256 of the name, minus every statement whose
  `statement_key` matches miniF2F-test: two Workbook statements
  (`lean_workbook_9150`, `lean_workbook_plus_65025`) are exact duplicates of
  a test problem and are excluded. One batch Lean run drops statements that
  no longer elaborate under Mathlib v4.15: 13 of 500 here.
- **Training rows** (`prover_loop/sft.py`): deduplicated, at most two proofs
  per problem, and any row that states an evaluation problem raises
  `Contamination` rather than being filtered. The loss covers the completion
  (proof, closing fence, EOS) only; the prompt is masked.
- **Adapter** (`tools/train_lora.py`): LoRA r=16, α=32 on `q/k/v/o_proj`,
  saved with a manifest of base, data SHA-256, hyperparameters and final loss.
  Attention-only because that is what MAX's Llama LoRA path accepts; an
  adapter that touches the MLP is rejected as `LOAD_INVALID_ADAPTER`.
- **Serving.** MAX's LoRA kernel (SGMV) is GPU-only and needs the ModuleV3
  architecture with prefix caching off, so on CPU the adapter is merged
  instead. Train, save, merge and `max serve` were run end to end here on a
  tiny random Llama; the 7B adapter needs a ≥24 GB GPU (bf16 base).
