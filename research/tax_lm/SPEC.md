# tax-lm: a small, citation-checked model of US federal tax law

Status: **draft spec**, nothing implemented. Numbers marked *(est.)* are
order-of-magnitude guesses to be replaced by measurements in M0.

This track carries the lab's one idea into a second domain. In `prover_loop`
the model proposes and the Lean kernel decides. Tax law has no kernel, but it
has two things close to one:

1. **A citation resolver.** A cited provision either exists in a dated
   snapshot of the corpus and contains the quoted words, or it doesn't. That
   is decidable, cheap and model-free.
2. **A computation oracle.** A dollar figure for a fully specified fact
   pattern either matches an independent tax engine or it doesn't.

Neither proves an answer *correct*. Both prove an answer *grounded*, and they
are what the model is trained against and gated by at serve time. Everything
else (answer quality, reasoning) is measured, never trusted.

```text
question + as_of ─▶ retrieve(as_of) ─▶ model ─▶ answer JSON ─▶ verifier ─▶ shipped
                         ▲                          │            │ reject
                         │                          └── retry ◀──┘ (≤ n, then abstain)
   dated corpus snapshot ┘
   training: sample k ─▶ verifier ─▶ verified.jsonl ─▶ SFT/LoRA ─▶ re-serve   (same loop as prover_loop)
```

## 1. Scope

**Goals**

- G1. Answer questions about US federal tax law and IRS procedure, *as of a
  stated date*, with every legal claim tied to a verified citation and exact
  quote.
- G2. Compute simple liabilities/amounts through a tool call, not in the
  model's head.
- G3. Abstain (`"abstain": true`, with reason) rather than ship an ungrounded
  answer.
- G4. Run on the lab's CPU box (4 vCPU, 15 GB) under MAX at ≥ 5 tok/s; train
  on one rented GPU in hours, not days.

**Non-goals**

- State/local/foreign tax; tax planning advice; preparing or filing returns.
- Parametric recall as the knowledge store. A ≤ 3B model memorising a corpus
  that changes every year is the failure mode this design avoids: knowledge
  lives in the dated corpus, the model reads and cites it.
- Any taxpayer data, in training or in logs (see §10).

## 2. Authority model

Sources are ranked; the rank is data on every corpus node and the model is
trained to prefer the higher rank on conflict.

| Rank | Source | Binding on | Notes |
| --- | --- | --- | --- |
| 1 | Internal Revenue Code, 26 U.S.C. | everyone | |
| 2 | Treasury Regulations, 26 C.F.R. (final > temporary > proposed) | everyone | proposed regs are not law; tag them |
| 3 | Revenue Rulings, Revenue Procedures, Notices (Internal Revenue Bulletin) | IRS; taxpayers may rely | |
| 4 | Tax Court / federal court opinions | parties; precedent by circuit | M3+, optional |
| 5 | Internal Revenue Manual (IRM) | IRS employees only | procedure, **not law**; courts hold it confers no rights on taxpayers |
| 6 | Forms, instructions, publications | nobody | IRS's reading, useful and plain-language |
| 7 | PLRs, TAMs, CCAs (§6110 written determinations) | the requesting taxpayer only | not precedent; M3+, optional |

Invariant **A1**: an answer whose conclusion rests only on rank ≥ 5 must say so
(`"authority_note"`), e.g. "IRM procedure, not binding law".

## 3. Corpus

### 3.1 Sources

| Source | Where | Format | Point-in-time history |
| --- | --- | --- | --- |
| IRC | uscode.house.gov (OLRC) bulk download | USLM XML | OLRC release points (per public law); govinfo annual editions |
| Treas. Regs | eCFR API (ecfr.gov) | XML | eCFR point-in-time back to 2017; govinfo annual CFR before |
| IRB | irs.gov/irb | HTML (1996→), PDF older | issue date is the version |
| IRM | irs.gov/irm | HTML | per-section "effective date"; keep dated scrapes |
| Forms/pubs | irs.gov/forms-pubs, prior-year archive | PDF/HTML | revision date on the document |

All of these are US government works (17 U.S.C. §105), so redistribution of
the corpus and of models trained on it is unencumbered. Verify per source at
M0 anyway; court opinions and third-party annotations are *not* in scope for
that reason.

Size *(est.)*: IRC ≈ 5–10 M tokens, regs ≈ 20–40 M, IRM ≈ 30–80 M, IRB since
1996 ≈ 50–100 M, pubs/instructions for one year ≈ 10 M. Total per snapshot
≈ 100–250 M tokens; with annual history ×5–10.

### 3.2 Node schema

The corpus is a set of immutable nodes, one per smallest citable unit
(IRC paragraph, reg paragraph, IRM subsection, ruling section, pub section):

```json
{
  "id": "irc/61/a/1@2025-07-04",
  "cite": "26 U.S.C. § 61(a)(1)",
  "source": "irc", "rank": 1, "status": "final",
  "valid_from": "2025-07-04", "valid_to": null,
  "parent": "irc/61/a@2025-07-04",
  "heading": "Compensation for services, including fees, commissions, fringe benefits, and similar items",
  "text": "…", "sha256": "…",
  "provenance": {"url": "…", "retrieved": "2026-09-23", "release_point": "…"}
}
```

Invariants (property-tested in M0):

- **C1** `id` is unique; `sha256 = H(normalize(text))`.
- **C2** For a fixed canonical citation `c`, the intervals
  `[valid_from, valid_to)` of its versions are pairwise disjoint, so
  `resolve(c, t)` returns at most one node.
- **C3** Every node's `parent` resolves at the same `t`.
- **C4** `normalize` is idempotent: NFC, collapse whitespace, unify quotes and
  dashes; no other change to the text.

A **snapshot** `S_t` is `{n | n.valid_from ≤ t < n.valid_to}` and is
identified by its date and the hash of the node-hash list. Every run,
training row and answer records the snapshot hash it used.

### 3.3 Citation grammar

One grammar, used by the parser, the prompt, the verifier and the evaluator
(DRY: `tax_loop/cite.py` is the only place it lives).

```ebnf
cite     = irc | reg | irm | irb | pub ;
irc      = ("26 U.S.C." | "IRC") , " § " , section , { para } ;
reg      = "Treas. Reg. § " , part , "." , section , [ "-" , num , ["T"] ] , { para } ;
irm      = "IRM " , num , { "." , num } ;
irb      = ("Rev. Rul." | "Rev. Proc." | "Notice" | "Announcement") , " " , year , "-" , num , [ ", § " , num ] ;
pub      = ("Pub." | "Form" | "Instructions for Form") , " " , alnum , " (" , year , ")" , [ ", " , locator ] ;
section  = num , [ letters ] ;                 (* 45Z, 1400Z-2 handled as num "-" num *)
para     = "(" , ( letters | num | roman ) , ")" ;
```

`canonical(cite)` maps surface variants ("IRC §61(a)(1)", "sec. 61(a)(1)",
"26 USC 61(a)(1)") to one form; **G-rt**: `parse ∘ render = id` on canonical
forms, tested by round trip over every node's `cite`.

## 4. Verifier (the trusted base)

Small, deterministic, no model inside it. Everything else in this spec can be
wrong without corrupting data; this cannot, so it gets the tests.

### 4.1 Answer contract

```json
{
  "as_of": "2025-12-31",
  "answer": "…prose, every legal claim ends with [n]…",
  "citations": [{"n": 1, "cite": "26 U.S.C. § 63(c)(2)", "quote": "…exact words…"}],
  "computation": {"tool": "engine", "input": {…}, "output": {…}} ,
  "authority_note": null,
  "abstain": false, "abstain_reason": null
}
```

### 4.2 Acceptance

For answer `r` against snapshot `S = S_{r.as_of}`:

```text
accept(r) ⇔ well_formed(r)
          ∧ ( r.abstain ∧ r.abstain_reason ≠ ∅
            ∨ ¬r.abstain
              ∧ ∀ c ∈ r.citations : ∃! n ∈ S . n.cite = canonical(c.cite)
                                    ∧ normalize(c.quote) ⊑ normalize(n.text)    -- substring
                                    ∧ |tokens(c.quote)| ≥ 5
              ∧ markers(r.answer) = { c.n | c ∈ r.citations } ≠ ∅                -- no orphan, no unused
              ∧ (r.computation ≠ ∅ ⇒ engine(r.computation.input) = r.computation.output)
              ∧ A1(r) )
```

A missing field, an unparsable citation, a quote from a *different* version
of the right section, or an engine timeout is a rejection with a reason, as
in `prover_loop/verify.py` where a missing axiom report is a rejection.

What `accept` does **not** establish: that the quoted text supports the
claim, that the answer is complete, or that the law was read correctly.
Those are eval metrics (§7), never gates.

### 4.3 Computation oracle

One interface `engine(facts, tax_year) → amounts`, two independent back ends,
and a disagreement between them is logged and excluded, never resolved by
vote:

- IRS Direct File's open-sourced Fact Graph (US government work), for the
  1040 scope it covers;
- PolicyEngine-US or PSL Tax-Calculator (check licence at M0) as the second
  opinion.

The model never emits a final number that did not come out of `engine`
(**E1**, enforced by the contract: numbers in `answer` that are amounts must
appear in `computation.output`).

## 5. Model

| Item | Choice | Reason |
| --- | --- | --- |
| Base | `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0) | Qwen2 architecture, which MAX already runs on CPU here (README); 32k context; small enough for full fine-tuning on one 80 GB GPU |
| Fallback | `meta-llama/Llama-3.2-3B-Instruct` | Llama path = MAX's LoRA serving path; bigger, community licence |
| Serving | MAX, Q4_K GGUF on CPU, `--max-length 8192` | ≈ 1 GB weights *(est.)*, leaves room for retrieval index and engine |
| Retrieval | BM25 (exact cite/term hits matter) ∪ small dense embedder, filtered to `S_t` | statute text is keyword-heavy; dense alone misses "§ 1031" |

Tokenizer gate (as in `tools/prepare_model.py`): round-trip `§ ¶ — – “ ” ’ ½ ¢`
and a sample of every source through encode/decode; fail on any loss.

## 6. Training

Four stages; each writes a manifest (base, data SHA-256, snapshot hash,
hyperparameters, final loss) like `tools/train_lora.py` does.

**S1. Continued pretraining** (optional, gated by M2 ablation). Next-token on
the latest snapshot, mixed with ≈ 20–30 % general replay text to limit
forgetting. Budget: `6·N·D = 6 · 1.5e9 · 1e9 ≈ 9e18` FLOPs for one epoch-ish
over 1 B tokens ≈ 6–10 H100-hours *(est.)*. Purpose is register and citation
format, not memorisation; drop it if S2+S3 without it score within noise.

**S2. SFT on verified synthetic QA.** A larger teacher generates
`(question, as_of, retrieved nodes) → answer JSON` from sampled nodes and
node pairs (cross-references, reg-interprets-code, exception-to-rule).
Only `accept`ed answers become rows. Row policy reuses `prover_loop/sft.py`:
dedup, cap per source node, prompt masked, and contamination is an error.
Include: ≥ 10 % must-abstain questions (out of scope, not in `S_t`,
future-dated); ≥ 10 % temporal pairs (same question, two `as_of`s straddling
an amendment, different answers).

**S3. Expert iteration against the verifier.** Exactly `prover_loop.run`:
sample k answers per question at temperature, keep the accepted ones,
LoRA/SFT on them, re-serve, repeat. Reward is binary `accept`, which the model
cannot game by fluency. Add an auxiliary pairwise judge for *support* (does
the quote entail the claim) only as a filter that can remove rows, never add
them.

**S4. Calibration.** Fit the abstain threshold on a dev split to a target
selective accuracy (§7), not to maximise coverage.

## 7. Evaluation

Held-out sets, never trained on, checked by the same `Contamination` rule
(normalised-question key + 13-gram overlap against every training row):

| Set | Measures | Ground truth |
| --- | --- | --- |
| SARA (StAtutory Reasoning Assessment) | statutory reasoning + liability | Prolog, ~2017 law: evaluate with `as_of = 2017-…`, a free temporal test |
| TaxCalcBench | return computation | benchmark's reference returns |
| tax-lm-temporal (built here) | right law for the date | pairs straddling Pub. L. 119-21 (2025) and other amendments; answers derived from the snapshots, not written by hand |
| tax-lm-abstain (built here) | refusals | out-of-scope, nonexistent sections, future dates |
| tax-lm-expert (built here, small) | answer correctness | ≥ 200 questions, gold by a credentialed reviewer (EA/CPA) |

Metrics: citation validity = accepted / emitted (serve-time gate makes the
*shipped* rate 1 by construction; the raw rate is the training signal);
citation recall vs. gold cites; answer accuracy; numeric exact match within
$1; selective accuracy at fixed coverage; temporal accuracy; abstain
precision/recall. Report base vs. each stage, same seeds, same `k`.

Baselines: base model + same RAG, no training; a frontier model + same RAG.

## 8. Code layout (proposed)

Mirror `prover_loop/` so the loop, rows and manifests stay one shape:

```text
tax_loop/
  corpus.py     fetch + normalise + node schema, snapshot(t)
  cite.py       grammar, canonical(), resolve(cite, t)
  retrieve.py   BM25 ∪ dense over S_t
  engine.py     one interface, two back ends
  verify.py     accept() → Verdict(ok, reason)
  prompt.py     prompt(question, as_of, nodes) / parse answer JSON
  run.py        k samples → attempts.jsonl, verified.jsonl, summary.json
tests/test_tax_*.py
```

`prover_loop/sft.py`, `run.py` and `tools/train_lora.py` should be generalised
over a `Problem`/`Verifier` protocol rather than copied.

## 9. Milestones (test-first: each exit criterion is a failing test before it is code)

| M | Deliverable | Exit criterion |
| --- | --- | --- |
| M0 | IRC + regs snapshot for two dates, node schema, grammar | C1–C4 and G-rt pass over every node; measured token counts replace *(est.)* |
| M1 | `verify.accept`, `engine` stub over one back end | property tests: mutating one char of a quote ⇒ reject; quoting the other version ⇒ reject; hand cases for every reject reason |
| M2 | Baselines on all eval sets, S1 ablation | table in `results/tax-lm/`; decision on S1 |
| M3 | S2 + one S3 round, served on CPU under MAX | raw citation validity and temporal accuracy beat base + RAG on held-out sets; ≥ 5 tok/s on the CPU box |
| M4 | IRM, IRB, pubs; second engine; expert set | expert-set accuracy reported with CI; abstain calibrated |

## 10. Risks and rules

- **Stale law.** Mitigated by design (dated retrieval, `as_of` in every
  answer); the residual risk is snapshot lag. Every answer shows its snapshot
  date.
- **Grounded but wrong.** `accept` allows a real quote attached to a wrong
  conclusion. Measured by the support judge and the expert set; stated
  plainly in the UI.
- **Not advice.** Output is informational research, labelled as such; no
  Circular 230 representation is made.
- **Taxpayer data.** None in training, eval or logs. If a user supplies facts,
  they go to `engine` in memory and are not persisted; a preparer deploying
  this carries §7216 obligations the lab does not.
- **Proposed regs and IRM** are tagged by rank so the model cannot present
  them as law (A1).

## 11. Open questions

1. S1 or not: does domain CPT move citation validity beyond what S2/S3 give?
   (M2 decides.)
2. Granularity of nodes for long reg paragraphs: quote-length floor of 5
   tokens vs. paragraph size.
3. Court opinions (rank 4): worth the licensing/cleaning cost for a small
   model, or leave to retrieval over CourtListener at serve time?
4. Can a slice of the computation rules (e.g. §1(j) brackets, §63 standard
   deduction) be stated in Lean, making the oracle a proof rather than a
   second implementation?
