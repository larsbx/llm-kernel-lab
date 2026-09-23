# tax-lm: a small, calibrated, citation-checked model of US federal tax law

Status: **draft spec**, nothing implemented. Numbers marked *(est.)* are
order-of-magnitude guesses to be replaced by measurements in M0.

This track carries the lab's one idea into a second domain. In `prover_loop`
the model proposes and the Lean kernel decides. Tax law has no kernel, but it
has three things close to one:

1. **A citation resolver.** A provision either exists in a dated snapshot of
   the corpus and contains the cited span, or it doesn't. Decidable, cheap,
   model-free.
2. **A computation oracle.** An amount for a fully specified fact pattern
   either matches an independent tax engine or it doesn't.
3. **Oracle outcomes.** For a synthetic fact pattern the engine *knows* the
   answer to "is this a qualifying child?", "which filing status?", "is the
   §121 exclusion available?". Those are labels, not opinions.

The design also takes what made TypeSafe's **Jev** useful (§1.1): typed
decisions instead of prose, calibrated probabilities, many questions against
one state in one pass, criteria given at call time, and arithmetic, dates and
literal extraction kept out of the model.

```text
fact state + as_of ─▶ retrieve(as_of) ─▶ criteria nodes ─┐
                                                         ▼
      typed questions (bool | choice | score) ─▶ decision pass (one forward, no sampling)
                                                         │  p, confidence, evidence spans
                                                         ▼
                         verifier (spans resolve, types check, engine agrees) ─▶ shipped
                                                         │ confidence < τ
                                                         └─▶ abstain / route to a human
   training: engine-labelled fact patterns ─▶ proper-scoring-rule loss ─▶ calibrate ─▶ re-serve
```

## 1. Scope

**Goals**

- G1. Answer typed questions about US federal tax law and IRS procedure, *as
  of a stated date*, each with a calibrated probability and the corpus spans
  it rests on.
- G2. Amounts and dates come from code (the engine, a date library), never
  from the model.
- G3. Abstain or hand off to a human below a confidence threshold, instead of
  shipping a guess.
- G4. Run on the lab's CPU box (4 vCPU, 15 GB) under MAX; a decision call in
  ≤ 1 s *(target)*; train on one rented GPU in hours.
- G5. Explanations in prose are optional, derived from G1's decisions and
  spans, and subject to the same verifier (§4.3).

**Non-goals**

- State/local/foreign tax; planning advice; preparing or filing returns.
- Parametric recall as the knowledge store. Law lives in the dated corpus;
  the model reads it.
- Any taxpayer data, in training or in logs (§10).

### 1.1 What is taken from Jev, and what is not

Jev (TypeSafe AI, early access 15 Sep 2026) takes a state plus named
questions of three types, `noul` (yes/no → p), `choice` (≤ 255 options →
distribution) and `score` (2–10 levels → distribution + mean), and returns
only those typed values with calibrated probabilities. Its criteria are
natural language at call time on fixed weights, it answers all questions in
one parallel pass, and it is trained on synthetic data with "RL for
Calibrated Decisions". Its documented weak spots are arithmetic, dates,
literal extraction and generated output.

| Jev property | In tax-lm | Why it fits tax |
| --- | --- | --- |
| Typed outputs, no free text | the decision contract (§4.1); prose is a separate, optional mode | most tax questions are tests with discrete outcomes; a type error is impossible, not unlikely |
| Calibrated p ("90 % means right 90 % of the time") | trained with a proper scoring rule on engine labels, then temperature-scaled (§6); ECE and Brier are release gates (§7) | the abstain/hand-off decision needs a probability that means something |
| All questions in one pass, independent | shared state prefix, one short suffix per question, batched with prefix caching (§5.2) | eligibility is a checklist: §152 dependency tests, EITC, §121, filing status, run together |
| Criteria at call time, fixed weights | the criteria **are the retrieved statute text**, passed with the question | the law changes yearly; weights shouldn't |
| Choose from candidates, don't extract literally | citations are node IDs from the retrieved set and quotes are span offsets into them | a fabricated citation or misquote cannot be expressed |
| Numbers/dates handed to code | engine for amounts, a date library for deadlines and `as_of` | same weak spots in a small model, same fix |
| Synthetic training data | fact patterns sampled and labelled by the engine (§6.1) | unlimited, labelled, privacy-free |

Not taken: Jev's opacity. It returns bare floats, which is its main
criticism for high-stakes use. Every tax-lm decision carries the evidence
spans it read, and the verifier checks them.

Jev's RLCD algorithm is unpublished. Where the outcome is observed directly,
as with an engine label, RL is unnecessary: log loss is a proper scoring rule,
so plain supervised training already targets calibration in distribution.
RL with a log-score reward is kept for the one case that needs it, a decision
reached through sampled intermediate steps (§6.3).

## 2. Authority model

Sources are ranked; the rank is data on every corpus node.

| Rank | Source | Binding on | Notes |
| --- | --- | --- | --- |
| 1 | Internal Revenue Code, 26 U.S.C. | everyone | |
| 2 | Treasury Regulations, 26 C.F.R. (final > temporary > proposed) | everyone | proposed regs are not law; tag them |
| 3 | Revenue Rulings, Revenue Procedures, Notices (Internal Revenue Bulletin) | IRS; taxpayers may rely | |
| 4 | Tax Court / federal court opinions | parties; precedent by circuit | M4+, optional |
| 5 | Internal Revenue Manual (IRM) | IRS employees only | procedure, **not law**; courts hold it confers no rights on taxpayers |
| 6 | Forms, instructions, publications | nobody | IRS's reading, plain-language |
| 7 | PLRs, TAMs, CCAs (§6110 written determinations) | the requesting taxpayer only | not precedent; M4+, optional |

Invariant **A1**: a decision whose evidence is only rank ≥ 5 carries
`authority: "non-binding"`.

## 3. Corpus

### 3.1 Sources

| Source | Where | Format | Point-in-time history |
| --- | --- | --- | --- |
| IRC | uscode.house.gov (OLRC) bulk download | USLM XML | OLRC release points; govinfo annual editions |
| Treas. Regs | eCFR API (ecfr.gov) | XML | eCFR point-in-time back to 2017; govinfo annual CFR before |
| IRB | irs.gov/irb | HTML (1996→), PDF older | issue date is the version |
| IRM | irs.gov/irm | HTML | per-section effective date; keep dated scrapes |
| Forms/pubs | irs.gov/forms-pubs, prior-year archive | PDF/HTML | revision date on the document |

All are US government works (17 U.S.C. §105), so the corpus and models
trained on it can be redistributed; verify per source at M0.

Size *(est.)*: IRC ≈ 5–10 M tokens, regs ≈ 20–40 M, IRM ≈ 30–80 M, IRB since
1996 ≈ 50–100 M, one year of pubs/instructions ≈ 10 M. ≈ 100–250 M tokens per
snapshot; ×5–10 with annual history.

### 3.2 Node schema

One immutable node per smallest citable unit:

```json
{
  "id": "irc/152/c/1@2025-07-04",
  "cite": "26 U.S.C. § 152(c)(1)",
  "source": "irc", "rank": 1, "status": "final",
  "valid_from": "2025-07-04", "valid_to": null,
  "parent": "irc/152/c@2025-07-04",
  "text": "…", "sha256": "…",
  "provenance": {"url": "…", "retrieved": "…", "release_point": "…"}
}
```

Invariants (property-tested in M0):

- **C1** `id` unique; `sha256 = H(normalize(text))`.
- **C2** for a canonical citation `c`, the intervals `[valid_from, valid_to)`
  of its versions are pairwise disjoint, so `resolve(c, t)` has ≤ 1 result.
- **C3** `parent` resolves at the same `t`.
- **C4** `normalize` is idempotent (NFC, whitespace, quote/dash unification,
  nothing else).

Snapshot `S_t = {n | n.valid_from ≤ t < n.valid_to}`, identified by date and
the hash of its node-hash list. Every run, row and response records it.

### 3.3 Citation grammar

One grammar in one module (`tax_loop/cite.py`), used by the parser, the
renderer and the evaluator:

```ebnf
cite     = irc | reg | irm | irb | pub ;
irc      = ("26 U.S.C." | "IRC") , " § " , section , { para } ;
reg      = "Treas. Reg. § " , part , "." , section , [ "-" , num , ["T"] ] , { para } ;
irm      = "IRM " , num , { "." , num } ;
irb      = ("Rev. Rul." | "Rev. Proc." | "Notice" | "Announcement") , " " , year , "-" , num , [ ", § " , num ] ;
pub      = ("Pub." | "Form" | "Instructions for Form") , " " , alnum , " (" , year , ")" , [ ", " , locator ] ;
section  = num , [ letters ] , [ "-" , num ] ;
para     = "(" , ( letters | num | roman ) , ")" ;
```

**G-rt**: `parse ∘ render = id` on canonical forms, over every node's `cite`.
The model never writes a citation string: it picks a node ID and `render`
produces the text.

## 4. Contracts and verifier

### 4.1 Decision request and response

The request shape follows Jev's (state + named typed questions), with tax
additions: `as_of`, and criteria that are corpus nodes rather than free text.

```json
{
  "as_of": "2025-12-31",
  "state": {"household": "…free text or structured facts…"},
  "questions": {
    "qc_age": {"type": "bool",   "instructions": "Does Maya meet the age test?",
               "criteria": ["irc/152/c/3@…"]},
    "status": {"type": "choice", "instructions": "Filing status for 2025?",
               "options": ["single", "mfj", "mfs", "hoh", "qss"], "criteria": ["irc/2/b@…", "irc/1/…"]},
    "risk":   {"type": "score",  "instructions": "How well documented is the residency test?",
               "levels": ["none", "weak", "adequate", "strong"]}
  }
}
```

`criteria` may be omitted; retrieval over `S_{as_of}` then fills it. The
resolved criteria IDs are returned either way.

```json
{
  "snapshot": "2025-12-31/9f2c…",
  "answers": {
    "qc_age": {"p": 0.97, "evidence": [{"node": "irc/152/c/3@…", "span": [0, 212]}]},
    "status": {"choice": "hoh", "dist": {"single": 0.06, "mfj": 0.0, "mfs": 0.0, "hoh": 0.93, "qss": 0.01},
               "confidence": 0.93, "evidence": […]},
    "risk":   {"level": "adequate", "dist": {…}, "mean": 2.1, "confidence": 0.58, "evidence": […]}
  },
  "abstain": ["risk"],
  "authority": {"qc_age": "binding", "status": "binding", "risk": "binding"}
}
```

Types: `bool → p ∈ [0,1]`; `choice → dist ∈ Δ(options)`, `|options| ≤ 64`,
`confidence = max dist`; `score → dist ∈ Δ(levels)`, `2 ≤ |levels| ≤ 10`,
`mean = Σ i·dist_i`. A question is in `abstain` when its confidence
(`max(p, 1−p)` for bool) is below `τ_q`, which the caller may set per
question (stakes-dependent, as Jev's docs advise 0.5–0.9).

### 4.2 Acceptance

For response `r` to request `q`, with `S = S_{q.as_of}`:

```text
accept(r) ⇔ well_typed(r, q)                                   -- every answer has q's type; dists sum to 1 ± 1e-6
          ∧ ∀ a ∈ r.answers, ∀ (id, [i, j)) ∈ a.evidence :
                id ∈ retrieved(q) ∧ id ∈ S ∧ 0 ≤ i < j ≤ |S[id].text|
          ∧ ∀ a : a.evidence ≠ ∅                                -- no decision without a read
          ∧ A1(r)
          ∧ ∀ amounts/dates in r : produced by engine/date code (E1)
```

With spans and IDs chosen from candidates, the old failure modes (misquote,
wrong version, invented section) are unrepresentable. `accept` is still run,
because a malformed or out-of-range answer is a bug to catch, not to trust,
and a missing or timed-out check is a rejection with a reason, as in
`prover_loop/verify.py`.

`accept` does **not** establish that the decision is right or that the
evidence supports it. That is measured (§7).

### 4.3 Explain mode (optional)

Prose over already-accepted decisions: the generator is constrained to the
answer JSON grammar, and every `[n]` marker must refer to an evidence span
from §4.1, rendered verbatim. Same verifier plus `markers = cited ≠ ∅`.
Explain mode never introduces a decision or number the decision pass didn't
produce.

### 4.4 Computation oracle

`engine(facts, tax_year) → amounts ∪ labels`, two independent back ends; a
disagreement is logged and the case excluded, never voted:

- IRS Direct File's open-sourced Fact Graph (US government work), for the
  1040 scope it covers;
- PolicyEngine-US or PSL Tax-Calculator (check licence at M0).

It provides amounts at serve time and labels at training time (§6.1).

## 5. Model

### 5.1 Base and serving

| Item | Choice | Reason |
| --- | --- | --- |
| Base | `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0) | Qwen2 architecture, which MAX already runs on CPU here (README); 32k context; full fine-tune fits one 80 GB GPU |
| Fallback | `meta-llama/Llama-3.2-3B-Instruct` | Llama = MAX's LoRA serving path; community licence |
| Serving | MAX, Q4_K GGUF on CPU, `--max-length 8192`, prefix caching on | ≈ 1 GB weights *(est.)* |
| Retrieval | BM25 ∪ small dense embedder, filtered to `S_t` | statute text is keyword-heavy; dense alone misses "§ 1031" |

Tokenizer gate as in `tools/prepare_model.py`: round-trip `§ ¶ — – “ ” ’ ½`
and a sample of every source; fail on any loss.

### 5.2 Decision pass

No sampling. For each question the input is

```text
[state] [criteria text, spans numbered] [question] [options as reserved single tokens] <answer>
└──────────── shared prefix (cached) ──────────┘ └──────── per-question suffix ────────┘
```

and the readout is the logits at `<answer>` restricted to the question's
option tokens, softmaxed with the fitted temperature `T`. Questions share the
cached prefix and run as one batch of short suffixes, so they are independent
(no question sees another's answer) and cost ≈ one prefix plus n short passes.

Evidence: a second readout at `<evidence>` over the numbered span tokens,
top spans above a floor. Both readouts are single forward positions, so
latency is prefill-bound.

Serve path: MAX if its completions endpoint returns logprobs over a
constrained token set (check at M0); otherwise a thin scorer (llama.cpp or
PyTorch) behind the same interface. The interface, not the runtime, is the
contract.

## 6. Training

Each stage writes a manifest (base, data SHA-256, snapshot hash,
hyperparameters, final loss, fitted `T`), as `tools/train_lora.py` does.

### 6.1 Data: engine-labelled fact patterns

Sample structured households and transactions from a generative spec (ages,
relationships, residency months, income by type, support fractions,
property holding periods …), render each as varied natural-language states,
and ask the engine for labels: eligibility booleans, filing status,
applicable form or section (choice), amounts (kept for E1 tests, not trained
as outputs). Oversample near thresholds (age 18/19/23/24, 6 months residency,
2-of-5 years, phase-out edges) where calibration matters.

Every question's criteria are the real nodes the engine's rule implements,
so the model learns to read the statute it is given, not to memorise it.
Temporal pairs (same facts, `as_of` straddling an amendment such as
Pub. L. 119-21) are ≥ 10 % of rows. Questions the engine can't label
(documentation quality, reasonableness) come from a teacher model, are
tagged `label_source: "teacher"`, and are excluded from calibration metrics.

### 6.2 Stages

**S1. Continued pretraining** (optional, M2 ablation decides). Next-token on
the latest snapshot plus ≈ 20–30 % general replay. `6·N·D ≈ 6 · 1.5e9 · 1e9
≈ 9e18` FLOPs ≈ 6–10 H100-hours *(est.)*.

**S2. Decision training.** Cross-entropy on the option token (log loss, a
proper scoring rule) and on evidence spans, over §6.1 rows. No label
smoothing (it biases calibration). Contamination check reuses the rule in
`prover_loop/sft.py`: an eval state's key or a 13-gram overlap in training
data is an error, not a filter.

**S3. Calibration.** Fit temperature `T` on a held-out dev split by
minimising NLL; report reliability diagrams per question family. Fit `τ_q`
defaults for a target selective accuracy.

**S4. Expert iteration for explain mode** (optional). Exactly
`prover_loop.run`: sample k explanations, keep the accepted ones, SFT,
re-serve.

### 6.3 When RL is used

Only for decisions reached through sampled intermediate steps (e.g. a
multi-hop question answered after an explain-mode trace). Reward is the log
score `log p(label)` of the final typed answer against the engine label,
which is proper, so RL cannot improve reward by becoming over- or
under-confident. Everything else is supervised.

## 7. Evaluation

Held-out, never trained on, guarded by the same `Contamination` rule. Eval
generators use disjoint templates and disjoint random seeds from training
(Jev's "not in our training distribution" discipline).

| Set | Measures | Ground truth |
| --- | --- | --- |
| tax-lm-decisions (built here) | accuracy, calibration | engine labels on held-out templates |
| tax-lm-temporal (built here) | right law for the date | pairs straddling amendments, labels from each snapshot |
| tax-lm-abstain (built here) | refusals | out-of-scope, nonexistent provisions, future dates |
| SARA | statutory reasoning | Prolog, ~2017 law: evaluate with a 2017 `as_of` |
| TaxCalcBench | return computation (via engine calls) | reference returns |
| tax-lm-expert (built here, small) | real-world correctness | ≥ 200 questions graded by an EA/CPA |

Metrics: accuracy; **ECE** (15 bins) and **Brier**, per question family;
selective accuracy vs. coverage curve and its area; evidence precision/recall
against the engine rule's nodes; temporal accuracy; abstain precision/recall;
latency p50/p95 for a 10-question call on the CPU box.

Release gates (M3): ECE ≤ 0.05 and Brier below base + same prompt on
tax-lm-decisions; selective accuracy ≥ 0.95 at coverage reported alongside;
no accepted response with an out-of-set evidence ID (should be 0 by
construction; a non-zero count is a bug).

Baselines: base model with the same readout, untrained; a frontier model with
the same questions via structured output.

## 8. Code layout (proposed)

Mirror `prover_loop/` so loops, rows and manifests stay one shape:

```text
tax_loop/
  corpus.py     fetch + normalise + node schema, snapshot(t)
  cite.py       grammar, canonical(), render(), resolve(cite, t)
  retrieve.py   BM25 ∪ dense over S_t
  engine.py     one interface, two back ends; labels + amounts
  facts.py      fact-pattern generator + NL rendering (§6.1)
  decide.py     request → prefix/suffix batch → typed answers (§5.2)
  verify.py     accept() → Verdict(ok, reason)
  calibrate.py  fit T, τ_q; ECE, Brier, reliability
  run.py        attempts.jsonl, verified.jsonl, summary.json
tests/test_tax_*.py
```

`prover_loop/sft.py`, `run.py` and `tools/train_lora.py` should be generalised
over a `Problem`/`Verifier` protocol rather than copied.

## 9. Milestones (test-first: each exit criterion is a failing test first)

| M | Deliverable | Exit criterion |
| --- | --- | --- |
| M0 | IRC + regs snapshots for two dates; node schema; grammar; logprob readout check on MAX | C1–C4 and G-rt pass over every node; measured sizes replace *(est.)*; serve path chosen |
| M1 | `engine` (one back end), `facts`, `verify` | property tests: out-of-set ID ⇒ reject; span past end ⇒ reject; dist not summing to 1 ⇒ reject; two back ends disagree ⇒ excluded |
| M2 | baselines on all sets; S1 ablation | table in `results/tax-lm/`; S1 kept or dropped |
| M3 | S2 + S3 on the §152/§2/§32/§121 families, served on CPU | release gates of §7; 10-question call p95 ≤ 1 s *(target)* |
| M4 | IRM, IRB, pubs; second engine; explain mode; expert set | expert-set accuracy with CI; calibration holds on expert set within reported ECE |

## 10. Risks and rules

- **Stale law.** Dated retrieval and `as_of` everywhere; residual risk is
  snapshot lag, shown with every response.
- **Calibrated on synthetic, miscalibrated on real.** Synthetic states are
  cleaner than real ones. The expert set measures the shift; `τ_q` defaults
  are set on it, not on synthetic dev.
- **Grounded but wrong.** Real spans can back a wrong decision. Measured by
  evidence recall against the engine rule's nodes and by the expert set.
- **Engine coverage.** Families the engine can't label have only teacher
  labels: reported separately, never in calibration gates.
- **Not advice.** Informational research, labelled as such; no Circular 230
  representation is made.
- **Taxpayer data.** None in training, eval or logs. Supplied facts stay in
  memory for the call. A preparer deploying this carries §7216 obligations
  the lab does not.

## 11. Open questions

1. S1 or not (M2 decides).
2. Option cardinality: 64 single tokens is enough for tax choices seen so
   far; hierarchical choice if a family needs more (Jev allows 255).
3. Can the decision and evidence readouts share one position without losing
   calibration?
4. Can a slice of the rules (§1(j) brackets, §63 standard deduction, §152
   tests) be stated in Lean, making the label oracle a proof rather than a
   second implementation?
