# tax-lm × deontic circuits

Status: **draft**, companion to `SPEC.md`. Uses the types from
`larsbx/usul-al-fiqh-deontic-circuits` (ADQ v37 `deontic_obligation/6`, the
v38 contextual path lift, and the finite sheaf context site) as tax-lm's
normative layer.

## 0. The split

Both projects already agree on who decides what:

| | deontic circuits | tax-lm |
| --- | --- | --- |
| proposes | LLM harness (extraction, explanation) | decision model (typed, calibrated) |
| canonical data | fact graph with warrants and tiers | dated corpus nodes with ranks |
| decides | small deterministic deontic kernel | verifier + engine |
| never | "LLM-only verdict generation" | a number or citation the model wrote itself |

What's missing on the tax side is the step between *facts* and *verdict*:
which obligations, prohibitions and permissions apply to this taxpayer in
this context, under which exceptions, and why. That's exactly the kernel the
deontic repo specifies. The fit is one change of role:

```text
ADQ:     evidence cosheaf ─▶ affect/context diagnosis ─▶ deontic sheaf ─▶ gluing | obstruction
tax-lm:  state + corpus   ─▶ decision model (context atoms, p) ─▶ deontic kernel ─▶ verdict + trace | obstruction
```

**The Jev-style model becomes the evidence layer and stops being the verdict
layer.** It answers typed questions about *facts* ("did Maya live with the
taxpayer > ½ the year?", "is this payment compensation?") with calibrated
p. The kernel lifts those into deontic verdicts using warrants from the
corpus. This also removes Jev's main weakness, opacity: every verdict is a
kernel derivation with a minimal warrant path.

## 1. Type mapping

The ADQ normalised atom

```text
deontic_obligation(Affect, Object, Subject, Operator, Modality, Warrant)
```

generalises to a domain-neutral norm by renaming its first slot to what the
norm regulates:

```text
norm(Act, Object, Subject, Operator, Modality, Warrant)
```

| ADQ slot | Tax reading | Example |
| --- | --- | --- |
| `Affect` → `Act` | a regulated act: file, pay, claim, elect, report, withhold, keep records | `claim` |
| `Object` | what the act is about | `child_tax_credit`, `form_1040`, `section_179_expense` |
| `Subject` | taxpayer class the norm binds | `individual`, `married_individual`, `employer`, `exempt_org`, `irs_employee` |
| `Operator` | `o f r d p n`, unchanged | `p` |
| `Modality` | `unconditional`, `conditional(C)`, `comparative(C)`, `scalar(C)` | `conditional(qualifying_child(X))` |
| `Warrant` | a corpus node ID (+ span) at `as_of` | `irc/24/a@2025-07-04` |

### 1.1 Operators

The six ADQ operators cover tax law without extension; the five classical
aḥkām map onto the same six:

| op | fiqh | tax | typical statutory text |
| --- | --- | --- | --- |
| `o` | wājib | obligation | "shall make a return", "shall pay", "shall deduct and withhold" |
| `f` | ḥarām | prohibition | "no deduction shall be allowed", "may not", penalty-backed acts (§6700 promotion) |
| `p` | mubāḥ | permission / entitlement | "there shall be allowed as a credit", "may elect" |
| `r` | mandūb | recommended | safe harbours and "should" in IRS guidance: rank ≥ 3 only |
| `d` | makrūh | discouraged | listed-transaction / reportable-transaction exposure; rank ≥ 3 only |
| `n` | outside taklīf | outside the norm's reach | exempt entity, non-resident for that rule, not a taxable event |

Invariant **D1**: `r` and `d` never carry a rank-1/2 warrant alone. The Code
and regulations speak in `o f p`; recommendation and discouragement are IRS
posture, which is exactly the authority note A1 in `SPEC.md` made a type.

### 1.2 Subject carries the authority column

`SPEC.md` §2 has a "Binding on" column. In deontic terms that column **is**
the subject: IRM directives are `o`/`r` norms with `Subject = irs_employee`;
a PLR binds `Subject = requesting_taxpayer(id)`. `subject_matches/2` from the
v38 bridge then makes "the IRM does not bind taxpayers" a kernel fact, not a
prompt instruction. A verdict for a taxpayer derived from an
`irs_employee` norm is a `subject_collision` obstruction.

### 1.3 Warrant tiers

| ADQ tier | tax sources | may support a taxpayer verdict? |
| --- | --- | --- |
| 1 | IRC; final and temporary regs | yes |
| 2 | Rev. Rul., Rev. Proc., Notices; court opinions | yes, with rank recorded |
| 3 | IRM; forms, instructions, pubs | context/annotation; verdicts only for `Subject = irs_employee` |
| 4 (quarantine) | PLR/TAM/CCA for other taxpayers; proposed regs; teacher labels; **any model output** | never directly (`tier_quarantine`) |

Model outputs are tier 4 by construction. They enter the kernel only as
context atoms (§2), never as norms.

### 1.4 Modality and where numbers go

A modality's condition is a boolean formula over **context atoms**, and each
atom has a typed source:

```text
atom  := given(fact)            -- stated by the caller, p ∈ {0, 1}
       | computed(engine, expr) -- engine/date code, p ∈ {0, 1}   (thresholds, phase-outs, deadlines)
       | perceived(q, p)        -- a decision-model answer, calibrated p ∈ [0, 1]
```

`scalar(C)` and `comparative(C)` conditions (gross income ≥ filing
threshold, "greater of", AGI phase-outs) are always `computed`. This is
Jev's "numbers to code" rule, enforced by the type of the atom.

### 1.5 Control status, capacity, and defeaters

The ADQ capacity guard (a raw involuntary affect is not liability; a
cultivated, endorsed or action-guiding one is) has a close tax analogue in
the mental-state gradations for penalties:

| ADQ control status | tax analogue | effect |
| --- | --- | --- |
| `raw_occurrence`, `involuntary_startle` | reasonable cause and good faith (§6664(c), §6651(a) "unless … due to reasonable cause") | defeats the penalty norm |
| `cultivated`, `endorsed` | negligence / disregard (§6662(b)(1), (c)) | penalty norm applies |
| `action_guiding`, `dominant` | willfulness, fraud (§6663; FBAR willful penalty) | higher-tier penalty norm applies |

These are `defeater` records in the bundle schema (`target`, `defeats`,
`condition`, `warrant_id`). Statutory exceptions ("except as provided in
subsection (b)", "notwithstanding …") are defeaters too, with priority
from the warrant (lex specialis, then later-in-time, then rank).

### 1.6 Anti-pairs are mutually exclusive tax positions

`deontic_anti_pair/4` in ADQ is a displacement constraint, not a
contradiction. In tax that shape is the family of **exclusive elections and
no-double-benefit rules**:

- itemise ⊕ standard deduction (§63(b), (e));
- AOTC ⊕ lifetime learning credit for the same student and year (§25A(c)(2)(A));
- the same expense can't fund two benefits (e.g. §280C, the §25A/§529
  coordination rules);
- filing status is a choice of exactly one.

`anti_pair_failure` becomes "two exclusive positions both claimed", reported
with both warrants.

### 1.7 Paths are the tax timeline

The v38 path lift (contexts refined by action-prefix extension; "a
permission can become forbidden later without contradiction") is how tax
elections behave over time:

- separate → joint is permitted for three years after the due date
  (§6013(b)); joint → separate is not, after the due date
  (Treas. Reg. §1.6013-1(a)(1)). Same act, same object, different operator
  on a longer prefix.
- accounting-method and many entity elections, once made, flip later changes
  from `p` to `f` absent consent.

So `ctx(World, Agent, Prefix)` with `Prefix` a list of dated filings and
elections, and `as_of` selecting the corpus snapshot, is the tax context.
Restriction along a prefix and restriction along snapshots (law changes)
are the two refinement directions of the context poset.

## 2. Pipeline

```text
request(state, as_of, questions)
  │
  ├─ retrieve S_as_of ─▶ norms N (accepted bundle records whose warrants ∈ S_as_of)
  │
  ├─ atoms needed by N's modalities ─▶ { given | computed(engine) | perceived(decision model) }
  │                                       perceived atoms = Jev-style questions, one batched pass
  │
  ├─ kernel(N, atoms, prefix)                      -- the deontic repo's kernel, unchanged
  │     local sections per context ─▶ gluing | obstruction(category, sections)
  │
  └─ response
        verdicts:     norm → operator, P(applies), minimal warrant path (node IDs + spans)
        obstructions: typed, with the sections that failed to glue
        abstain:      verdicts whose P interval straddles τ
```

### 2.1 Probabilities through a deterministic kernel

The kernel stays boolean. With perceived atoms carrying marginals `p_i`, the
probability that a verdict holds is the probability of its derivation
formula φ over those atoms:

- if the atoms are independent: `P(φ)` by weighted model counting
  (exact enumeration for ≤ ~16 uncertain atoms; knowledge compilation, as in
  ProbLog, beyond that);
- if they aren't known to be independent (the default: they were read from
  the same state), report the **Fréchet interval**, e.g.
  `P(a ∧ b) ∈ [max(0, p_a + p_b − 1), min(p_a, p_b)]`, lifted through φ.

Abstain when the interval straddles `τ`. That turns the kernel into the
place where calibration pays off: a well-calibrated evidence layer gives
honest verdict intervals without the kernel ever becoming probabilistic
logic.

## 3. What each side gains

**tax-lm gets**

- A verdict layer with derivations, so its answers are explanations by
  construction instead of floats.
- Labels beyond the engine. The engine covers amounts and 1040 eligibility;
  the kernel also labels obligations the engine doesn't model (filing
  requirement §6012, recordkeeping §6001, information returns, penalty
  exposure). Kernel(facts) is a second oracle for §6.1 of `SPEC.md`.
- A differential test for the extracted rules: on synthetic fact patterns,
  kernel verdicts over the bundle must agree with the engine wherever both
  answer. A disagreement is an extraction bug, found before any training.
- Typed failure: `obstruction` records instead of a low-confidence guess.

**the deontic repo gets**

- A second domain that exercises the kernel on a large, versioned,
  authoritative, public-domain corpus with an independent oracle to test
  against, which fiqh doesn't have.
- The decision model as the calibrated evidence cosheaf the sheaf doc
  describes, and the dated corpus as a concrete warrant store.

## 4. Changes needed

In `usul-al-fiqh-deontic-circuits` (proposal only; nothing pushed there):

1. Split the bundle schema into a domain-neutral core and vocabularies:
   `norm` (the six-slot atom), `warrant`, `defeater`, `anti_pair`,
   `refinement_edge`, `obstruction` in the core; `affect_root` and the
   `hukm` enum become the fiqh vocabulary, and a tax vocabulary adds
   `act`, `tax_object`, `tax_subject`.
2. Keep one kernel. tax-lm calls it as an external checker, the same way
   `prover_loop/verify.py` calls Lean: canonical JSON in, verdicts/trace/
   obstructions JSON out, a missing or malformed report is a rejection. The
   kernel is Prolog today (`v38_contextual_affect_deontics.pl`) and a Gleam
   checker later; the JSON interface is the contract either way.
3. Add `computed/2` and `perceived/2` atom sources to `ctx/3` conditions,
   and a probability-interval pass over derivations (§2.1), outside the
   boolean core.

In `llm-kernel-lab`:

1. `tax_loop/norms.py`: LLM-proposed norm extraction from corpus nodes into
   candidate bundle records (status `candidate`), never auto-accepted;
   review queue per the deontic repo's M5.
2. `tax_loop/kernel.py`: subprocess adapter to the deontic kernel.
3. Response contract (`SPEC.md` §4.1) gains `verdicts` and `obstructions`;
   `accept` gains: every verdict warrant ∈ `S_as_of`, tier ≤ 2 for
   taxpayer verdicts (tier ≤ 3 for `irs_employee`), D1.

## 5. First slice

Test-first, over one small family where every piece exists:

- **Norms**: §6012(a) filing obligation (`o`, conditional on computed gross
  income vs. threshold), §24 child tax credit (`p`, conditional on perceived
  qualifying-child atoms from §152(c)), §63 itemise ⊕ standard (anti-pair),
  §6651(a) failure-to-file penalty with the reasonable-cause defeater.
- **Contexts**: two snapshots (TY2024, TY2025, straddling Pub. L. 119-21)
  × two prefixes (before and after filing a joint return).
- **Tests** (one per obstruction category that can arise here):
  non-contradictory separation across snapshots, `subject_collision` (IRM
  norm applied to a taxpayer), `modality_unsatisfied`, `tier_quarantine`
  (a PLR or a model output used as a warrant), `anti_pair_failure`
  (both itemised and standard), the §6013 prefix asymmetry, and the
  Fréchet interval straddling τ ⇒ abstain.
- **Differential**: kernel vs. engine on 10⁴ sampled households for the
  CTC and filing-requirement verdicts; zero disagreements is the exit
  criterion.

## 6. Open questions

1. Obligations of *other* parties (employer withholding, information
   returns) put multiple agents in one world. Do the v38 predicates need
   a joint-agent path, or is one world per agent with shared facts enough?
2. `r` and `d` in tax are thin. Keep them for IRS posture, or narrow the
   tax vocabulary to `o f p n` and leave `r d` fiqh-only?
3. Priority among defeaters: lex specialis vs. later-in-time vs. rank. Is
   that a kernel rule or bundle data (as ADQ keeps corpus facts out of
   code)? Leaning data, with the kernel only enforcing that a priority
   exists when two defeaters collide.
