# Triangle-bias protocol for prover models

## Hypothesis

LLMs are heavily exposed to geometric explanations phrased through triangles,
angles, unit circles, sine/cosine identities, and complex phases.  For tensor
and invariant problems those coordinates may be unnecessary and can introduce
extra analytic structure.

This protocol tests that claim instead of assuming it.

## Paired representations

Each problem is rendered in up to three semantics-preserving forms:

**T-form — conventional trig surface**

Uses the source literature's angle/trig/phase terminology.  This form is an
experimental prompt only and is exempt from the authoritative source-code
firewall.

**A-form — algebraic invariant surface**

Uses quadratic/bilinear forms, Gram determinants, conic rotors, projective
ratios, polynomial identities, and explicit nonzero-denominator guards.

**C-form — contraction surface**

Uses tensor indices, contractions, matrix/tensor products, and rank predicates
without geometric narration when that is natural.

The proposition checked by Lean must be the same proposition after translation,
or the pair is invalid.

## Controls

Hold fixed:

- model and model revision;
- prompt preamble except for representation;
- theorem assumptions and conclusion;
- decoding temperature/top-p;
- token budget;
- number of samples `k`;
- Lean/Mathlib revision;
- timeout and hardware class.

Randomize presentation order and retain failed generations.

## Measurements

Primary:
- verified pass@k;
- fraction of completions accepted by Lean;
- forbidden-transcendental leakage in A/C forms.

Secondary:
- completion token count;
- verification wall time;
- timeout rate;
- distinct verified proof count;
- generated helper-lemma count;
- exact-expression size.

## Interpretation

A higher verified rate for A-form would be evidence that algebraic
representation is useful to the prover.  A higher T-form rate would be evidence
that pretrained geometric priors are useful.  Either result is publishable.

No conclusion is drawn from prose plausibility.  Only verifier outcomes and
predeclared measurable artifacts enter the comparison.

## First corpus

Start with small exact identities that have both natural trig and algebraic
forms:

1. rotor closure and composition;
2. Gram/spread scale invariance;
3. orthogonality preservation by a rotor;
4. tensor-mode Gram covariance under basis change;
5. rank invariance under invertible mode transformations.

Then move to frontier-derived lemmas only after translation equivalence is
reviewed.
