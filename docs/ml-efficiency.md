# ML efficiency: exact speculative correction

Run the reference checks:

```sh
python -m pytest -q tests/test_speculative_sampling.py tests/test_speculative_acceptance.py
python tools/check_speculative_sampling.py --lean /path/to/lean-4.15.0/bin/lean
```

After committing, add `--require-clean --out runs/speculative/acceptance.json`.
The gate compares every acceptance input directly with its Git blob, compiles
fresh Lean modules, checks their types and axioms, compares Lean/Python masses,
and runs rejection controls. It needs Lean **4.15.0** and Python 3.11 or later;
this proof lane needs neither Mathlib nor a model/GPU download. The Python tests
add only pytest. `pixi run sampling-test` and `pixi run sampling-acceptance`
provide the equivalent repository tasks, with pinned Lean on `PATH`.

## Implemented increment

`sampling_lab.speculative` provides the exact acceptance/residual correction
used in speculative decoding. It is a one-token reference primitive with an
explicit finite-probability contract. It does not yet batch draft tokens through
MAX, reuse model caches, or establish a serving speedup.

```python
from secrets import randbelow
from sampling_lab import prepare

# Vocabulary positions must name the same tokens in both distributions.
plan = prepare(target_weights=[1, 4, 2], draft_weights=[3, 1, 1])
token = plan.draw(randbelow)

# Standalone illustration of an independently sampled draft proposal:
draft_plan = prepare(target_weights=[3, 1, 1], draft_weights=[3, 1, 1])
proposed_token = draft_plan.draw(randbelow)
corrected = plan.correct(proposed_token, randbelow)
```

The first call draws from the normalized target distribution by proposing and
correcting internally. In the second call, the target-distribution guarantee
requires the input token to have been sampled from the stated draft distribution
at the same conditioning prefix. A model integration must establish that
relationship for the actual proposal.

Inputs are nonempty, equally sized sequences of nonnegative Python integers;
each distribution must have positive total. The constructor copies them into an
immutable plan. Booleans, floats, negative weights, zero totals, invalid token
indices and out-of-range random tickets are rejected. No floating normalization,
rounding or machine-integer arithmetic is used. `randbelow(n)` must supply
independent uniform integers in `[0,n)`; the implementation checks its output
range but cannot establish uniformity or independence.

`resolve(ticket, residual_ticket=None)` exposes the pure finite-ticket mapping
for exact replay. Accepted proposals use no residual draw; rejected proposals
require one. Unused residual tickets are refused. Zero-probability draft tokens
cannot be supplied to `correct`.

Preparation requires linear vocabulary storage/work. Prepared draws use binary
search in cumulative integer masses; correction of an external draft token
only searches residual intervals after rejection. These operation counts omit
integer bit complexity and do not constitute a measured acceleration result.
For standalone categorical sampling, direct target sampling may be cheaper.

## Exact law

Let `p_i`, `q_i` be target/draft integer weights, with positive totals `P`, `Q`.
Construct:

```text
D   = P * Q
t_i = p_i * Q
d_i = q_i * P
a_i = min(t_i, d_i)
r_i = t_i - a_i
b_i = d_i - a_i
R   = sum(r_i) = sum(b_i)
```

A uniform ticket in `[0,D)` selects a draft token from intervals of width `d_i`.
The first `a_i` positions of its interval accept the token. A rejection uses an
independent uniform ticket in `[0,R)` and residual intervals of width `r_i`.
For `R > 0`, token `i` therefore has probability:

$$
\frac{a_i}{D}+\frac{R}{D}\frac{r_i}{R}
=\frac{a_i+r_i}{D}=\frac{p_i}{P}.
$$

When `R = 0`, every proposal accepts, and no residual distribution is evaluated.
The Lean theorem avoids division: for a finite uniform ticket space `S`, it
proves positivity, total output mass `S`, and
`outputMass_i * P = S * p_i`. When rejection is possible, `S = D * R`; otherwise
`S = D`. The product-space interpretation includes an unused second coordinate
on accepted paths. The runtime draws that coordinate only on rejection.

The algorithm is established prior work, not a novelty claim: see
[Leviathan, Kalman and Matias, ICML 2023, section 2.3](https://proceedings.mlr.press/v202/leviathan23a.html).

## Evidence and boundaries

| Layer | Acceptance | Scope |
|---|---|---|
| Lean model | `lean/SpeculativeSampling.lean` | Universal finite integer mass laws; positive totals; unequal denominators and zero token weights permitted. |
| Public types | `lean/SpeculativeSamplingContract.lean` | Six independently stated obligations reject a weakened exported theorem. |
| Axioms | Required reports from twelve model and six contract declarations | Only the repository's standard Lean axioms are allowed; missing/duplicate reports and extra assumptions fail. |
| Python reference | `tests/test_speculative_sampling.py` | Exact rational enumeration of 744 small distribution pairs for both ticket resolution and external-token correction, plus malformed inputs and large integers. |
| Cross-language replay | `tools/check_speculative_sampling.py` | 747 mass-table comparisons, including three additional unequal-support/large-integer cases. This is finite conformance, not universal Python refinement. |
| Rejection controls | Required by the proof gate | False proposition, `sorry`, compiler-trusted decision, missing report, and actual theorem weakening. |
| Hosted evidence | `Exact speculative sampling` workflow artifact | Exact checked-out revision, acceptance-source hashes, proof reports, conformance and explicit unverified boundaries. |

The Python interval decoder, RNG, native compiler, floating-point model logits,
model/proposal provenance and multi-token decoder are outside the current Lean
theorem. In particular, integer weights define the exact distribution being
preserved. Converting model probabilities to those weights needs a separate
error/representation contract. Kernel-reduced proofs and compiled conformance
execution are recorded as different kinds of evidence.

## Next acceptance milestone

Connect a draft and target model at an identical tokenizer/prefix, record the
actual post-processing distributions, apply correction, and verify rollback and
cache behavior after rejection. Compare complete decoding latency, target-model
evaluations, memory and output law against ordinary decoding at fixed model,
prompt set, hardware and sampling settings. Speedup must be measured end to end.

## Other efficiency targets

| Target | Main cost | Candidate improvement | Preservation obligation |
|---|---|---|---|
| Long-context attention | Quadratic pair interactions; intermediate memory traffic | Fused/tiled exact attention | Attention and masking semantics plus numerical error; no assumed arithmetic complexity reduction. |
| Repeated prefix computation | Re-evaluating unchanged causal states | KV caching | Same model, prefix, position and attention state. |
| Exact dense Gaussian processes | Cubic factorization; quadratic storage | Structured or sparse methods | Exact structure conditions or explicit posterior approximation. |
| Kernel SVM / nearest neighbors | Sample-count scaling / per-query scan | Kernel approximation or indexing | Decision-function or retrieval-quality contract. |
| Hyperparameter search | Combinatorial model evaluations | Budgeted search / early termination | Validation protocol and budget; no exhaustive-optimum guarantee. |
| Diffusion sampling | Repeated network evaluations | Better solvers / distillation | Matched quality and explicit numerical/model changes. |

Primary references: [FlashAttention](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract.html),
[Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/chapters/RW2.pdf),
[SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html),
[nearest neighbors](https://scikit-learn.org/stable/modules/neighbors.html),
[random search](https://www.jmlr.org/papers/v13/bergstra12a.html),
[DPM-Solver](https://proceedings.neurips.cc/paper_files/paper/2022/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html).
