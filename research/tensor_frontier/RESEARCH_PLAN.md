# Research plan: algebraic tensor empirical mathematics

## 1. Objective

Use MAX/Mojo to expand the empirically observable frontier of tensor
mathematics while keeping theorem authority outside the accelerator.

The program optimizes *search and measurement*, not truth.  Every experiment
must identify:

1. the mathematical object being sampled;
2. the algebraic invariant being measured;
3. the kernel family that dominates runtime;
4. the numerical or combinatorial observation produced;
5. the exact witness format, when one exists;
6. the independent acceptance predicate.

## 2. Authority ladder

| Grade | Artifact | Meaning |
| --- | --- | --- |
| E0 | timing / profile | engineering evidence only |
| E1 | numerical observation | candidate phenomenon |
| E2 | replicated observation | stable across seeds/devices/implementations |
| E3 | exact reconstructed witness | finite algebraic object checks exactly |
| E4 | independently verified certificate | checker accepts the stated predicate |
| E5 | Lean-checked theorem | theorem-grade inside this repository's proof boundary |

Grades never auto-promote.  In particular E1/E2 data cannot be described as a
proof, counterexample, or theorem.

## 3. Initial frontier programs

### TF-1 — Moment polytopes and tensor scaling

Question: can larger structured families expose new marginal-spectrum facets,
non-maximality phenomena, or exceptional orbit-closure behavior?

Dominant kernels:
- mode contractions;
- mode Gram matrices;
- batched matrix products;
- repeated algebraic scaling updates;
- rank/minor tests.

Algebraic contract:
- marginals are represented by polynomial/rational matrix data;
- no angle coordinates;
- logarithmic parameterizations are implementation diagnostics only;
- candidate facet inequalities must be reconstructed over exact rationals or
  an explicitly declared algebraic extension before promotion.

First deliverable: a seeded census producing candidate rational facet normals
and exact residuals.

### TF-2 — Spiked tensor phase transitions

Question: how do finite-size recovery/detection transitions move with order,
dimension, sparsity, signal model, and algorithm?

Dominant kernels:
- high-order tensor-vector contraction;
- batched power/AMP-like updates;
- hypergraph-count statistics;
- large seed sweeps.

Algebraic contract:
- the planted tensor is polynomial data;
- noise is specified by its finite generator and moment/covariance contract,
  not by an exponential density formula;
- success is measured by squared overlap / Gram invariants, not inverse
  trigonometric angles.

First deliverable: reproducible finite-size transition tables with seed,
generator, and exact planted-instance metadata.

### TF-3 — Rank growth in tensor dynamics

Question: when does high-dimensional evolution remain compressible, and which
parameters control rank growth?

Targets include tensor-train kinetic models and tensor-network probability
dynamics.

Dominant kernels:
- contractions and permutations;
- Gram construction;
- QR/elimination;
- truncation candidates;
- fused local update + contraction.

Algebraic contract:
- rank is certified by minors/elimination when exact certification is needed;
- approximation error uses squared residuals;
- square roots are unnecessary at the contract boundary;
- SVD-derived quantities may guide search but cannot be the sole certificate.

First deliverable: rank-versus-time/parameter datasets plus exact rank checks on
selected snapshots.

### TF-4 — Secant varieties, identifiability, and border-rank reconnaissance

Question: can massive batched tangent/Jacobian tests find exceptional loci or
new structured low-rank degenerations?

Dominant kernels:
- Jacobian assembly;
- tangent-space products;
- batched rank tests;
- candidate decomposition evaluation.

Algebraic contract:
- Terracini-style rank observations are reconstructed exactly before claim
  promotion;
- near decompositions are not border-rank certificates by themselves.

First deliverable: an anomaly corpus containing the tensor, parameterization,
observed deficiency, exact reconstructed matrix, and checker verdict.

### TF-5 — PEPS / structured tensor-network contractions

Question: which contraction, layout, and environment-update structures are the
largest MAX/Mojo wins, and do higher reachable bond dimensions change empirical
convergence conclusions?

Dominant kernels:
- structured contractions;
- layout transforms;
- Gram/QR steps;
- batched small/medium factorizations.

Algebraic contract:
- phase factors are represented as norm-one algebra elements or rotor pairs;
- angles and complex exponential notation are presentation only;
- squared residuals and polynomial identities are preferred observables.

First deliverable: a contraction corpus with shape/layout metadata and
cross-implementation numerical replication.

## 4. Kernel program

The common kernel vocabulary is deliberately smaller than the application
vocabulary:

```text
contract
permute_contract
mode_gram
batched_rank
rotor_apply
gram_residual
tt_local_update
structured_reduce
```

The optimization order is:

1. establish a scalar/exact reference predicate;
2. establish a conventional CPU/GPU baseline;
3. implement the Mojo kernel;
4. prove differential agreement on a bounded corpus;
5. profile layout, memory traffic, launch count, and arithmetic intensity;
6. scale the scientific experiment only after agreement.

Do not optimize a kernel whose mathematical predicate is still ambiguous.

## 5. Transcendental extraction pass

Every imported paper/problem receives a vocabulary audit.  Occurrences of
angle, sin/cos/tan, inverse trig, complex phase, exp/log, entropy, Fourier
phase, matrix exponential, and square-root normalization are recorded.

Each occurrence receives one of three statuses:

- **A — exact algebraization:** same quantity/predicate in algebraic form;
- **S — surrogate:** useful algebraic statistic but not mathematically
  equivalent;
- **D — diagnostic:** analytic quantity retained only as non-authoritative
  instrumentation.

A surrogate may never inherit the theorem statement of the source quantity.

## 6. LLM representation experiment

For selected problems generate paired prompts:

- T-form: conventional triangle/trigonometric vocabulary;
- A-form: Gram, bilinear, rotor, projective-ratio, and polynomial vocabulary;
- C-form: contraction/index vocabulary when natural.

Hold statement semantics, model, decoding parameters, and verifier fixed.

Measure pass@k, Lean acceptance, transcendental leakage, completion length,
verification cost, and diversity of exact witnesses.  This tests the hypothesis
that triangle language is a representational habit rather than a mathematical
necessity.

## 7. Reproducibility record

Each run must record at minimum:

```text
experiment_id
problem_contract_hash
kernel_revision
MAX/Mojo revision
device
seed / generator
shape and dtype
algebraic encoding
transcendental classification
raw observation
reconstruction status
checker/version
authority grade
```

## 8. Milestones

M0 — scaffold and firewall:
- exact rational rotor/Gram/spread primitives;
- automated source audit;
- experiment manifest;
- paired-representation protocol.

M1 — kernel baseline:
- contraction, mode-Gram, batched-rank corpus;
- CPU reference and Mojo implementations;
- differential tests and benchmark harness.

M2 — first scientific probe:
- TF-1 moment-polytope scaling census;
- candidate reconstruction pipeline.

M3 — independent replication:
- second implementation/oracle;
- device/seed replication;
- evidence-grade promotion machinery.

M4 — LLM study:
- paired T/A/C corpus;
- verified pass@k analysis;
- publishable negative results retained.

## 9. Explicit non-goals

This track does not make floating-point arithmetic authoritative, does not
reinterpret an analytic theorem merely because an algebraic surrogate correlates
with it, does not treat an LLM completion as evidence without execution, and
does not require every useful numerical algorithm internally to be symbolic.
The firewall governs mathematical contracts and claim promotion.
