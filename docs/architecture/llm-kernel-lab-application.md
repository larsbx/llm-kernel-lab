# Estate template application: llm-kernel-lab

Status: second adopter of Estate Repository Template v1.

## Why this repository is the second adopter

The first adopter, `finite-mandelbrot-research`, has a Mojo-centered executable
acceptance boundary. This repository exercises a materially different authority
shape: Lean is canonical for proof acceptance, while Python, MAX/model inference,
and Mojo occupy supporting roles.

If the same manifest and audit can describe both repositories without changing
their authority, the template is demonstrating cross-domain and cross-language
generality rather than merely encoding one repository's layout.

## Authority boundary

The proof path is:

```text
problem statement
    │
    ▼
candidate generator / model
    │    non-authoritative
    ▼
prover_loop orchestration
    │
    ▼
pinned Lean + Mathlib verifier
    │
    ├── rejects ──▶ evidence only
    │
    └── accepts ──▶ verified proof record
```

Only the final Lean boundary decides whether a candidate is accepted as a proof.
A model completion, Python verdict, benchmark score, tensor experiment, or Mojo
computation cannot substitute for that boundary.

## Current-to-target map

| Authority plane | Current paths | Target | Meaning |
| --- | --- | --- | --- |
| policy | `research/tensor_frontier/estate-adoption.toml` | `policy/` | authority routing and governance |
| kernel | `lean/` | `kernel/` | canonical proof acceptance |
| reference | `tensor_lab/` | `reference/` | exact executable reference semantics |
| generation | `prover_loop/` | `generation/` | candidate generation and verification orchestration |
| oracles | `mojo/` | `oracles/` | non-authoritative high-performance research computation |
| experiments | `research/tensor_frontier/*.md`, `benchmarks/` | `experiments/` | research programmes and benchmark experiments |
| conformance | `tests/` | `conformance/` | regression and boundary tests |
| evidence | `results/`, `problems/` | `evidence/` | benchmark inputs and generated evidence |
| tooling | `tools/`, `gpu/` | `tools/` | audits, preparation, training and run orchestration |
| docs | `README.md` | `docs/` | exposition |

The map is transitional. No mass move is part of the adoption PR.

## Language roles

### Lean

- authority: canonical;
- roles: kernel, proof acceptance;
- acceptance authority: yes.

The pinned Lean/Mathlib environment is part of the accepted proof boundary. A
missing report, verifier failure, prohibited axiom, timeout, or malformed candidate
fails closed.

### Python

- authority: supporting;
- roles: orchestration, reference, audit, training;
- acceptance authority: no.

Python may invoke the Lean verifier and faithfully record its result. That does not
make the Python caller the proof authority.

### Mojo

- authority: supporting;
- roles: oracle, experiment, benchmark;
- acceptance authority: no.

Mojo/MAX acceleration may produce exact empirical evidence or accelerate discovery.
It cannot promote a theorem candidate.

### Bash

- authority: supporting;
- role: GPU/run orchestration;
- acceptance authority: no.

## Relationship to tensor-frontier routing

`research/tensor_frontier/estate-adoption.toml` already distinguishes:

- discovery ownership;
- reusable exact-kernel ownership;
- independent-oracle ownership;
- semantic-contract ownership;
- domain claim ownership; and
- proof authority.

That policy remains in force. Estate Template v1 lifts the same discipline to the
whole repository instead of replacing the tensor-specific routing contract.

## Migration sequence

1. Declare and audit the existing authority boundaries.
2. Keep the present tree stable while the second-adopter contract is reviewed.
3. Extract the generic template and `audit_estate_layout.py` into shared estate
   tooling now that two repositories exercise it.
4. Separate candidate-generation code from the Lean verification adapter.
5. Move exact tensor reference code into the reference plane.
6. Move Mojo tensor discovery kernels into explicit oracle/experiment namespaces.
7. Separate immutable benchmark inputs from generated evidence.
8. Remove transitional path mappings only after consumers and CI use the canonical
   layout.

## Non-goals

This adoption does not:

- change which proofs Lean accepts;
- make model output authoritative;
- make MAX or Mojo a proof checker;
- promote tensor observations into theorem claims;
- change the training/evaluation split;
- change tensor-frontier cross-repository claim ownership; or
- require a bulk directory reshuffle.

## Second-adopter consequence

With `finite-mandelbrot-research` and `llm-kernel-lab` both adopting v1, the
template's stated promotion condition is met: the next architectural step is to
extract the generic contract and audit into shared estate governance/tooling and
have repositories pin that implementation rather than maintain private copies.
