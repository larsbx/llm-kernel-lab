# Estate v2 application: llm-kernel-lab

Status: transitional v2 consumer.

## Why this repository matters

`llm-kernel-lab` exercises a different authority shape from the Mojo-centered
finite-mathematics repositories: Lean is canonical for proof acceptance, while
Python, MAX/model inference, Mojo, benchmarks, and tensor experiments occupy
supporting or non-authoritative roles.

The v2 migration preserves that distinction and replaces the superseded local v1
template/audit copies with the canonical `larsbx/estate-governance` contract.

## Authority boundary

The proof path is:

```text
problem statement
    |
    v
candidate generator / model
    |    non-authoritative
    v
prover_loop orchestration
    |
    v
pinned Lean + Mathlib verifier
    |
    +-- rejects --> evidence only
    |
    +-- accepts --> verified proof record
```

Only the final Lean boundary decides whether a candidate is accepted as a proof.
A model completion, Python verdict, benchmark score, tensor experiment, or Mojo
computation cannot substitute for that boundary.

## Current-to-target map

| Authority plane | Current paths | Target | Meaning |
| --- | --- | --- | --- |
| policy | `ESTATE.toml`, `research/tensor_frontier/estate-adoption.toml` | `policy/` | repository and tensor-routing governance |
| kernel | `lean/` | `kernel/` | canonical proof acceptance |
| reference | `tensor_lab/` | `reference/` | exact executable reference semantics |
| generation | `prover_loop/` | `generation/` | candidate generation and verification orchestration |
| oracles | `mojo/` | `oracles/` | non-authoritative high-performance research computation |
| experiments | `research/tensor_frontier/*.md`, `benchmarks/`, `experiments/` | `experiments/` | research programmes and benchmark experiments |
| conformance | `tests/` | `conformance/` | regression and boundary tests |
| evidence | `results/`, `problems/` | `evidence/` | benchmark inputs and generated evidence |
| tooling | `tools/`, `gpu/`, `flake.nix` | `tools/` | preparation, training, run orchestration, and provisioning |
| docs | `README.md`, `ARCHITECTURE.md`, `docs/` | `docs/` | exposition |

The map is transitional. No mass move is part of the v2 adoption.

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

`research/tensor_frontier/estate-adoption.toml` continues to distinguish:

- discovery ownership;
- reusable exact-kernel ownership;
- independent-oracle ownership;
- semantic-contract ownership;
- domain claim ownership; and
- proof authority.

That tensor-specific policy composes with `ESTATE.toml`; it does not replace the
repository-wide authority manifest.

## v1 retirement

The v2 migration removes the repository-private generic v1 audit and contract.
CI instead retrieves `kernel/audit_estate_layout.py` from the exact
`estate-governance` revision declared in `ESTATE.toml` and verifies its
SHA-256 before execution.

The old lowercase `estate.toml` is superseded by `ESTATE.toml`.

## Migration sequence

1. Adopt v2 authority and provenance fields without changing proof semantics.
2. Keep the present tree stable while the v2 gate proves the mapping.
3. Separate candidate generation from the Lean verification adapter.
4. Move exact tensor reference code into the reference plane.
5. Move Mojo tensor discovery kernels into explicit oracle/experiment namespaces.
6. Separate immutable benchmark inputs from generated evidence.
7. Remove transitional path mappings only after path migrations land atomically.

## Non-goals

This migration does not:

- change which proofs Lean accepts;
- make model output authoritative;
- make MAX or Mojo a proof checker;
- promote tensor observations into theorem claims;
- change the training/evaluation split;
- change tensor-frontier cross-repository claim ownership; or
- require a bulk directory reshuffle.
