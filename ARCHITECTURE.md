# Architecture

This repository follows **Estate Repository Template v1**.

The machine-readable authority map is [`estate.toml`](estate.toml). The reusable
contract is [`docs/architecture/estate-repository-template-v1.md`](docs/architecture/estate-repository-template-v1.md),
and the repository-specific migration map is
[`docs/architecture/llm-kernel-lab-application.md`](docs/architecture/llm-kernel-lab-application.md).

## Authority summary

Authority is assigned **before** domain and language.

- **Lean is the sole proof-acceptance authority.** A theorem candidate is accepted
  only through the pinned Lean kernel boundary.
- **Python does not prove theorems.** It orchestrates generation, verification,
  data preparation, training, audits, and exact reference calculations.
- **Mojo does not prove theorems.** Tensor and MAX/Mojo work supplies
  non-authoritative computation, experiments, benchmarks, and candidate evidence.
- **Model output is candidate material.** MAX/model inference never promotes a
  completion to a theorem.
- **Research evidence is not proof authority.** Exact empirical results may motivate
  or falsify candidate formulations, but theorem status remains with the canonical
  checker.

The existing `research/tensor_frontier/estate-adoption.toml` remains the
cross-repository routing policy for tensor-frontier work. It does not delegate
domain-claim acceptance.

## Transitional layout

The current tree is intentionally retained. `estate.toml` maps existing paths to
their target authority planes so architecture can be audited before files are moved.

No file move in this adoption changes proof status, model authority, benchmark
meaning, or the semantics of the Lean verifier.
