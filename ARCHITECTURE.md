# Architecture

This repository follows **Estate Repository Template v2**, whose canonical
source is `larsbx/estate-governance`.

The machine-readable authority map is [`ESTATE.toml`](ESTATE.toml). The
repository-specific migration map remains
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

## Governance pin

`ESTATE.toml` pins the canonical v2 audit by governance commit and SHA-256.
CI downloads that exact audit from `larsbx/estate-governance`, verifies the
digest, and runs it against this checkout. No repository-private copy of the
generic estate audit is authoritative.

## Transitional layout

The repository remains `layout.status = "transitional"`. Existing paths map
to v2 authority planes through `ESTATE.toml`; this migration does not mass-move
source.

No path change in the estate migration changes proof status, model authority,
benchmark meaning, or the semantics of the Lean verifier.
