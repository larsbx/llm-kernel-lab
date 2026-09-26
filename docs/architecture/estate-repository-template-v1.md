# Estate repository template v1

Status: reusable estate architecture contract.

## Purpose

The template gives every repository in the estate the same answer to four questions:

1. what is authoritative;
2. what mathematical or product domain owns an artifact;
3. which language implements that role; and
4. whether an artifact is canonical, supporting evidence, experimental, generated, or vendored.

The ordering is **authority first, domain second, language third**. A language name
must never be the top-level reason an artifact is trusted.

Each adopter has one root `estate.toml`. That file is the machine-readable source
for repository identity, authority planes, target paths, current transitional paths,
language roles, and migration state.

## Standard planes

Only applicable planes are created. Empty silos are forbidden.

| Plane | Default target | Meaning |
| --- | --- | --- |
| policy | `policy/` | governance, repository authority, backend and acceptance policy |
| kernel | `kernel/` | canonical executable validation owned by the repository |
| proof | `proof/` | theorem/claim state, formal proof packages, proof records and models |
| reference | `reference/` | independently executable semantics and golden-vector generation |
| oracles | `oracles/` | non-authoritative differential checking and research engines |
| experiments | `experiments/` | disposable spikes with explicit promotion/deletion criteria |
| schemas | `schemas/` | versioned boundary and serialization contracts |
| conformance | `conformance/` | accepted, rejected, malformed and boundary vectors |
| vendor | `vendor/` | pinned external code; never visually indistinguishable from local ownership |
| tools | `tools/` | audits, generators and repository maintenance only |
| docs | `docs/` | exposition, research notes, handoffs and audits |
| paper | `paper/` | publication artifacts |
| examples | `examples/` | worked examples that are not proof authority |

A repository may omit irrelevant planes. It may add domain-specific planes only when
their authority is explicit in `estate.toml`.

## Canonical skeleton

```text
repository/
├── estate.toml
├── ARCHITECTURE.md
├── policy/
├── kernel/
│   └── <domain>/
│       └── <language only where useful>
├── proof/
│   └── <claim-or-domain>/
├── reference/
├── oracles/
├── experiments/
├── schemas/
├── conformance/
├── vendor/
├── tools/
├── docs/
│   ├── architecture/
│   ├── mathematics-or-domain/
│   ├── research/
│   ├── handoffs/
│   └── audits/
├── tests/
├── examples/
└── paper/
```

The skeleton is illustrative, not a command to create every directory.

## Authority rules

1. Every acceptance or effect boundary has exactly one canonical implementation.
2. A second implementation is a reference, oracle, formal refinement, generated
   adapter, or conformance checker until an explicit authority migration says otherwise.
3. Cross-language disagreement fails closed.
4. Experiments and oracles never issue acceptance verdicts.
5. Generated surfaces are derived artifacts. Their source must be named.
6. Vendored code is pinned external code and must be distinguishable from locally
   owned source.
7. A computation is not a theorem merely because it is deterministic or exhaustive.
8. Moving a file cannot change mathematical or operational authority.

## Language rule

`estate.toml` records roles, not language prestige. Examples:

```toml
[[language]]
name = "Mojo"
authority = "canonical"
roles = ["kernel"]
acceptance_authority = true

[[language]]
name = "Julia"
authority = "supporting"
roles = ["oracle", "experiment"]
acceptance_authority = false
```

An estate repository may use a completely different language assignment while
retaining the same planes.

## Transitional adoption

Large existing repositories use `layout_status = "transitional"`. Each plane
records its future `target` and the existing `current` paths or `current_globs`.
This permits architecture enforcement before disruptive moves.

Migration order:

1. declare authority without changing it;
2. add the estate audit to CI;
3. separate canonical, reference, oracle, experiment, vendor and generated roles;
4. move one bounded context at a time;
5. update imports and tests in the same PR;
6. delete obsolete compatibility mappings only after CI proves the new boundary.

Mass tree reshuffles are discouraged because they obscure semantic changes.

## Required CI gate

Every adopter runs an estate-layout audit that at minimum checks:

- a valid repository identity;
- unique plane identifiers and target paths;
- required current mappings exist;
- authority values and language roles are valid;
- the canonical language/implementation is explicitly named;
- a connected polyglot manifest, when present, names the same repository;
- architecture entrypoints exist.

The first implementation is `tools/audit_estate_layout.py` in
`finite-mandelbrot-research`. It is deliberately repository-independent so it
can be copied or extracted into shared estate tooling later.

## Promotion path

Once at least two repositories adopt v1, extract the generic contract and audit into
the estate's shared governance/tooling repository. Consumer repositories should then
pin that implementation and keep only `estate.toml` plus domain-specific policy.
