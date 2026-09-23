# Tensor frontier lab

This directory specifies a research track inside `llm-kernel-lab` for using
MAX/Mojo as a high-throughput **discovery engine** for tensor mathematics.

The lab is intentionally asymmetric:

```text
literature problem
    |
    v
algebraic problem contract
    |
    +--> LLM representation variants
    |
    v
MAX/Mojo empirical search
    |
    v
candidate witness / anomaly
    |
    v
exact reconstruction + independent oracle
    |
    v
Lean/exact checker
    |
    v
claim eligible for promotion
```

A floating-point hit, benchmark win, phase diagram, fitted exponent, or LLM
completion is evidence, not proof.

The track is governed by:

- `LITERATURE_REVIEW_2026-09.md` — current literature baseline, frontier gaps,
  and MAX/Mojo experiment opportunities through September 2026.
- `RESEARCH_PLAN.md` — frontier questions, work packages, evidence ladder.
- `TRANSCENDENTAL_FIREWALL.md` — algebraic vocabulary and non-equivalence
  rules.
- `TRIANGLE_BIAS_PROTOCOL.md` — paired LLM experiment for trig versus
  algebraic representations.
- `../../benchmarks/tensor_frontier/manifest.toml` — initial kernel/experiment
  registry.

The first executable oracle lives in `../../tensor_lab/algebraic.py`.  It is
small on purpose: exact rational identities first, accelerator kernels second.
