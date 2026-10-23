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
- `ESTATE_INTEGRATION.md` — how the lab plugs into the rest of the research
  estate without moving claim authority.
- `estate-adoption.toml` — machine-readable repository routing and authority
  declarations.

The executable exact-reference layer lives in `../../tensor_lab/`:
`algebraic.py` provides rational algebraic predicates and `tensors.py`
provides the first exact tensor kernel oracle, `mode_gram`. These references
stay small on purpose: exact predicates first, accelerator kernels second.

Tensor-frontier Markdown is checked by `../../tools/check_markdown_math.py`;
`pixi run tensor-doc-audit` rejects control-character corruption, lone
single-dollar display fences, and inline dollar delimiters inside display math.
