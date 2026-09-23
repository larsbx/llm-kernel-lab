# Mojo tensor-frontier kernels

This directory is the accelerator side of the tensor empirical-mathematics track.

It is intentionally empty of mathematical authority.  A kernel may emit an
observation or candidate witness; it may not decide that a theorem, rank claim,
facet, phase transition, or decomposition is true.

## Entry criteria for a kernel

Before adding a Mojo implementation, the PR must name:

1. the scalar/reference predicate it refines;
2. the input domain and all nonzero/invertibility guards;
3. the comparison corpus;
4. the tolerated numerical relation, if the implementation is inexact;
5. the evidence grade its output is allowed to reach.

The first planned interfaces are:

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

## Algebraic surface

The kernel API should consume algebraic data directly where practical:

- Gram/bilinear data instead of angles;
- rotor pairs `(c,s)` satisfying `c*c+s*s=1` instead of trig calls;
- projective pairs instead of tangent division;
- paired quadratic-algebra coordinates instead of complex-phase syntax;
- quadrance/squared residual instead of square-root norms;
- direct multiplicative weights instead of exp/log parameterization.

The source audit in `../../tools/audit_transcendentals.py` scans this directory
once `.mojo` files are added.

## Performance policy

Optimize the awkward mathematics around GEMM: layout-sensitive contraction,
permutation+contraction fusion, reductions, mode-Gram formation, batched rank
probes, and structured local updates.  Do not add a custom GEMM merely to
duplicate an already optimized primitive.

Benchmarks live in `../../benchmarks/tensor_frontier/` and must report both
throughput and differential agreement with the declared reference predicate.
