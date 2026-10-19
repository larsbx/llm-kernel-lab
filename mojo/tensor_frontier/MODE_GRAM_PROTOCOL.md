# Mode-Gram kernel protocol

This document fixes the comparison contract that a future Mojo `mode_gram`
kernel must satisfy before performance experiments are admitted to the tensor
frontier evidence stream.

## Reference predicate

The authority predicate is `tensor_lab.tensors.mode_gram`.

It accepts a non-empty rectangular real rank-3 tensor and a mode in
`{0,1,2}`, and returns the exact rational matrix

`G_mode = T_(mode) T_(mode)^T`.

The exact predicate is authoritative only on its bounded rational domain. It is
not a production implementation.

## Conventional baseline

`tensor_lab.numpy_baseline.mode_gram_numpy` is the first inexact performance
baseline. It forms the selected unfolding with NumPy and evaluates the Gram
product through NumPy matrix multiplication.

This baseline has no theorem authority.

## Comparison corpus

`../../benchmarks/tensor_frontier/mode_gram_cases.json` is the deterministic
cross-language input corpus. Every implementation must run every listed tensor
for all three modes.

The corpus deliberately includes scalar, dense cubic, signed rectangular, and
skinny layouts. Future cases may be added, but existing case identifiers and
semantics must remain stable.

## Numerical relation

For FP64 results, each observed entry must satisfy

`abs(observed - reference) <= 1e-12 + 1e-12 * abs(reference)`.

The helper `mode_gram_allclose` implements this declared relation for the
reference tests.

Passing this relation is differential evidence only. It does not prove a
mathematical claim and does not justify increasing an authority grade.

## Mojo entry criteria

A Mojo implementation may be added after these items are present:

1. exact predicate: `tensor_lab.tensors.mode_gram`;
2. input domain: dense non-empty real rank-3 tensors, modes 0/1/2;
3. comparison corpus: `mode_gram_cases.json`;
4. FP64 relation: the tolerance above;
5. evidence ceiling: E0 for performance and E1 for replicated numerical
   observations, with no theorem promotion.

The first Mojo version should optimize layout and reduction around existing
matrix-multiplication primitives rather than reimplement GEMM.

## Benchmark record

Each run must record tensor shape, mode, dtype, implementation revision,
device, warmup count, measured iterations, wall time, effective bytes moved,
and maximum differential residual against the declared reference relation.
