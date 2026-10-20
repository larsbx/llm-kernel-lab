# Checked CPU batches and reusable workspace

This is an opt-in homogeneous FP64 CPU experiment. The scalar mode-Gram API,
exact Fraction oracle, committed corpus and independent NumPy baseline are
unchanged. Performance observations remain E0; numerical replication is at most
E1. See the [measured report](../../results/tensor_frontier/MODE_GRAM_BATCH_2026-10-20.md).

## Python API

Build with `pixi run tensor-mode-gram-batch-build`, then:

```python
import numpy as np
from tensor_lab.mojo_mode_gram_batch import MojoModeGramBatch

batch = np.ones((8, 16, 24, 8), dtype=np.float64)
kernel = MojoModeGramBatch()
workspace = kernel.workspace(batch.shape, mode=1)
output = workspace.compute(batch)  # borrowed, reused by the next call
saved_output = output.copy()       # copy if it must outlive that call
allocated_output = kernel(batch, mode=1)  # fresh workspace on each invocation
```

`batch` has shape `(batch_size, d0, d1, d2)` with all dimensions positive.
All members have the same shape and mode. Real numeric inputs are converted to
aligned native-endian C-order FP64. Noncontiguous, reversed, byte-swapped and
rational inputs are accepted through conversion. Booleans, complex values,
strings, nonfinite FP64 inputs and nonfinite outputs are rejected. Total input
and output element counts must each fit `2**31-1`. Fractions are rounded to
FP64 at the same computational boundary as the scalar API.

`compute(..., fused=True)` combines finite checking with mode-1 packing.
The default `fused=False` scans the whole input before packing. Fusion is opt-in
because the measured benefit varies by case and one broad-run regression did
not reproduce in a focused recheck. Modes 0 and 2 use the same
scan in both strategies and need no explicit packing. The entire batch is
validated before any Gram output is written; a late invalid member leaves the
previous output untouched. Scratch can change on failure. Output overflow
raises `FloatingPointError`; discard the output buffer after that failure.

A workspace owns one output array and, for mode 1, a scratch array equal in
size to the whole input batch. `workspace_bytes` reports their combined size.
Setup and dtype/layout conversions can allocate; the native batch routine
itself does not allocate. BLAS may manage its own internal storage. The API
rejects concurrent use of the same workspace and input aliasing with its
buffers. Callers must keep inputs stable during execution and must finish
reading or copy borrowed results before reusing a workspace. The lock does
not extend the lifetime of a borrowed result past the next call.

## Native contract

`mode_gram_batch_f64` is a Linux-64 C ABI with twelve machine-width integers:

```text
(input_address, batch_size, d0, d1, d2, mode, input_elements,
 output_address, output_elements, workspace_address, workspace_elements, fused)
```

Return codes are 0 success, 1 invalid mode, 2 invalid shape/LP64 bound,
3 length/null descriptor failure, 4 nonfinite input, 5 nonfinite output,
6 invalid strategy. `fused` must be 0 or 1. Output length is
`batch_size * shape[mode]**2`; mode-1 workspace length equals input length.
Other modes require workspace length zero and permit a null workspace address.
Raw pointer callers own valid, aligned, disjoint allocations with the declared
lengths, stable input and adequate lifetime. Descriptor checks do not establish
allocation validity or detect raw-pointer overlap. The Python API supplies
these guarantees for its own buffers.

One Python/native crossing serves the batch, followed by a sequential native
loop of CBLAS GEMMs. This is neither parallel batching nor a vendor batched-BLAS
primitive. The matched-provider benchmark uses the experimental LP64-to-ILP64
adapter from the controlled investigation; the default build still uses system
LP64 BLAS. The adapter must never be replaced by a direct LP64 call to an ILP64
symbol.

## Validation and benchmarking

```bash
pixi run tensor-mode-gram-batch-test
```

This task requires the compiler and native tests; it cannot silently skip.
CI runs it in addition to the scalar native task. Batch tests compare every
member against both the exact oracle and NumPy for all three modes and both
preparation strategies. Guards cover late failures, layouts, aliasing, reuse,
overflow, sizes and concurrency rejection. Build metadata hashes both the batch
source and its imported scalar helper source.

The report gives the matched-provider build and benchmark commands. The driver
compares repeated scalar APIs, allocated native batches, reusable native batches
with separate/fused preparation, and vectorized NumPy batches with input/output
finite checks. It verifies the actual loaded backend and source/binary hashes,
requires all five thread controls to equal one, and records the OpenBLAS runtime
thread count. Every warmup and measured member is checked against the exact
oracle outside timing. The allocated/reused comparison includes differences in
wrapper setup and checks; it is not an isolated allocation timer.

## GPU follow-up boundary

No GPU is exposed in the execution environment used for this CPU report.
No GPU kernel, CUDA migration result or GPU speedup is claimed. A GPU follow-up
needs an executable, independently checked experiment on available hardware:

1. Port the unfolding/packing and guard semantics for all modes, retaining FP64
   and the same deterministic corpus and exact oracle relation.
2. Compare a Mojo implementation against a strong independently implemented
   vendor-library batch path (cuBLAS on NVIDIA), recording hardware, driver,
   compiler, library versions and threading/stream settings.
3. Separate device-resident steady-state batches from host-to-device transfers,
   device allocations, launch, synchronization and result transfer. Report both
   complete checked API latency and kernel-only timings with explicit boundaries.
4. Test sufficiently large shapes/batches and report actual FP64 capability;
   mixed precision or tensor-core changes require a separately declared contract.
5. Measure before deciding whether custom fusion improves on the vendor path.

CUDA migration would entail porting and validating kernels, layouts and runtime
integration. This CPU CBLAS wrapper supplies no evidence about that effort or
performance. Portable source and reduced orchestration overhead are hypotheses
to test on each supported target, not established consequences of this report.
