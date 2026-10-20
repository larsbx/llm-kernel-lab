# First Mojo mode-Gram implementation

This is a Linux-64 FP64 **CPU** experiment. It refines the bounded exact
`tensor_lab.tensors.mode_gram` predicate under `MODE_GRAM_PROTOCOL.md` and keeps
that oracle and `tensor_lab.numpy_baseline` unchanged. Performance remains E0;
cross-implementation numerical agreement is E1 at most. No theorem, rank or
spectral claim follows from a successful run.

## Layout and primitive

The checked Python boundary accepts rectangular, nonempty rank-3 real numeric
inputs and integer modes 0, 1, 2. It converts to aligned native-endian FP64
C-order storage; Fortran order, slices, negative strides and byte-swapped arrays
are supported by this conversion. Strings, complex values, booleans, nonfinite
FP64 inputs and nonfinite outputs are rejected. Fractions are rounded to FP64.
The fixed tolerance is a conformance requirement on the tested domain, not a
uniform forward-error guarantee for all real tensors or ill-conditioned sums.

For tensor shape `(d0,d1,d2)`, the Mojo kernel uses these unfoldings:

| Mode | Row | Column | Implementation |
| --- | --- | --- | --- |
| 0 | i | j*d2+k | Original C-order buffer; NoTrans/Trans |
| 1 | j | i*d2+k | Mojo packs once; NoTrans/Trans |
| 2 | k | i*d1+j | Original buffer as `(d0*d1,d2)`; Trans/NoTrans |

All modes call the existing **LP64 `cblas_dgemm`** primitive through Mojo's C
FFI. There is no Python/NumPy multiplication in the kernel and no custom GEMM.
This first implementation concentrates the Mojo work on layout and validation.
It does not implement a GPU backend.

## Native boundary

`mode_gram_f64` exports eight machine-width signed integers and returns Int32:
`(input_address, d0, d1, d2, mode, input_elements, output_address, output_elements)`.
Addresses point to Float64 allocations; lengths count elements, not bytes.
This ABI is for Linux-64. Callers must supply valid, aligned, disjoint allocations
of the declared lengths and keep them alive throughout the call. The Python
wrapper owns that contract. The native entry validates descriptors before
accessing memory, then checks finite input and output values.

Status codes: 0 success; 1 invalid mode; 2 nonpositive shape or size limit;
3 length mismatch/null address; 4 nonfinite input; 5 nonfinite output.
Both input and output element counts are limited to `2**31-1`; checks avoid
intermediate overflow and keep BLAS dimensions within its LP64 ABI. Allocation
failure is subject to the Mojo allocator's process-failure behavior. A failed
call never yields a numerical observation through the Python wrapper.

## Build and test

Install an LP64 CBLAS library exposing `cblas_dgemm` (Ubuntu: `libblas3`).
The build defaults to `libblas.so.3`; `MODE_GRAM_BLAS` may name a different
LP64 library soname or absolute path. ILP64 BLAS is incompatible. Do not point
this setting at NumPy's bundled ILP64 OpenBLAS. `MOJO` selects the compiler;
otherwise it is found on PATH.

```bash
pixi install --locked
pixi run tensor-mode-gram-build
pixi run tensor-mode-gram-test
pixi run test
```

The dedicated test task requires compilation and fails if Mojo is unavailable;
it cannot silently skip. Ordinary Python-only repository testing skips native
tests when Mojo is absent. CI installs LP64 BLAS and runs the mandatory task.

The committed corpus is exercised in all three modes against both the exact
oracle and independent NumPy baseline. Additional tests cover rectangular
layouts, rational values, input immutability, shape/mode/size failures, null
addresses, nonfinite input/output, and acceptance/rejection on both sides of
the entrywise tolerance. The benchmark also refuses evidence if any measured
output fails this relation.

## Benchmark

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  pixi run tensor-mode-gram-bench --output results/tensor_frontier/mode-gram-local.json
```

The driver refuses to overwrite evidence. It records all four committed cases
and two deterministic dyadic medium cases, all modes, three rounds, ten warmup
calls and fifty measured calls per implementation per round. Parameters are
configurable. Implementation order alternates between rounds. Every result is
checked against the exact rational oracle outside its timed region.

Both APIs receive the same resident C-contiguous FP64 tensor. Wall time covers
the public API, including validation, allocations, Python-to-Mojo FFI, layout
packing and BLAS. Compilation, imports, tensor generation and exact-reference
work are excluded. This is an end-to-end CPU API comparison, not isolated GEMM
latency. The Mojo and NumPy BLAS providers may differ and are recorded; timing
ratios must not be attributed solely to Mojo versus Python.

Each record includes shape, mode, dtype, revision, device, warmups, iterations,
raw samples, wall time, maximum absolute residual and maximum tolerance ratio.
Effective bytes use the common logical floor `8*(input_elements+output_elements)`
per call. This is not measured DRAM traffic: caches, validation reads, internal
BLAS packing and mode-1 packing are excluded. Effective throughput uses this
same floor for both implementations. Compiler/build flags, source and binary
hashes, NumPy/BLAS configuration, thread environment and corpus hash accompany
the records. The saved run is documented under `results/tensor_frontier/`.

## Controlled performance investigation

[The controlled CPU report](../../results/tensor_frontier/MODE_GRAM_CONTROLLED_2026-10-20.md)
separates provider, primitive, native scan/packing, wrapper, allocation and FFI
probes, preserving full public-API measurements and exact checks. The opt-in
`tensor-mode-gram-profile` task does not build implicitly; use the report's
explicit build commands and supply `--library` for native measurements.

The native library also exports benchmark-only `mode_gram_probe_noop` (the same
eight integers as the public ABI), `mode_gram_probe_scan(address, count)` and
`mode_gram_probe_pack(input_address, d0, d1, d2, packed_address)`. Scan and pack
share production helpers. Raw pointer callers own valid aligned buffers of the
declared size; pack requires disjoint input/output buffers containing
`d0*d1*d2` elements. These probes are not public Gram APIs. The no-op never
accesses memory; scan returns 3 for invalid/null descriptors and 4 for nonfinite
values; pack returns 2 for invalid dimensions and 3 for null pointers.

`tools/build_mode_gram_numpy_bridge.py` builds an explicit benchmark-only ABI
adapter to NumPy's loaded ILP64 OpenBLAS. It widens LP64 dimensions before calling
private ILP64 symbols, records library hashes, and adds one C call. Never link
Mojo's LP64 call directly to ILP64 BLAS. The adapter is an experiment, not a
portable deployment dependency or a change to the default system-BLAS build.

Stage measurements overlap and have different cache/dispatch effects. They must
not be added or subtracted as an exact decomposition. The saved investigation
uses the available Mojo 1.1.0 release; locked-nightly validation is still pending.


## Batched CPU follow-up

The optional [batch/workspace API](MODE_GRAM_BATCH.md) amortizes the native
boundary across homogeneous batches and reuses output/packing storage. Finite
checks remain mandatory; mode-1 scan/packing fusion is explicit opt-in. The
[batch report](../../results/tensor_frontier/MODE_GRAM_BATCH_2026-10-20.md) records
matched-provider comparisons against both repeated scalar APIs and checked
vectorized NumPy, including larger cases where NumPy remains faster. This does
not replace the scalar API or establish GPU performance.
