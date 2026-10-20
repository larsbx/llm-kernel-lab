# Mode-Gram CPU batching and fusion investigation — 2026-10-20 UTC

Performance authority: **E0**. Numerical replication: **E1 at most**. No theorem,
rank, spectral, GPU, CUDA-migration or general Mojo speedup claim follows.

## Result

Native batching and reusable buffers reduce overhead on the tested small and
medium tensors. With 32 tensors of shape 24×24×24, the reusable fused Mojo path
measured 1.19–1.31× faster than the checked vectorized NumPy batch path across
modes 0–2. At 48×48×48, NumPy won most comparisons; mode 1 at batch size 32 was
an exception in this run. The advantage is workload- and API-specific.

Fusing the mode-1 input scan with packing helped the 24³ and 48³ cases, but
regressed at 16×24×8 with batch size 32. A five-round focused recheck did not reproduce that regression: fused and
separate preparation were essentially tied, and both beat checked NumPy.
Both runs are retained rather than selecting only the favorable result.

These results justify exposing a checked native batch/workspace API and an
explicit fusion experiment. They do not justify replacing the scalar API,
switching all matrix products to SYRK, or declaring Mojo universally faster.

## Evidence and provenance

- [Broad raw samples](mode-gram-batch-c487b3a.json.gz): 1,134 records = seven
  shapes × three modes × batch sizes 1, 8, 32 × six paths × three rounds.
  Each record contains five warmups and 30 measured samples; checks are outside
  timing. Warmup checks are mandatory, though warmup times are not retained.
- [Focused recheck](mode-gram-batch-recheck-c487b3a.json.gz): 90 records for
  16×24×8, batch size 32, all modes, six paths, five rounds, ten warmups and
  100 measured samples. The exact driver text and its SHA-256 are embedded.
- Every member of every warmup and measured result passed the unchanged
  `abs(actual-reference) <= 1e-12 + 1e-12*abs(reference)` relation. Broad-run
  maximum absolute residual and tolerance ratio were both **zero**.
- Inputs use the four committed corpus cases, the existing deterministic
  16×24×8 and 24³ dyadic cases, and a deterministic 48³ extension. Members
  cycle through base + `[0, 1/8, -1/4, 3/8]`, providing distinct members with
  unchanged deterministic construction. References are computed separately
  using the exact Fraction oracle, not NumPy or BLAS.
- Measured local source commit: `c487b3ab29eb919a531a99397c5ce13b3f2392ac`.
  Published equivalent: `3de0e78f0305f088d694a8a7f7f31b9ff7dcd0ad`.
  Both Git trees: `383fd7de9be3046129de3ef4609587f5cd531105`.
  API publication changed commit identity, not file contents. Both native builds
  and both runs used a clean tree. Raw metadata records source, transitive
  dependency, binary, bridge, BLAS and corpus hashes.

The final wrapper makes fusion opt-in; the measured source used fused defaults.
The benchmark's allocated lane explicitly requests fusion in the final source,
preserving its measured meaning. Explicit `fused=True`/`False` workspace paths
and native arithmetic are unchanged. No timing is relabeled as a new run.

## Controlled environment

- CPU: AMD EPYC 9V74 80-Core Processor; process pinned to logical CPU 0.
  Linux 6.18.55 x86-64, glibc 2.39; cgroup `cpu.max = 800000 100000`.
  No concurrent test/build workload was run during measurements. Host frequency
  and external workloads are not controlled; warm-buffer medians are local data.
- Python 3.12.14; NumPy 2.3.5; Mojo **1.1.0 (8189361e)**, `build --emit shared-lib
  -O3`. C adapter: GCC 13.3.0, `-O3 -Wall -Wextra -Werror -fPIC -shared`.
- Both implementations reach the **same loaded NumPy OpenBLAS binary**:
  `libscipy_openblas64_-fdde5778.so`, SHA-256
  `ee2039cf13a45e1e89f04d0cea9585657d93ffaa425594e1935161013fdf4e12`.
  Configuration: `OpenBLAS 0.3.30 USE64BITINT DYNAMIC_ARCH NO_AFFINITY SkylakeX
  MAX_THREADS=64`. `SkylakeX` is the provider-reported kernel configuration, not
  the host CPU brand. Full NumPy configuration and dependency paths are in JSON.
- Mojo's LP64 call goes through the benchmark-only widening adapter to
  `scipy_cblas_dgemm64_`. NumPy calls its own ILP64 GEMM/SYRK dispatch. This
  matches the provider, not necessarily the chosen primitive or packing.
  Adapter SHA-256:
  `66d01f078c9c125bcdfbc550210b4e5e875bb7e95495305ee3234024db98cb3e`.
- `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `BLIS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` all equal **1** before launch.
  The actual shared OpenBLAS runtime thread count was verified as **1**.
  The LP64 adapter has no separate threading implementation.
- Pixi is unavailable. The prior locked-package download limitation remains
  unresolved; this is available-release validation, not locked-nightly
  validation. Local test dependencies were installed in ignored build storage
  (NumPy 2.3.5, pytest 8.4.2, huggingface_hub 2.2.0). No GPU device is exposed.

## Timed paths and numerical safeguards

All paths receive the same resident C-contiguous FP64 batch. Compilation,
imports, deterministic generation, workspace construction for reuse paths and
exact reference/checking work are excluded. Timing includes Python boundary
work, native calls, guards and synchronous CPU computation. Lane order rotates
and reverses deterministically across rounds. Input immutability is checked.

| Path | Timed work |
| --- | --- |
| Mojo scalar loop | Checked scalar API per item, per-item output allocation, Python loop and final stack |
| NumPy scalar loop | Existing independent scalar baseline per item and final stack; lacks finite guards |
| Mojo allocated batch | Fresh workspace/output, checked wrapper, one native crossing, fused mode-1 preparation, sequential GEMMs |
| Mojo reusable separate | Existing buffers, checked wrapper, one native crossing, full scan then mode-1 packing, sequential GEMMs |
| Mojo reusable fused | Same reusable path, finite checking combined with mode-1 packing |
| NumPy reusable checked | Vectorized batched `matmul`, reused output, input/output finite checks; mode-1 unfolding may allocate |

Native batches validate the whole input before the first GEMM. Mode 1 uses an
input-sized scratch buffer for the whole batch so a late invalid item leaves
all output untouched. Modes 0 and 2 need no explicit packing. Every native path
uses GEMM. The NumPy checked batch is a stronger comparison than a Python loop,
but is a separately written wrapper with NumPy dispatch and temporary guard
arrays. These are equivalent finite-value requirements, not identical code.

All outputs are checked against the independent exact oracle. The new native
unit tests also compare with the unchanged NumPy baseline. Shared BLAS providers
do not replace these independent checks.

## Measured batch latency

Numbers below are **microseconds per whole batch of 32**, computed as the
median of the three round medians. Divide by 32 for amortized per-tensor time.
Raw distributions are retained; these are not confidence intervals.

| Shape | Mode | Mojo scalar loop | NumPy scalar loop | Mojo allocated | Mojo separate reuse | Mojo fused reuse | NumPy checked reuse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16×24×8 | 0 | 326.832 | 318.500 | 146.811 | 134.543 | 132.940 | 181.564 |
| 16×24×8 | 1 | 493.853 | 607.255 | 422.721 | 262.606 | 459.962 | 349.416 |
| 16×24×8 | 2 | 281.724 | 238.574 | 101.623 | 92.624 | 92.940 | 129.014 |
| 24³ | 0 | 941.252 | 981.943 | 733.344 | 732.372 | 702.888 | 920.807 |
| 24³ | 1 | 1,114.328 | 1,242.581 | 856.299 | 961.628 | 857.691 | 1,126.772 |
| 24³ | 2 | 905.383 | 994.728 | 662.908 | 665.101 | 672.758 | 798.987 |
| 48³ | 0 | 11,346.763 | 8,537.579 | 11,429.272 | 11,406.803 | 11,564.566 | 9,632.462 |
| 48³ | 1 | 12,959.677 | 12,543.255 | 12,907.769 | 14,149.734 | 12,784.623 | 13,953.879 |
| 48³ | 2 | 11,401.470 | 8,792.778 | 11,151.879 | 11,400.609 | 11,201.263 | 10,150.997 |


For a small 2×2×2 tensor in mode 2, the batch-32 scalar Mojo loop took 181.358 µs,
the allocated native batch 20.716 µs, and the reusable fused batch 12.825 µs.
Checked vectorized NumPy took 17.987 µs. This is a wrapper-dominated case, not
proof of a faster multiplication algorithm. At batch size 1 the same allocated
native path took 15.524 µs versus 8.383 µs for reused buffers; workspace setup
can outweigh the benefit if a caller does not actually reuse it.

At 24³, batch-32 mode 1, fusion reduced reusable native batch time from
961.628 to 857.691 µs (10.8% lower); checked NumPy took 1,126.772 µs. At 48³,
batch-32 mode 0, native fused reuse took 11,564.566 µs versus NumPy's
9,632.462 µs. Batching no longer eliminates the dominant matrix-product and
validation cost. The previous controlled report's primitive and scan probes
explain plausible remaining costs, but this batch run is not a new GEMM/SYRK
isolation experiment.

The allocated-versus-reused paths also differ in Python setup and validation
calls, so their difference must not be called pure allocation cost. Modes 0
and 2 have identical preparation for the fused/separate switches; their timing
variation illustrates measurement noise. In particular, small percentage
changes and isolated wins should not be treated as stable optimizations.

## Focused regression and optimization decision

| Mode | Separate reuse, µs | Fused reuse, µs | Checked NumPy reuse, µs |
| --- | ---: | ---: | ---: |
| 0 | 135.504 | 133.531 | 190.376 |
| 1 | 249.516 | 248.939 | 316.797 |
| 2 | 96.371 | 92.755 | 129.024 |


For mode 1, the broad run measured 459.962 µs fused versus 262.606 µs separate;
the focused recheck measured 248.939 versus 249.516 µs (0.23% lower with fusion).
Round medians overlapped, so the recheck does not demonstrate a useful fusion
advantage for this shape. The cause of the broad-run regression is unresolved;
these data do not isolate host scheduling, cache effects or memory placement.
The focused maximum absolute residual and tolerance ratio were also zero.

Keep native batching and buffer reuse available as an opt-in API. Use separate
scan/packing by default (`fused=False`); retain `fused=True` for explicitly
measured workloads such as the cube cases. No shape threshold is inferred from
this small corpus. The native implementation and both strategy paths are the
ones measured above. Final changes only make wrapper defaults conservative and
the allocated benchmark lane's fusion argument explicit.

## Reproduce

Use the measured published source commit above for exact source provenance,
or the follow-up branch with explicit fusion arguments. Start a fresh process
with the following environment and an available compiler. Select an allowed CPU
if CPU 0 is not in your affinity set. The adapter requires a compatible NumPy
OpenBLAS build; failures are explicit and must not be replaced by invented data.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MOJO=/absolute/path/to/mojo
python tools/build_mode_gram_numpy_bridge.py
python tools/build_mode_gram.py \
  --blas "$PWD/.build/tensor_frontier/libnumpy_cblas_lp64.so" \
  --output .build/tensor_frontier/libscalar_batch_control.so
python tools/build_mode_gram_batch.py \
  --blas "$PWD/.build/tensor_frontier/libnumpy_cblas_lp64.so" \
  --output .build/tensor_frontier/libbatch_matched.so
taskset -c 0 python -m benchmarks.tensor_frontier.benchmark_mode_gram_batch \
  --scalar-library .build/tensor_frontier/libscalar_batch_control.so \
  --batch-library .build/tensor_frontier/libbatch_matched.so \
  --blas "$PWD/.build/tensor_frontier/libnumpy_cblas_lp64.so" \
  --output results/tensor_frontier/mode-gram-batch-local.json
```

For this host, Python imports came from
`PYTHONPATH="$PWD/.build/python-deps:$PWD/.build/mojo-toolchain:$PWD"`, and
`MOJO="$PWD/.build/mojo-toolchain/bin/mojo"`. These are local installation paths,
not required locations. Output paths must not exist; evidence is never overwritten.

To reproduce the focused run, extract `focused_driver` from its gzip JSON to
`.build/recheck_batch.py`, use a new output filename in that driver, and run it
under the same environment with `taskset -c 0`. The embedded source documents
exactly how the existing corpus was restricted without changing arithmetic.

## Validation and integration

The local repository suite passed **256 tests with 6 skipped** using available
Mojo 1.1.0. All four repository audits passed (estate layout, transcendental
firewall, Markdown math, tensor estate adoption). The **68 native batch tests**
also passed with the matched NumPy backend. Nine batch-harness tests check
numerical drift/nonfinite rejection, shape/count guards, NumPy guard ordering
and transitive build provenance. The six skips concern unavailable Lean/Mathlib
and optional training dependencies, not a skipped mandatory Mojo task.

Both whole-suite and matched-provider native checks preserve the independent
exact oracle and NumPy comparison. CI includes a mandatory batch native task;
queued or unrun CI is not counted as passed.

PR #17's thread-control review was fixed in
`2522e6ed03f1170dcdb29b98822e665c99bdb948`; all five environment controls are now
required, with unset/non-one BLIS and Accelerate regressions. Its historical
raw evidence was not modified.

This follow-up targets `perf/mode-gram-controlled` (PR #17), which targets
`feat/tensor-mode-gram-baseline` (PR #15). PR #15 targets
`feat/tensor-mode-gram-reference` (PR #14), which targets `main`. PR #16 already
merged into the baseline feature branch. #14, #15 and #17 remain open at this
checkpoint; no main integration or locked-toolchain CI success is asserted.

[API and GPU follow-up contract](../../mojo/tensor_frontier/MODE_GRAM_BATCH.md)
records ownership, guard behavior and the experiments needed on actual GPU
hardware. CPU batching/fusion results do not establish CUDA migration ease,
GPU portability or GPU speedup. The next GPU experiment must compare both
resident and transfer-inclusive FP64 batches against an independent strong
vendor-library baseline before drawing those conclusions.
