# Mode-Gram CPU evidence — 2026-10-19

Implementation revision: `06585cc64e79b7e856e167215dcb9881e51dd8b2` (clean tree at build and measurement).

Raw evidence: [mode-gram-cpu-06585cc.json](mode-gram-cpu-06585cc.json).

## Outcome

All 108 records passed the declared FP64 entrywise relation against the exact
rational oracle. Maximum absolute residual and tolerance ratio were both zero
for this integer/dyadic benchmark corpus. There are 18 shape/mode combinations,
2 implementations and 3 rounds; each record has 10 warmups and 50 measured
calls. Every measured result was checked, outside the timed region.

**No speedup was observed.** The Mojo/CBLAS public API was slower than NumPy on
every tested shape/mode using the median of the three per-round medians.
This is an E0 result on one shared CPU environment, not a general performance
conclusion. Numerical replication is E1 at most, with no theorem authority.

## Measured medium cases

Times below are microseconds per public API call (median of three round medians).

| Shape | Mode | Mojo/CBLAS µs | NumPy µs | Mojo / NumPy time |
| --- | ---: | ---: | ---: | ---: |
| 16x24x8 | 0 | 39.299 | 8.733 | 4.50 |
| 16x24x8 | 1 | 52.548 | 13.915 | 3.78 |
| 16x24x8 | 2 | 21.307 | 6.795 | 3.14 |
| 24x24x24 | 0 | 273.647 | 26.534 | 10.31 |
| 24x24x24 | 1 | 281.749 | 32.303 | 8.72 |
| 24x24x24 | 2 | 211.449 | 26.964 | 7.84 |

## Environment and limits

- AMD EPYC 9V74 80-Core CPU; Linux x86-64; affinity and cgroup quota in JSON.
- Mojo `1.1.0.dev2026090805 (34562fa1)`, compiled with `-O3`.
- Python 3.12.14; NumPy 2.3.5.
- Mojo links system LP64 `libblas.so.3`; NumPy uses bundled ILP64 OpenBLAS 0.3.30.
- `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.
- Compilation and exact-reference work are excluded. Validation, allocations,
  FFI, packing and GEMM are included. Mojo has stricter guards than the existing
  baseline. Provider and guard differences prevent language-only attribution.
- Effective bytes are the common logical input-read/output-write floor, not
  physical traffic measured by hardware counters.
- The available compiler emits a deprecation warning for count-based `alloc`.
  The unpacked toolchain needed a local config with its installation prefix
  relocated; no repository toolchain pin was changed.

## Validation

Executed the repository estate-layout audit, transcendental firewall, Markdown
math audit and tensor-estate adoption audit successfully. The full local suite
finished with **154 passed, 6 skipped** using the available Mojo toolchain.
Skips: one module requiring PyTorch and five tests requiring Lean/Mathlib.
The 62 new native differential/guard tests passed; six additional benchmark
fail-closed tests passed. The exact oracle, baseline and corpus were unchanged.

`pixi install --locked` could not complete: the package download for
`azure-storage-blobs-cpp-12.18.0-h74b55db_1.conda` failed through the network tunnel.
Therefore validation with the lockfile compiler `1.2.0.dev2026092205` remains
outstanding. The available compiler is within the manifest constraint; this
record does not claim a successful locked-environment run. CI requires native
compilation and differential tests once its dependencies can be installed.

## Reproduction

From the implementation revision, with LP64 CBLAS and a working Mojo compiler:

```bash
python tools/build_mode_gram.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.tensor_frontier.benchmark_mode_gram \
  --output results/tensor_frontier/mode-gram-replay.json
```

Set `MOJO` if the compiler is not on PATH and `MODE_GRAM_BLAS` if selecting
another LP64 CBLAS library. Standard locked-environment commands and boundary
details are in `mojo/tensor_frontier/MODE_GRAM_IMPLEMENTATION.md`.
