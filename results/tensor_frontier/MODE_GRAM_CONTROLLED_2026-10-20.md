# Controlled mode-Gram CPU investigation — 2026-10-20 UTC

## Finding

The dominant confound in the earlier comparison was the BLAS provider. In this
run, 24×24×24 mode 0 took **282.730 µs** through Mojo/system BLAS and
**28.889 µs** through Mojo using the exact OpenBLAS binary loaded by NumPy.
NumPy measured **26.930 µs** in the matched-backend run. This is a measured
configuration effect on this host, not an inherent language comparison.

The implemented targeted change is an **opt-in benchmark backend adapter**:
LP64 CBLAS dimensions are widened to NumPy OpenBLAS's ILP64 ABI. It forwards
GEMM/SYRK without performing arithmetic. It adds one C call and deliberately
is not a new production default: NumPy's private symbols are version-specific.
The production GEMM algorithm, input/output guards, and default provider stay
in place. Instrumentation shares the production finite-scan and mode-1 packing
helpers; it does not replace them with NumPy approximations.

**E0 performance; at most E1 numerical replication.** No GPU, theorem,
rank/spectral, or general speedup claim follows.

## Evidence and source identity

- [System-provider raw samples](mode-gram-system-b5aa7ab.json.gz).
- [Matched-provider raw samples](mode-gram-matched-b5aa7ab.json.gz).
- Each gzip file decompresses to the original unedited JSON measurement output.
- Measured local source revision: `b5aa7ab8a8a613a4a81a43ea5b98b45345eb956d`.
- Published source snapshot: [`ab280cf998bcb63d917f73fa8c03bfce2e9d1fe2`](https://github.com/larsbx/llm-kernel-lab/commit/ab280cf998bcb63d917f73fa8c03bfce2e9d1fe2).
- Both revisions have the identical Git tree
  `38fe2fe0f1489406a91b42969d514e4cc466353f`. The publication API assigns its own
  commit identity; the measurement files retain their original provenance.
- Both measurements report a clean working tree. Source, corpus, native binary,
  adapter and backend hashes accompany the raw samples.

Two sequential, separate-process runs contain **2,976 records**: 1,488 per
provider configuration. Each uses eight tensors, all three modes, three rounds,
five warmups and thirty measured calls per lane. The four committed corpus
cases and the two previous dyadic cases are unchanged; added cases are
48×48×48 and 96×16×16 with the same deterministic dyadic formula.
The existing exact Fraction oracle computes every reference, including the
large cases. Every warmup and measured numerical result is checked outside the
timed region. All full-Gram and defined-upper-triangle results passed the
existing tolerance, with maximum absolute residual and tolerance ratio zero.
Non-numerical probes have separate structural/status checks, not fabricated
numerical residuals.

The preliminary provider-only run was excluded: it overlapped setup/validation.
The two saved runs were serialized after validation, with no concurrent test
or build jobs launched by this investigation. Both pin affinity to CPU 0;
external shared-host activity remains uncontrolled.

## Public API measurements

Microseconds; median of three per-round medians. System and matched columns
come from separate runs. NumPy's column is from the matched run only; raw
samples also preserve the NumPy control in the system run.

| Shape | Mode | Mojo system BLAS µs | Mojo matched BLAS µs | NumPy matched run µs | Matched Mojo / NumPy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 24x24x24 | 0 | 282.730 | 28.889 | 26.930 | 1.07 |
| 24x24x24 | 1 | 299.035 | 32.774 | 44.477 | 0.74 |
| 24x24x24 | 2 | 204.883 | 30.025 | 26.741 | 1.12 |
| 48x48x48 | 0 | 4558.454 | 344.505 | 225.578 | 1.53 |
| 48x48x48 | 1 | 4645.109 | 394.290 | 276.053 | 1.43 |
| 48x48x48 | 2 | 3341.046 | 421.701 | 252.513 | 1.67 |
| 96x16x16 | 0 | 1786.492 | 151.310 | 98.532 | 1.54 |
| 96x16x16 | 1 | 355.528 | 44.832 | 56.053 | 0.80 |
| 96x16x16 | 2 | 241.342 | 34.657 | 31.116 | 1.11 |

The matched result is shape/mode dependent: mode 1 is locally below NumPy at
24³ and 96×16×16, whereas the other listed cases remain above it. This does not
establish a general advantage. The 48³ cases make multiplication costs clearly
larger than fixed wrapper overhead; the scalar corpus shows the opposite regime.

## What the controlled probes explain

| Matched-backend probe | 24³ mode 0 µs | 48³ mode 0 µs | 48³ mode 1 µs |
| --- | ---: | ---: | ---: |
| python_prepare_resident | 1.873 | 4.296 | 4.307 |
| output_allocate | 0.381 | 0.641 | 0.641 |
| mojo_ffi_noop | 0.932 | 0.911 | 0.911 |
| mojo_input_finite_scan | 6.550 | 49.084 | 49.173 |
| mojo_output_finite_scan | 0.671 | 1.452 | 1.462 |
| lp64_gemm | 13.881 | 268.182 | 275.177 |
| numpy_blas_gemm | 14.146 | 267.986 | 288.527 |
| numpy_blas_syrk_upper | 21.788 | 211.777 | 224.612 |
| mojo_native_preallocated | 21.036 | 326.359 | 418.642 |
| mojo_public | 28.889 | 344.505 | 394.290 |
| numpy_public | 26.930 | 225.578 | 276.053 |

1. **Provider:** in the system run, 48³ mode-0 direct LP64 GEMM took
   4,567.552 µs versus 280.797 µs through NumPy's OpenBLAS GEMM. In the matched
   run, adapter GEMM and direct OpenBLAS GEMM took 268.182 and 267.986 µs.
   Equalizing the backend removes the large provider difference; the adapter
   is not a hidden arithmetic implementation.
2. **Validation:** native input scanning takes 6.550 µs at 24³ and 49.084 µs
   at 48³ mode 0. Output scanning grows with the Gram dimension. NumPy's public
   baseline does not enforce the same finite-input/output contract. These
   checks remain enabled; removing them would change the comparison contract.
3. **Packing:** the shared native mode-1 pack helper, into preallocated storage,
   takes 5.208 µs at 24³ and 65.047 µs at 48³. Modes 0 and 2 avoid this pack.
   The public native lane still includes the mode-1 allocation/free; the pack
   probe intentionally excludes them. Native allocator cost is not separately
   identified by this experiment.
4. **Fixed overhead:** the eight-argument Python-to-Mojo no-op is roughly
   0.9 µs in the listed mode-0/1 matched cases. Python preparation and empty
   output allocation are separately measured; empty allocation does not imply
   physical page touching. These fixed costs matter on tiny tensors but cannot
   explain a multi-millisecond system-BLAS gap.
5. **Primitive:** [NumPy 2.3.5's matmul source](https://github.com/numpy/numpy/blob/v2.3.5/numpy/_core/src/umath/matmul.c.src)
   selects SYRK for eligible shared-pointer transpose products and mirrors the
   triangle. Mojo uses full GEMM. Direct SYRK upper-only is **slower** than GEMM
   at 24³ mode 0 (21.788 versus 14.146 µs), but faster at 48³ mode 0 (211.777
   versus 267.986 µs). The distinct-right-operand NumPy lane controls this
   dispatch difference. Upper-only timings exclude mirroring and are not full
   Gram timings. The data do not justify a blanket GEMM-to-SYRK change.

These are isolated warm-cache probes with overlapping work, **not an additive
cost decomposition**. Do not sum stage medians or subtract independently timed
lanes to claim exact overheads. For example, matched 48³ mode-1 preallocated
native latency exceeds its public-API median; this is timing variation, not
negative allocation cost. Raw per-round samples make that variation visible.
No hardware counters or DRAM-bandwidth claim is made.

## Exact environment

- AMD EPYC 9V74 80-Core Processor; affinity `[0]` in both saved runs;
  cgroup CPU quota `800000 100000`.
- Python 3.12.14; NumPy 2.3.5.
- Mojo **1.1.0 (8189361e)** from locally installed `mojo==1.1.0`; `-O3`.
  This is different from both the earlier nightly and the lockfile compiler.
- System provider: Ubuntu `libblas3 3.12.0-3build1.1`, resolved
  `/usr/lib/x86_64-linux-gnu/blas/libblas.so.3.12.0`; LP64 `cblas_dgemm`.
  No runtime thread-count API is exposed, so its raw metadata correctly records
  `actual_threads: null`. It must not be inferred from environment variables.
- NumPy provider: `libscipy_openblas64_-fdde5778.so`, OpenBLAS 0.3.30,
  `USE64BITINT DYNAMIC_ARCH NO_AFFINITY SkylakeX MAX_THREADS=64`, pthreads.
  Runtime thread count was set to one and verified before and after each run.
- `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` before launch.
- Matched backend uses that **same resolved file and hash** through the C ABI
  adapter; `shared_numpy_backend: true`. GEMM symbol resolution and NumPy's
  actual dynamic dependencies are checked, rather than trusting build-time paths.
- JSON includes full `np.show_config`, dynamic dependencies, SHA-256 hashes,
  compiler/link commands, adapter compiler version and runtime BLAS configuration.
- The local compiler warns about count-based `alloc` and unavailable Crashpad.
  Compilation succeeds. No repository toolchain pin was changed.

## Validation and integration limits

- Full available-toolchain suite: **173 passed, 6 skipped**. Skips are the
  PyTorch-dependent module and five Lean/Mathlib tests.
- **64 native tests** pass with the matched provider, including the corpus,
  rational/layout cases, guards, shared helpers and an end-to-end profile.
- **17 profile tests** cover both direct provider ABIs, the adapter, layouts,
  upper-triangle boundaries, quick-return immutability, bad numerical samples,
  thread settings, invalid counts and source/binary hash enforcement.
- Estate-layout, transcendental firewall, Markdown-math and tensor-estate
  adoption audits pass. Oracle, NumPy baseline and committed corpus are unchanged.
- Pixi is absent in this execution environment. Locked-nightly validation remains
  outstanding; available-toolchain success is not a locked-environment result.
- PR #16 is integrated into `feat/tensor-mode-gram-baseline`; #14 and #15 remain
  open. This follow-up targets that baseline branch. It does not imply integration
  into `main`. Preserve #14 → #15 when promoting the existing stack.

## Reproduction

Use Linux x86-64 with a working Mojo compiler (`MOJO` may select its path), GCC,
LP64 `libblas.so.3`, and the supported NumPy 2.3.5 ILP64 OpenBLAS wheel. Build
before measuring; do not run the test suite concurrently with timings.

```bash
python tools/build_mode_gram_numpy_bridge.py
python tools/build_mode_gram.py --output .build/tensor_frontier/libmode_gram_system.so
python tools/build_mode_gram.py \
  --blas "$PWD/.build/tensor_frontier/libnumpy_cblas_lp64.so" \
  --output .build/tensor_frontier/libmode_gram_matched.so

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 0 python -m benchmarks.tensor_frontier.profile_mode_gram \
  --library .build/tensor_frontier/libmode_gram_system.so \
  --output results/tensor_frontier/system-replay.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 0 python -m benchmarks.tensor_frontier.profile_mode_gram \
  --library .build/tensor_frontier/libmode_gram_matched.so \
  --blas "$PWD/.build/tensor_frontier/libnumpy_cblas_lp64.so" \
  --output results/tensor_frontier/matched-replay.json
```

Choose an allowed CPU on other hosts. Output paths must be new. The default
profile uses five warmups, thirty iterations and three rounds; `--no-large`
limits it to the previous six cases. Omit `--library` only for a provider/Python
investigation: that is recorded explicitly and generates no Mojo measurements.
For a non-wheel NumPy build, pass its actual dependency with `--numpy-blas` and
`--numpy-abi lp64`; unsupported layouts/symbols fail closed. The adapter builder
is intentionally specific to the recorded ILP64 wheel ABI.

Inspect compressed evidence without executing any repository code:

```bash
python -m gzip -d results/tensor_frontier/mode-gram-matched-b5aa7ab.json.gz
python -m json.tool results/tensor_frontier/mode-gram-matched-b5aa7ab.json
```

Next measurements should validate the locked compiler and a supported optimized
LP64 OpenBLAS deployment (avoiding private NumPy symbols), then assess native
scan/packing optimization or a measured primitive dispatch policy. The current
evidence supports backend selection first; it does not support weakening guards,
a universal SYRK replacement, or a GPU claim.
