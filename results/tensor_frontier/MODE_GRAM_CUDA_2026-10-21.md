# CUDA prototype: offline validation, hardware acceptance open

## Result

A bounded FP64 Mojo CUDA prototype and an independent cuBLAS strided-batch
comparison harness are implemented for all three tensor modes. NVIDIA-target
compilation and CPU-executable contract checks passed. **No GPU numerical or
performance measurement was made.** The CPU batch speedups are not GPU evidence.

The [contract and reproduction commands](../../mojo/tensor_frontier/MODE_GRAM_CUDA.md)
specify shapes, numerical tolerance, failure behavior, stream ownership and
timing boundaries. The exact Fraction oracle, deterministic committed corpus
and original NumPy baseline were not modified.

## Evidence and identities

- Local validated source commit: `e9b5af2c6ef9a9ed764a07a90536e4ab0fa39d0e`.
- GitHub source publication commit: `bb4cbb22cdb3cfcdef246fab135d3359caa747e6`.
- Both have the identical complete Git tree: `12aef4d569a1e0ad1415cfe72375350ab5e0c2a5`.
- Base: `e0456eceff9f539b6508fe0b12f884128fc77c45`, the head of PR #17 including merged #18.
- [Offline build receipt](mode-gram-cuda-offline-build-2026-10-21.json): explicit
  `nvidia:sm_80`, Mojo 1.1.0 (`8189361e`), source/binary hashes and command.
- [Environment probe](mode-gram-cuda-environment-2026-10-21.json): no NVIDIA
  device nodes, `nvidia-smi` or `nvcc`; CuPy reports `cudaErrorInsufficientDriver`.
  Its measurement array is empty and status is `probe_only`.

The compiler target is selected for offline compilation; it is not a detected
GPU capability. MAX 26.6.0 and CuPy 13.6.0 were installed for API inspection.
Local tests used Python 3.12.14 and NumPy 2.5.4; the new CI lane explicitly pins
NumPy 2.3.5, Mojo 1.1.0 and MAX 26.6.0. Existing Pixi nightly validation is not
claimed. The build emits an allocation-API deprecation warning; compilation
and the compiled host probe succeed.

## Checks completed

Final repository suite: **304 passed, 12 skipped**. This includes **36 new
host-executable checks**; six new hardware cases were explicitly skipped in
the CPU suite. The remaining six skips belong to pre-existing optional tests.

The new checks cover independent column-major cuBLAS geometry, actual vendor
call argument interpretation on host memory, compiled Mojo contraction/indexing
against Fraction for every small committed corpus case and every mode, multiple
batch members, shape/mode/element/work limits, layout and Fraction conversion,
invalid input preservation, alias/concurrency rejection, poisoning after runtime
failure, numerical disagreement and build-receipt tampering.

The required hardware lane was separately attempted with `CUDA_REQUIRED=1`.
It **failed**, rather than skipping, at CUDA driver initialization before kernel
execution. This is an environment blocker, not a GPU correctness result.

Estate-layout, transcendental-firewall, Markdown-math and tensor-estate audits
passed; `git diff --check` passed. No mathematical or GPU evidence grade is
promoted by these checks. CI results, once available, belong to the PR checks.

## Experiments still unmeasured

1. Real-device kernel execution and Mojo/CuPy external-stream interoperability.
2. All-mode/all-member numerical agreement with Fraction and cuBLAS, including
   late-invalid-input preservation, reuse, overflow and concurrency on hardware.
3. Representative 16x24x8, 24x24x24, 48x48x48 and 96x16x16 runs for batches
   1, 8 and 32, retaining unfavorable results.
4. Actual device/FP64 capabilities, driver/runtime/cuBLAS versions, clocks and
   loaded-library provenance. None can be inferred from the `sm_80` build flag.
5. Device-resident and complete checked transfer-inclusive performance, with
   separate packing, raw allocation, enqueue, synchronization and transfer probes.
6. Profiler-isolated kernel durations. CUDA event intervals include possible
   host submission gaps and must not be relabeled as pure kernel time.

Use the required hardware tests before collecting performance evidence. The
benchmark aborts on missing hardware, stale source/binary receipts, a target
mismatch or numerical disagreement. It never substitutes a CPU implementation.

## Integration

This work is stacked on `perf/mode-gram-controlled`, preserving #14 → #15 → #17.
It does not merge those dependencies or place the tensor-frontier stack on main.
The PR is a draft pending review and hardware acceptance. The validated host
work is independently reviewable; CUDA correctness, throughput and speedup
remain open obligations.
