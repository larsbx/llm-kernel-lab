# Bounded Mojo CUDA mode-Gram prototype

Status: offline compiled and host checked; GPU execution and performance are
unvalidated. The exact `tensor_lab.tensors.mode_gram` Fraction oracle and the
committed `benchmarks/tensor_frontier/mode_gram_cases.json` remain unchanged.
Results cannot exceed E0 performance / E1 numerical replication. No theorem
authority, CUDA speedup, or transfer-cost estimate follows from the CPU work.

## Scope and input contract

`mode_gram_cuda.mojo` implements FP64 direct contraction for modes 0, 1 and 2.
Each GPU thread writes one complete output element and performs its reduction
in fixed index order. Mode indexing is fused into contraction, avoiding an
explicit packed device buffer. This intentionally simple bounded prototype is
an experimental comparator, not a production GEMM replacement or a claim that
custom multiplication beats a vendor primitive. No mixed precision is used.

The checked Python boundary is `tensor_lab.cuda_mode_gram.CudaWorkspace`.
It reuses the established homogeneous batch shape, conversion, finite-value,
LP64 count, borrowed-output, aliasing and concurrent-use rules. All input
validation finishes on the host before upload or output writes. A late invalid
member leaves the previous borrowed output unchanged. Nonfinite output raises
`FloatingPointError`; discard that output. Runtime failures poison the workspace,
which must be closed and recreated. Inputs must remain stable during execution.
Use the context manager to release the native session and device buffers.

An additional explicit bound limits each launch to `2**28` multiply-add terms:
`batch * d0 * d1 * d2 * shape[mode]`. The prototype rejects larger work before
allocating. Raising this bound requires a reviewed experiment; it is not a claim
about hardware capacity. A raw device-pointer C ABI exists only for the wrapper;
it requires valid, disjoint CUDA allocations on the selected device and stable
lifetimes. Descriptor checks cannot prove raw-pointer validity. No CPU fallback
exists. The host probe export is test-only and is never used by the CUDA API.

## Independent vendor comparison

The cuBLAS lane uses CuPy 13.6.0's direct `dgemmStridedBatched` binding, not its
matmul dispatch. Independent NumPy `moveaxis`/reshape packing produces
`(batch, rows, reduction)` row-major matrices. cuBLAS views these as
column-major `(reduction, rows)` and computes DGEMM `(T,N)` with leading
dimensions `reduction`, strides `rows*reduction`, and output stride `rows*rows`.
Mode 2 uses the original contiguous tensor as column-major `(rows, reduction)`
with `(N,T)` and leading dimension `rows`, so it needs no packing.
The full Gram is computed; no triangle is silently reconstructed. CPU tests
independently interpret those descriptors and compare every member to Fraction.

Host scalar pointer mode and pedantic FP64 math are set and read back. CUDA's
`CUBLAS_PEDANTIC_MATH` numeric enum is 2; CuPy 13.6 accepts it through
`setMathMode` but does not export a named constant. Atomics policy remains the
library default and is not represented as a verified setting. Both lanes use
one private nonblocking CUDA stream per workspace and run serially. Mojo wraps
the exact same stream handle used for its events and transfers. The compiled
kernel is retained in the session, outside steady-state timing.

## Timing boundaries

Every warmup and measured execution is checked outside timing against the
unchanged relation `abs(actual-reference) <= 1e-12 + 1e-12*abs(reference)`.
A disagreement aborts the benchmark and prevents publication of a successful
report. Raw individual samples are retained, including unfavorable timings.

| Field | Included work |
| --- | --- |
| `host_validation_ns` | Shape/conversion and finite input checks |
| `host_packing_ns` | Independent cuBLAS host unfolding; Mojo has no packing |
| `raw_device_allocation_ns`, `raw_device_free_ns` | Separate raw CUDA allocation/release probes for input/output sizes |
| `workspace_setup_ns` | Context/kernel/handle setup and reusable buffer allocation |
| `h2d_wall_ns`, `d2h_wall_ns` | Pageable host transfers, staging and completion wait |
| `host_enqueue_ns` | Python/native boundary and launch submission |
| `host_sync_wait_ns` | Host wait after recording the ending event |
| `device_stream_interval_ns` | CUDA event interval on the actual execution stream |
| `resident_launch_sync_ns` | Reused device buffers, launch and synchronization; no transfer or host packing |
| `checked_reuse_end_to_end_ns` | Validation, packing, upload, launch, synchronization, download and finite-output check |
| `checked_allocated_end_to_end_ns` | Fresh workspace plus checked execution, result copy and teardown |

The event interval can include GPU idle time while the host submits work.
It is **not** an isolated profiler kernel duration. Record a separate Nsight
Systems/Compute trace on hardware before making a kernel-only throughput claim.
Stage probes are separate runs and their medians must not be added together.
Tensor allocations bypass CuPy's memory pool; vendor internal allocations are
uncontrolled. Pageable transfers are intentional and recorded; pinned-memory
or overlapped transfer experiments need separately labeled lanes.

## Reproduce

The first validated compiler is Mojo 1.1.0 (`8189361e`) with MAX 26.6.0.
The existing repository Pixi nightly lock is unchanged and not represented as
validated for these GPU APIs. The separate CPU CI workflow pins the stable
prototype toolchain and requires offline GPU compilation plus host tests.

```bash
python -m pip install mojo==1.1.0 max==26.6.0 numpy==2.3.5 pytest==8.4.2
MOJO_CUDA_REQUIRED=1 python -m pytest -q tests/test_tensor_mode_gram_cuda.py -k 'not required_gpu'
python tools/build_mode_gram_cuda.py --target nvidia:sm_80
```

On a compatible NVIDIA host, install the CUDA-12 CuPy wheel and matching NVIDIA
runtime/cuBLAS libraries and driver. The wheel alone does not provide a GPU or
necessarily all toolkit libraries. Use the actual device architecture in the
build command; `sm_80` above is only the offline compilation target.

```bash
python -m pip install cupy-cuda12x==13.6.0
python -m benchmarks.tensor_frontier.benchmark_mode_gram_cuda --probe-only --output probe.json
# Rebuild for the probe's actual major/minor capability before these commands.
CUDA_REQUIRED=1 python -m pytest -q tests/test_tensor_mode_gram_cuda.py -k required_gpu
python -m benchmarks.tensor_frontier.benchmark_mode_gram_cuda --output cuda-raw.json --batch-sizes 1 8 32 --warmup 2 --iterations 10 --rounds 3
```

The required GPU lane fails on missing hardware, dependencies, stale binaries,
launch errors and differential failures. Ordinary CPU runs explicitly skip its
six hardware cases; those skips are not acceptance. The benchmark checks the
binary/source build receipt and exact target/device match. It records compiler,
driver/runtime/cuBLAS versions, device properties and FP64 ratio (when exposed),
`nvidia-smi` output, loaded library hashes, source/corpus hashes, revision and
dirty state. Hardware clock/power/sharing conditions must be reviewed from that
metadata before interpreting performance. No CPU measurements are substituted.

Default workloads include all committed corpus cases, deterministic 16x24x8,
24x24x24, 48x48x48 and 96x16x16 extensions, all modes, batches 1/8/32, and the
CPU experiment's four dyadic offsets. The exact oracle runs outside all timers.

## API sources consulted

- [Modular GPU example](https://github.com/modular/modular/blob/main/max/examples/gpu-functions/vector_addition.mojo)
- [Modular external-stream test](https://github.com/modular/modular/blob/main/max/kernels/test/gpu/device_context/test_external_stream.mojo)
- [DeviceContext API](https://max.modular.com/stable/api/mojo/max/gpu/host/device_context/DeviceContext/)
- [NVIDIA cuBLAS reference](https://docs.nvidia.com/cuda/cublas/index.html)
- [CuPy 13.6 cuBLAS binding](https://github.com/cupy/cupy/blob/v13.6.0/cupy_backends/cuda/libs/cublas.pyx)

Inspected on 2026-10-21. Compilation validates the used Mojo API against the
installed stable toolchain; it does not validate external-stream interoperation
or numerical execution on a device.
