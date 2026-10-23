#!/usr/bin/env python3
"""Fail-closed Mojo/cuBLAS FP64 comparison; no GPU means no measurements."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from benchmarks.tensor_frontier.benchmark_mode_gram import cases, CORPUS, ROOT
from benchmarks.tensor_frontier.benchmark_mode_gram_batch import check_batch
from tensor_lab.cuda_mode_gram import CudaWorkspace, checked_input, pack_cublas, DEFAULT_LIBRARY
from tensor_lab.tensors import mode_gram
from tools.build_mode_gram_cuda import checked_build, sha


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': str(exc)}


def probe(device=0):
    result = dict(platform=platform.platform(), python=platform.python_version(),
        numpy=np.__version__, device_nodes=[str(p) for p in Path('/dev').glob('nvidia*')],
        nvidia_smi=command(['nvidia-smi','-q','-x']), nvcc=command(['nvcc','--version']),
        mojo=command([os.environ.get('MOJO','mojo'),'--version']),
        gpu_environment={key:os.environ.get(key) for key in (
            'CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','CUDA_MODULE_LOADING',
            'CUBLAS_WORKSPACE_CONFIG','CUBLAS_EMULATION_STRATEGY','NVIDIA_TF32_OVERRIDE')},
        gpu_available=False, measurements=[])
    try:
        import cupy as cp
        result['cupy'] = cp.__version__
        result['device_count'] = cp.cuda.runtime.getDeviceCount()
        if cp.cuda.runtime.is_hip or device >= result['device_count']:
            raise RuntimeError('selected NVIDIA CUDA device is unavailable')
        with cp.cuda.Device(device):
            result['device_id'] = device
            result['properties'] = cp.cuda.runtime.getDeviceProperties(device)
            result['driver_version'] = cp.cuda.runtime.driverGetVersion()
            result['cuda_runtime_version'] = cp.cuda.runtime.runtimeGetVersion()
            # CUDA reports the device's architectural single/double precision ratio.
            result['fp64_ratio'] = cp.cuda.Device(device).attributes.get('SingleToDoublePrecisionPerfRatio')
        result['gpu_available'] = True
    except Exception as exc:
        result['unavailability'] = f'{type(exc).__name__}: {exc}'
    return result


def source_receipt():
    paths = ['tensor_lab/cuda_mode_gram.py', 'tensor_lab/tensors.py',
        'tensor_lab/numpy_baseline.py', 'tensor_lab/mojo_mode_gram_batch.py',
        'tensor_lab/mojo_mode_gram.py',
        'benchmarks/tensor_frontier/benchmark_mode_gram_cuda.py',
        'benchmarks/tensor_frontier/benchmark_mode_gram.py',
        'benchmarks/tensor_frontier/benchmark_mode_gram_batch.py']
    return {p: sha(ROOT/p) for p in paths}


def loaded_libraries():
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        path = line.split()[-1]
        if path.startswith('/') and any(x in path.lower() for x in ('cublas', 'cuda', 'modular', 'max', 'nvrtc')):
            if Path(path).is_file():
                paths.add(path)
    return {p: sha(p) for p in sorted(paths)}


def timed(function):
    start = time.perf_counter_ns()
    value = function()
    return value, time.perf_counter_ns() - start


def check(references, output):
    if not np.isfinite(output).all():
        raise FloatingPointError('nonfinite result; refusing performance evidence')
    absolute, ratio = check_batch(references, output)
    return dict(max_residual=absolute, max_tolerance_ratio=ratio)


def sample(workspace, batch, references):
    """Isolated probes are NOT additive. Every execution is checked outside timing."""
    w, cp = workspace, workspace.cp
    row = {}
    pointers = []
    try:
        def allocate():
            pointers.append(cp.cuda.runtime.malloc(w._x.nbytes))
            pointers.append(cp.cuda.runtime.malloc(w._y.nbytes))
        _, row['raw_device_allocation_ns'] = timed(allocate)
    finally:
        begin = time.perf_counter_ns()
        for pointer in pointers:
            cp.cuda.runtime.free(pointer)
        row['raw_device_free_ns'] = time.perf_counter_ns() - begin
    array, row['host_validation_ns'] = timed(lambda: checked_input(batch, w.mode, w.shape))
    packed, row['host_packing_ns'] = timed(lambda: array if w.lane == 'mojo' else pack_cublas(array, w.mode))
    def upload():
        w._x.set(packed.reshape(-1), stream=w.stream)
        w.stream.synchronize()
        w._uploaded = True
    _, row['h2d_wall_ns'] = timed(upload)
    # Events are on the EXACT stream passed to Mojo and used by cuBLAS.
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record(w.stream)
    _, row['host_enqueue_ns'] = timed(w._launch)
    end.record(w.stream)
    _, row['host_sync_wait_ns'] = timed(w.stream.synchronize)
    row['device_stream_interval_ns'] = cp.cuda.get_elapsed_time(start, end) * 1e6
    # Includes host dispatch gaps between event records; not a profiler's kernel duration.
    def download():
        w._y.get(stream=w.stream, out=w._output, blocking=True)
        return w._output
    output, row['d2h_wall_ns'] = timed(download)
    row.update(check(references, output))
    def absorb(result):
        evidence = check(references, result)
        for key, value in evidence.items():
            row[key] = max(row[key], value)
    def resident():
        w._launch()
        w.stream.synchronize()
    _, row['resident_launch_sync_ns'] = timed(resident)
    absorb(w._download())
    output, row['checked_reuse_end_to_end_ns'] = timed(lambda: w.compute(batch))
    absorb(output)
    return row


def benchmark(library, *, device=0, batch_sizes=(1,8,32), warmup=2, iterations=10, rounds=3, large=True):
    if (type(device) is not int or device < 0 or type(warmup) is not int or warmup < 0
            or type(iterations) is not int or iterations <= 0 or type(rounds) is not int or rounds <= 0
            or not batch_sizes or any(type(b) is not int or b <= 0 for b in batch_sizes)):
        raise ValueError('invalid benchmark counts/device')
    build = checked_build(library)
    environment = probe(device)
    if not environment['gpu_available']:
        raise RuntimeError('GPU preflight failed: '+environment.get('unavailability','unknown reason'))
    import cupy as cp
    # Require a build for the actual device, not an unrecorded compatibility guess.
    props = environment['properties']
    target = f"nvidia:sm_{props['major']}{props['minor']}"
    if build['target'] != target:
        raise ValueError(f'rebuild for selected device: --target {target}')
    corpus = cases()
    if large:
        for shape in ((48,48,48), (96,16,16)):
            data = ((np.arange(np.prod(shape),dtype=np.int64)*17)%65-32)/8
            corpus.append(dict(id='dyadic_'+'x'.join(map(str,shape)),tensor=data.reshape(shape).tolist()))
    records = []
    for case in corpus:
        original = np.asarray(case['tensor'],dtype=np.float64)
        variants = [original+offset for offset in (0.,.125,-.25,.375)]
        for mode in range(3):
            print(f"Exact oracle: {case['id']} mode {mode}", flush=True)
            exact = [mode_gram(x.tolist(),mode) for x in variants]
            for batch_size in batch_sizes:
                batch = np.stack([variants[i%4] for i in range(batch_size)])
                references = [exact[i%4] for i in range(batch_size)]
                before = batch.copy()
                for repeat in range(rounds):
                    for lane in (('mojo','cublas') if repeat%2 == 0 else ('cublas','mojo')):
                        with cp.cuda.Device(device):
                            w, setup_ns = timed(lambda: CudaWorkspace(batch.shape,mode,lane,library,device))
                            with w:
                                for _ in range(warmup):
                                    check(references,w.compute(batch))
                                for index in range(iterations):
                                    row = sample(w,batch,references)
                                    # Separate allocated API includes setup and teardown; reusable API does not.
                                    def allocated():
                                        with CudaWorkspace(batch.shape,mode,lane,library,device) as fresh:
                                            result = fresh.compute(batch).copy()
                                        return result
                                    result, row['checked_allocated_end_to_end_ns'] = timed(allocated)
                                    for key, value in check(references,result).items():
                                        row[key] = max(row[key], value)
                                    row.update(case=case['id'], shape=list(original.shape), mode=mode,
                                        batch_size=batch_size, lane=lane, round=repeat, sample=index,
                                        workspace_setup_ns=setup_ns, input_sha256=hashlib.sha256(batch.tobytes()).hexdigest())
                                    records.append(row)
                                if lane == 'cublas':
                                    environment['cublas_version'] = cp.cuda.cublas.getVersion(w._handle)
                    if not np.array_equal(batch,before):
                        raise ValueError('input mutated')
    return dict(schema=1,status='measured',created_at_utc=datetime.now(timezone.utc).isoformat(),
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
        environment=environment,build=build,sources_sha256=source_receipt(),corpus_sha256=sha(CORPUS),
        loaded_libraries_sha256=loaded_libraries(),command_line=sys.argv,
        settings=dict(batch_sizes=batch_sizes,warmup=warmup,iterations=iterations,rounds=rounds,
            streams='one private nonblocking stream per workspace; serial lanes',
            cublas='DGEMM strided batched; pedantic math (verified); library-default atomics mode; host scalar pointers (verified)',
            transfer='pageable NumPy host buffers; staging/synchronization included',
            allocation='CuPy pool bypassed for tensor buffers; library internal allocations uncontrolled'),
        reference='unchanged exact Fraction mode_gram; every warmup and measured execution checked',
        relation='abs(actual-reference) <= 1e-12 + 1e-12*abs(reference)',
        authority='E0 performance; at most E1 numerical replication; no theorem authority',
        limits=['No cross-hardware generalization.',
            'Device event interval includes host dispatch gaps; pure kernel duration requires a GPU profiler.',
            'Stage probes are separate executions and are not additive.',
            'Workspace setup includes context/handle setup; raw cudaMalloc/cudaFree probes are separately recorded.',
            'Mojo fuses mode indexing into direct contraction; cuBLAS packing is on host. Compare complete costs as well as resident costs.'],
        records=records)


def json_default(value):
    if isinstance(value,bytes):
        return value.decode(errors='replace')
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,default=DEFAULT_LIBRARY)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--probe-only',action='store_true')
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--batch-sizes',type=int,nargs='+',default=[1,8,32])
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--iterations',type=int,default=10)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--no-large',action='store_true')
    a=p.parse_args()
    if a.output.exists():
        p.error('output exists; choose a new evidence path')
    if a.probe_only:
        report=dict(schema=1,status='probe_only',created_at_utc=datetime.now(timezone.utc).isoformat(),
            environment=probe(a.device),sources_sha256=source_receipt(),corpus_sha256=sha(CORPUS),
            revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
            records=[])
    else:
        report=benchmark(a.library,device=a.device,batch_sizes=a.batch_sizes,
            warmup=a.warmup,iterations=a.iterations,rounds=a.rounds,large=not a.no_large)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False,default=json_default)
        f.write('\n')


if __name__=='__main__':
    main()
