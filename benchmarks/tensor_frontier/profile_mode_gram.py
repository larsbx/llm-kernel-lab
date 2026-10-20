#!/usr/bin/env python3
"""Controlled CPU probes; timings are E0 and are not an additive cost model.

No Mojo fallback: --library enables native lanes, omission records them unavailable.
Run in a fresh process with BLAS thread environment fixed before NumPy imports.
"""
from __future__ import annotations

import argparse
import ctypes as ct
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import subprocess
import time

import numpy as np

from benchmarks.tensor_frontier.benchmark_mode_gram import CORPUS, ROOT, cases, residual
from tensor_lab.mojo_mode_gram import MojoModeGram, prepare_tensor
from tensor_lab.numpy_baseline import mode_gram_numpy
from tensor_lab.tensors import mode_gram

THREAD_KEYS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
               'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def symbol_path(function):
    """Linux dladdr: identify the object actually exporting a resolved symbol."""
    class Info(ct.Structure):
        _fields_ = [('name', ct.c_char_p), ('base', ct.c_void_p),
                    ('symbol', ct.c_char_p), ('address', ct.c_void_p)]
    dladdr = ct.CDLL(None).dladdr
    dladdr.argtypes = [ct.c_void_p, ct.POINTER(Info)]
    dladdr.restype = ct.c_int
    info = Info()
    if not dladdr(ct.cast(function, ct.c_void_p), ct.byref(info)):
        raise RuntimeError('cannot resolve BLAS provider')
    return Path(os.fsdecode(info.name)).resolve()


def numpy_openblas():
    # Inspect loaded mappings, not the build-time directory in np.show_config().
    paths = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
             if 'numpy.libs/' in line and 'openblas' in line}
    if len(paths) != 1:
        raise RuntimeError('expected one loaded NumPy OpenBLAS; provide a supported environment')
    return next(iter(paths))


class Blas:
    """Benchmark-only CBLAS adapter with explicit LP64/ILP64 symbol contracts."""
    def __init__(self, path, abi='lp64'):
        self.lib = ct.CDLL(str(path))
        if abi == 'lp64':
            prefix, suffix, integer = '', '', ct.c_int32
        elif abi == 'numpy-ilp64':
            prefix, suffix, integer = 'scipy_', '64_', ct.c_int64
        else:
            raise ValueError('unsupported BLAS ABI')
        self.gemm = getattr(self.lib, prefix + 'cblas_dgemm' + suffix)
        self.syrk = getattr(self.lib, prefix + 'cblas_dsyrk' + suffix)
        # CBLAS enums remain C int even when dimensions use ILP64.
        self.gemm.argtypes = [ct.c_int] * 3 + [integer] * 3 + [ct.c_double,
            ct.c_void_p, integer, ct.c_void_p, integer, ct.c_double, ct.c_void_p, integer]
        self.syrk.argtypes = [ct.c_int] * 3 + [integer] * 2 + [ct.c_double,
            ct.c_void_p, integer, ct.c_double, ct.c_void_p, integer]
        self.gemm.restype = self.syrk.restype = None
        path = symbol_path(self.gemm)
        self.metadata = {'path': str(path), 'sha256': sha(path), 'abi': abi,
                         'gemm_symbol': self.gemm.__name__, 'syrk_symbol': self.syrk.__name__,
                         'dependencies': subprocess.check_output(['ldd', str(path)], text=True)}
        self.thread_getter = None
        for p, s in [(prefix, suffix), ('', '')]:
            try:
                config = getattr(self.lib, p + 'openblas_get_config' + s)
                core = getattr(self.lib, p + 'openblas_get_corename' + s)
                getter = getattr(self.lib, p + 'openblas_get_num_threads' + s)
                setter = getattr(self.lib, p + 'openblas_set_num_threads' + s)
            except AttributeError:
                continue
            config.restype = core.restype = ct.c_char_p
            getter.restype = ct.c_int
            setter.argtypes = [ct.c_int]; setter.restype = None
            setter(1)
            if getter() != 1:
                raise RuntimeError('failed to enforce single-thread OpenBLAS')
            self.thread_getter = getter
            self.metadata.update(configuration=config().decode(), architecture=core().decode(),
                                 actual_threads=getter(), thread_control='OpenBLAS runtime API')
            break
        if self.thread_getter is None:
            self.metadata.update(actual_threads=None,
                thread_control='unavailable; environment is recorded, not proof of thread count')

    def calls(self, matrix, out, transposed=False):
        # matrix is C contiguous; mode 2 uses original (width, rows) storage.
        rows = out.shape[0]
        width = matrix.size // rows
        leading = rows if transposed else width
        trans = 112 if transposed else 111
        ptr, dst = matrix.ctypes.data, out.ctypes.data
        gemm_args = (101, trans, 111 if transposed else 112, rows, rows, width,
                     1., ptr, leading, ptr, leading, 0., dst, rows)
        syrk_args = (101, 121, trans, rows, width, 1., ptr, leading, 0., dst, rows)
        # Zero M/N preserves valid leading dimensions; includes BLAS quick-return
        # dispatch as well as ctypes marshaling, NOT a pure Mojo FFI measurement.
        empty_args = (101, 111, 112, 0, 0, width, 1., ptr, width,
                      ptr, width, 0., dst, 1)
        def gemm():
            self.gemm(*gemm_args)
            return out
        def syrk():
            self.syrk(*syrk_args)
            return out
        return gemm, syrk, lambda: self.gemm(*empty_args)


def sample(function, check, warmup, iterations):
    for _ in range(warmup):
        check(function())
    times, errors = [], []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        value = function()
        times.append(time.perf_counter_ns() - start)
        errors.append(check(value))  # strictly outside timing
    return {'samples_ns': times, 'median_ns': statistics.median(times),
            'max_absolute_residual': max((e[0] for e in errors if e is not None), default=None),
            'max_tolerance_ratio': max((e[1] for e in errors if e is not None), default=None)}


def profile(library=None, blas='libblas.so.3', warmup=5, iterations=30, rounds=3,
            large=True, numpy_blas=None, numpy_abi='numpy-ilp64'):
    if warmup < 0 or iterations <= 0 or rounds <= 0:
        raise ValueError('invalid warmup/iterations/rounds')
    if any(os.environ.get(k) != '1' for k in THREAD_KEYS[:3]):
        raise ValueError('set OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 before launch')
    numpy_path = Path(numpy_blas or numpy_openblas()).resolve()
    # An explicit override must still be a dependency of NumPy's loaded module.
    dependencies = subprocess.check_output(
        ['ldd', np._core._multiarray_umath.__file__], text=True)
    linked_paths = {Path(token).resolve() for token in dependencies.split()
                    if token.startswith('/') and Path(token).exists()}
    if numpy_path not in linked_paths:
        raise ValueError('--numpy-blas is not a resolved NumPy dependency')
    lp64, bundled = Blas(blas), Blas(numpy_path, numpy_abi)
    shared_backend = False
    bridge_build = None
    try:
        linked_backend = getattr(lp64.lib, bundled.gemm.__name__)
        shared_backend = symbol_path(linked_backend) == Path(bundled.metadata['path'])
    except AttributeError:
        pass
    if shared_backend and lp64.metadata['path'] != bundled.metadata['path']:
        metadata_path = Path(lp64.metadata['path']).with_suffix('.build.json')
        bridge_build = json.loads(metadata_path.read_text())
        if (bridge_build['binary_sha256'] != lp64.metadata['sha256'] or
                bridge_build['backend']['sha256'] != bundled.metadata['sha256'] or
                bridge_build['source_sha256'] != sha(ROOT / 'benchmarks/tensor_frontier/numpy_blas_bridge.c')):
            raise ValueError('bridge provenance mismatch; rebuild')
    kernel, build = None, None
    if library:
        library = Path(library).resolve()
        build = json.loads(library.with_suffix('.build.json').read_text())
        if sha(library) != build['binary_sha256'] or sha(ROOT / 'mojo/tensor_frontier/mode_gram.mojo') != build['source_sha256']:
            raise ValueError('native build hash mismatch; rebuild')
        kernel = MojoModeGram(library)
        for symbol, arity in [('mode_gram_probe_noop', 8), ('mode_gram_probe_scan', 2),
                              ('mode_gram_probe_pack', 5)]:
            function = getattr(kernel.library, symbol)
            function.argtypes = [ct.c_ssize_t] * arity
            function.restype = ct.c_int32
        if symbol_path(kernel.library.cblas_dgemm) != Path(lp64.metadata['path']):
            raise ValueError('--blas must match the Mojo library resolved CBLAS provider')
    corpus = cases()
    if large:
        for shape in [(48, 48, 48), (96, 16, 16)]:
            values = ((np.arange(np.prod(shape), dtype=np.int64) * 17) % 65 - 32) / 8
            corpus.append({'id': 'dyadic_' + 'x'.join(map(str, shape)),
                           'tensor': values.reshape(shape).tolist()})
    records = []
    for case in corpus:
        tensor = np.asarray(case['tensor'], dtype=np.float64)
        tensor_copy_source = np.asfortranarray(tensor)
        before = hashlib.sha256(tensor.tobytes()).hexdigest()
        for mode in range(3):
            print(f"Exact oracle: {case['id']} mode {mode}", flush=True)
            reference = mode_gram(case['tensor'], mode)
            expected = np.asarray(reference, dtype=np.float64)
            rows = tensor.shape[mode]
            moved = np.moveaxis(tensor, mode, 0)
            unfolding = moved.reshape(rows, -1)
            out = np.empty((rows, rows), dtype=np.float64)
            matrix = (np.ascontiguousarray(unfolding) if mode == 1 else tensor)
            upper = np.triu_indices(rows)
            right_copy = unfolding.T.copy(order='K')
            check = lambda value: residual(reference, value)
            def check_upper(value):
                # DSYRK defines only the upper triangle; lower is not evidence.
                return residual(expected[upper].reshape(1, -1), value[upper].reshape(1, -1))
            def check_prepare(value):
                if value.dtype != np.float64 or not value.flags.c_contiguous or not value.flags.aligned or not np.array_equal(value, tensor):
                    raise ValueError('prepare/copy probe mismatch')
            def check_alloc(value):
                if value.shape != (rows, rows) or value.dtype != np.float64:
                    raise ValueError('allocation probe mismatch')
            def check_unfold(value):
                if not np.array_equal(value, unfolding):
                    raise ValueError('unfold probe mismatch')
            def check_true(value):
                if not value:
                    raise ValueError('finite scan failed')
            lanes = [
                ('numpy_public', lambda: mode_gram_numpy(tensor, mode), check, 'full Gram'),
                ('python_prepare_resident', lambda: prepare_tensor(tensor, mode), check_prepare, 'input representation'),
                ('python_prepare_fortran', lambda: prepare_tensor(tensor_copy_source, mode), check_prepare, 'input representation'),
                ('output_allocate', lambda: np.empty((rows, rows), dtype=np.float64), check_alloc, 'shape/dtype only; no page touch'),
                ('numpy_unfold', lambda: np.moveaxis(tensor, mode, 0).reshape(rows, -1), check_unfold, 'unfolding values'),
                ('numpy_input_finite_scan', lambda: np.isfinite(tensor).all(), check_true, 'NumPy scan proxy; not Mojo scalar validation'),
                ('numpy_preallocated_gram', lambda: np.matmul(unfolding, unfolding.T, out=out), check, 'full Gram'),
                ('numpy_preallocated_distinct_gemm', lambda: np.matmul(unfolding, right_copy, out=out), check, 'full Gram; distinct right operand disables shared-pointer SYRK'),
                ('numpy_output_finite_scan', lambda: np.isfinite(expected).all(), check_true, 'NumPy scan proxy; not Mojo scalar validation'),
            ]
            if mode == 1:
                packed = np.empty(moved.shape, dtype=np.float64)
                def packing():
                    np.copyto(packed, moved)
                    return packed.reshape(rows, -1)
                lanes.append(('numpy_pack_mode1_preallocated', packing, check_unfold, 'packing proxy; not Mojo loop'))
            for name, provider in [('lp64', lp64), ('numpy_blas', bundled)]:
                gemm, syrk, quick = provider.calls(matrix, out, mode == 2)
                lanes.extend([(name + '_gemm', gemm, check, 'full Gram; preallocated/prepacked'),
                              (name + '_syrk_upper', syrk, check_upper, 'upper triangle only; no mirroring'),
                              (name + '_ctypes_quick_return', quick, lambda _: None, 'no numerical output; ctypes plus BLAS dispatch')])
            if kernel:
                args = (tensor.ctypes.data, *tensor.shape, mode, tensor.size, out.ctypes.data, out.size)
                def native():
                    status = kernel.native(*args)
                    if status:
                        raise ValueError(f'native status {status}')
                    return out
                def check_status(status):
                    if status:
                        raise ValueError(f'native probe status {status}')
                input_args = (tensor.ctypes.data, tensor.size)
                output_args = (expected.ctypes.data, expected.size)
                lanes.extend([
                    ('mojo_ffi_noop', lambda: kernel.library.mode_gram_probe_noop(*args), check_status, 'Python ctypes to Mojo empty function; no numerical output'),
                    ('mojo_input_finite_scan', lambda: kernel.library.mode_gram_probe_scan(*input_args), check_status, 'shared native finite scan plus ctypes'),
                    ('mojo_output_finite_scan', lambda: kernel.library.mode_gram_probe_scan(*output_args), check_status, 'shared native finite scan plus ctypes')])
                if mode == 1:
                    native_packed = np.empty(moved.shape, dtype=np.float64)
                    pack_args = (tensor.ctypes.data, *tensor.shape, native_packed.ctypes.data)
                    def native_pack():
                        check_status(kernel.library.mode_gram_probe_pack(*pack_args))
                        return native_packed.reshape(rows, -1)
                    lanes.append(('mojo_pack_mode1_preallocated', native_pack, check_unfold, 'shared native packing plus ctypes; excludes allocation'))
                lanes.extend([('mojo_public', lambda: kernel(tensor, mode), check, 'full Gram'),
                              ('mojo_native_preallocated', native, check, 'full Gram; includes native guards, packing, FFI and GEMM')])
            for repeat in range(rounds):
                # Deterministic rotation, reversed on odd rounds; record order.
                shift = repeat % len(lanes)
                ordered = lanes[shift:] + lanes[:shift]
                if repeat % 2:
                    ordered = ordered[::-1]
                for order, (name, function, checker, scope) in enumerate(ordered):
                    record = sample(function, checker, warmup, iterations)
                    record.update(case=case['id'], shape=list(tensor.shape), mode=mode,
                        lane=name, round=repeat, order=order, check_scope=scope,
                        warmup=warmup, iterations=iterations, input_bytes=tensor.nbytes,
                        output_bytes=out.nbytes, gemm_flops=2 * rows * tensor.size,
                        performance_authority='E0')
                    records.append(record)
            if hashlib.sha256(tensor.tobytes()).hexdigest() != before:
                raise ValueError('input mutated')
    for provider in (lp64, bundled):
        if provider.thread_getter and provider.thread_getter() != 1:
            raise ValueError('BLAS thread count changed during run')
    sources = ['benchmarks/tensor_frontier/profile_mode_gram.py',
               'benchmarks/tensor_frontier/benchmark_mode_gram.py',
               'tensor_lab/numpy_baseline.py', 'tensor_lab/mojo_mode_gram.py',
               'tensor_lab/tensors.py', 'mojo/tensor_frontier/mode_gram.mojo',
               'benchmarks/tensor_frontier/numpy_blas_bridge.c',
               'tools/build_mode_gram_numpy_bridge.py']
    return {'schema': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'authority': 'E0 performance; at most E1 numerical observations; no theorem authority',
        'implementation_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'working_tree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
        'sources_sha256': {p: sha(ROOT / p) for p in sources}, 'corpus_sha256': sha(CORPUS),
        'command_line': sys.argv, 'numpy_dependencies': dependencies,
        'numpy_version': np.__version__, 'numpy_configuration': np.show_config(mode='dicts'),
        'python_version': platform.python_version(), 'platform': platform.platform(),
        'cpu': next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')), 'unknown'),
        'affinity': sorted(os.sched_getaffinity(0)),
        'cgroup_cpu_max': Path('/sys/fs/cgroup/cpu.max').read_text().strip() if Path('/sys/fs/cgroup/cpu.max').exists() else None,
        'thread_environment': {k: os.environ.get(k) for k in THREAD_KEYS},
        'providers': {'lp64': lp64.metadata, 'numpy_blas': bundled.metadata},
        'native_status': 'measured' if kernel else 'not requested; no Mojo timings generated',
        'native_build': build, 'bridge_build': bridge_build,
        'shared_numpy_backend': shared_backend, 'reference': 'tensor_lab.tensors.mode_gram; exact Fraction arithmetic for every case/mode',
        'comparison': 'abs(actual-reference) <= 1e-12 + 1e-12*abs(reference)',
        'limitations': ['Isolated warm-cache probes overlap and MUST NOT be summed or subtracted as an exact decomposition.',
            'NumPy scan/packing probes are proxies; mojo_* probes use shared native helpers plus ctypes.',
            'Quick return measures ctypes plus BLAS dispatch, not pure FFI or Mojo FFI.',
            'Provider comparison changes ABI and library implementation; not a language comparison.',
            'DSYRK upper timings omit triangle mirroring and cannot be presented as full Gram latency.'],
        'records': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--library', type=Path)
    parser.add_argument('--blas', default='libblas.so.3', help='LP64 CBLAS provider; must match --library')
    parser.add_argument('--numpy-blas', help='Explicit NumPy-linked CBLAS library on non-wheel installations')
    parser.add_argument('--numpy-abi', choices=['lp64', 'numpy-ilp64'], default='numpy-ilp64')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--no-large', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; choose a new evidence path')
    result = profile(args.library, args.blas, args.warmup, args.iterations, args.rounds, not args.no_large, args.numpy_blas, args.numpy_abi)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(f"Wrote {len(result['records'])} records to {args.output}")


if __name__ == '__main__':
    main()
