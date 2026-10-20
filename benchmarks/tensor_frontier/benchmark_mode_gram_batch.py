#!/usr/bin/env python3
"""Matched-backend CPU batch/fusion experiment; exact checks, E0 performance."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from benchmarks.tensor_frontier.benchmark_mode_gram import cases, residual, CORPUS, ROOT
from benchmarks.tensor_frontier.profile_mode_gram import Blas, numpy_openblas, symbol_path, sha, sample, THREAD_KEYS
from tensor_lab.mojo_mode_gram import MojoModeGram
from tensor_lab.mojo_mode_gram_batch import MojoModeGramBatch, prepare_batch
from tensor_lab.numpy_baseline import mode_gram_numpy
from tensor_lab.tensors import mode_gram


def checked_build(path):
    path = Path(path).resolve()
    build = json.loads(path.with_suffix('.build.json').read_text())
    if sha(path) != build['binary_sha256']:
        raise ValueError('binary hash mismatch')
    if not build.get('sources_sha256'):
        raise ValueError('missing dependency provenance; rebuild')
    for source, expected in build['sources_sha256'].items():
        if sha(ROOT / source) != expected:
            raise ValueError('source/dependency hash mismatch; rebuild')
    return build


def check_batch(references, actual):
    if actual.shape != (len(references), len(references[0]), len(references[0])):
        raise ValueError('batch result shape mismatch')
    errors = [residual(reference, value) for reference, value in zip(references, actual)]
    return max(x[0] for x in errors), max(x[1] for x in errors)


def numpy_checked(batch, mode, output):
    array = prepare_batch(batch, mode)
    if not np.isfinite(array).all():
        raise ValueError('nonfinite batch input')
    matrix = np.moveaxis(array, mode+1, 1).reshape(len(array), array.shape[mode+1], -1)
    np.matmul(matrix, matrix.swapaxes(-1,-2), out=output)
    if not np.isfinite(output).all():
        raise FloatingPointError('nonfinite batch output')
    return output


def benchmark(scalar_library, batch_library, blas, warmup=5, iterations=30, rounds=3,
              batch_sizes=(1,8,32), include_large=True):
    if warmup < 0 or iterations <= 0 or rounds <= 0 or not batch_sizes or any(type(b) is not int or b <= 0 for b in batch_sizes):
        raise ValueError('invalid benchmark counts')
    if any(os.environ.get(key) != '1' for key in THREAD_KEYS):
        raise ValueError('set all BLAS thread controls to 1 before launch: '+', '.join(THREAD_KEYS))
    scalar_build, batch_build = checked_build(scalar_library), checked_build(batch_library)
    numpy_provider = Blas(numpy_openblas(), 'numpy-ilp64')
    lp64 = Blas(blas)
    backend = Path(numpy_provider.metadata['path'])
    if symbol_path(getattr(lp64.lib, numpy_provider.gemm.__name__)) != backend:
        raise ValueError('benchmark requires the same loaded NumPy backend')
    bridge = json.loads(Path(lp64.metadata['path']).with_suffix('.build.json').read_text())
    if (bridge['binary_sha256'] != lp64.metadata['sha256'] or bridge['backend']['sha256'] != sha(backend)
            or bridge['source_sha256'] != sha(ROOT/'benchmarks/tensor_frontier/numpy_blas_bridge.c')):
        raise ValueError('bridge provenance mismatch')
    scalar, batched = MojoModeGram(scalar_library), MojoModeGramBatch(batch_library)
    for kernel in (scalar,batched):
        if symbol_path(kernel.library.cblas_dgemm) != Path(lp64.metadata['path']):
            raise ValueError('native library uses a different BLAS provider')
    corpus = cases()
    if include_large:
        shape=(48,48,48)
        data=((np.arange(np.prod(shape),dtype=np.int64)*17)%65-32)/8
        corpus.append({'id':'dyadic_48x48x48','tensor':data.reshape(shape).tolist()})
    records=[]
    for case in corpus:
        original=np.asarray(case['tensor'],dtype=np.float64)
        variants=[original+offset for offset in (0.,.125,-.25,.375)]
        for mode in range(3):
            print(f"Exact batch oracle: {case['id']} mode {mode}",flush=True)
            unique=[mode_gram(x.tolist(),mode) for x in variants]
            for batch_size in batch_sizes:
                batch=np.stack([variants[i%4] for i in range(batch_size)])
                references=[unique[i%4] for i in range(batch_size)]
                before=batch.copy()
                workspace=batched.workspace(batch.shape,mode)
                numpy_output=np.empty((batch_size,batch.shape[mode+1],batch.shape[mode+1]))
                lanes=[
                    ('mojo_scalar_loop',lambda:np.stack([scalar(x,mode) for x in batch]),'scalar checked API per item; stack/output allocations'),
                    ('numpy_scalar_loop',lambda:np.stack([mode_gram_numpy(x,mode) for x in batch]),'existing scalar baseline per item; lacks finite guards'),
                    ('mojo_batch_allocated',lambda:batched(batch,mode),'checked native batch; fused packing; workspace/output allocated per call'),
                    ('mojo_batch_reuse_unfused',lambda:workspace.compute(batch,fused=False),'checked native batch; separate scan/pack; reusable buffers'),
                    ('mojo_batch_reuse_fused',lambda:workspace.compute(batch,fused=True),'checked native batch; fused scan/pack; reusable buffers'),
                    ('numpy_batch_reuse_checked',lambda:numpy_checked(batch,mode,numpy_output),'vectorized NumPy matmul; input/output finite guards; reused output; mode-1 unfolding may allocate'),
                ]
                for repeat in range(rounds):
                    shift=repeat%len(lanes)
                    ordered=lanes[shift:]+lanes[:shift]
                    if repeat%2:ordered=ordered[::-1]
                    for order,(name,function,scope) in enumerate(ordered):
                        row=sample(function,lambda output:check_batch(references,output),warmup,iterations)
                        row.update(case=case['id'],shape=list(original.shape),batch_size=batch_size,mode=mode,
                            lane=name,scope=scope,round=repeat,order=order,warmup=warmup,iterations=iterations,
                            median_ns_per_tensor=row['median_ns']/batch_size,workspace_bytes=workspace.workspace_bytes,
                            performance_authority='E0',numerical_authority='E1')
                        records.append(row)
                if not np.array_equal(batch,before):raise ValueError('input mutated')
    if numpy_provider.thread_getter() != 1:raise ValueError('thread count changed')
    paths=['benchmarks/tensor_frontier/benchmark_mode_gram_batch.py',
        'benchmarks/tensor_frontier/benchmark_mode_gram.py','benchmarks/tensor_frontier/profile_mode_gram.py',
        'tensor_lab/mojo_mode_gram_batch.py','tensor_lab/mojo_mode_gram.py','tensor_lab/tensors.py',
        'tensor_lab/numpy_baseline.py','mojo/tensor_frontier/mode_gram_batch.mojo','mojo/tensor_frontier/mode_gram.mojo',
        'tools/build_mode_gram.py','tools/build_mode_gram_batch.py']
    return {'schema':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'implementation_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'working_tree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
        'authority':'E0 performance; at most E1 numerical replication; no theorem or GPU authority',
        'sources_sha256':{p:sha(ROOT/p) for p in paths},'corpus_sha256':sha(CORPUS),
        'scalar_build':scalar_build,'batch_build':batch_build,'bridge_build':bridge,
        'providers':{'lp64_adapter':lp64.metadata,'numpy_backend':numpy_provider.metadata},
        'shared_numpy_backend':True,'thread_environment':{k:os.environ.get(k) for k in THREAD_KEYS},
        'python_version':platform.python_version(),'numpy_version':np.__version__,'numpy_configuration':np.show_config(mode='dicts'),
        'platform':platform.platform(),'affinity':sorted(os.sched_getaffinity(0)),
        'cpu':next(s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),
        'cgroup_cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        'gpu_device_nodes':[str(p) for p in Path('/dev').glob('nvidia*')],
        'command_line':sys.argv,'reference':'unchanged exact Fraction mode_gram; every batch member checked outside timing',
        'batch_variants':'base + [0, 1/8, -1/4, 3/8], cycling by batch member',
        'comparison':'abs(actual-reference) <= 1e-12 + 1e-12*abs(reference)',
        'limits':['Sequential CPU native batching, not parallel/batched BLAS or GPU execution.',
                  'NumPy uses its own GEMM/SYRK dispatch; native paths all use GEMM.',
                  'Reused outputs are borrowed until the next call; workspace setup excluded only in reuse lanes.',
                  'Single host, warm buffers; no general speedup or hardware bandwidth claim.'],
        'records':records}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scalar-library',type=Path,required=True)
    parser.add_argument('--batch-library',type=Path,required=True)
    parser.add_argument('--blas',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--iterations',type=int,default=30)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--batch-sizes',type=int,nargs='+',default=[1,8,32])
    parser.add_argument('--no-large',action='store_true')
    args=parser.parse_args()
    if args.output.exists():parser.error('output exists')
    report=benchmark(args.scalar_library,args.batch_library,args.blas,args.warmup,args.iterations,args.rounds,
                     args.batch_sizes,not args.no_large)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report,handle,indent=2,allow_nan=False);handle.write('\n')
    print(f"Wrote {len(report['records'])} records to {args.output}")


if __name__=='__main__':main()
