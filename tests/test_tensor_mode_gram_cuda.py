"""CPU contracts, compiled host helper, and explicitly required CUDA execution."""
import ctypes
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
from contextlib import nullcontext
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.tensor_frontier.benchmark_mode_gram import cases
from benchmarks.tensor_frontier.benchmark_mode_gram_cuda import check, benchmark
from tensor_lab.cuda_mode_gram import (bounded_shape, checked_input, pack_cublas,
    cublas_geometry, CudaWorkspace, NativeMojo)
from tensor_lab.tensors import mode_gram
from tools.build_mode_gram_cuda import build, checked_build, DEFAULT_OUTPUT


def batches():
    for case in cases(False):
        original=np.asarray(case['tensor'],dtype=np.float64)
        yield np.stack([original,original+.125,original-.25])


@pytest.mark.parametrize('mode',range(3))
def test_independent_column_major_geometry(mode):
    # Interpret BLAS dimensions/strides literally as column-major; do not call
    # either accelerated lane or share the Mojo indexing formula.
    for batch in batches():
        g=cublas_geometry(batch.shape,mode)
        packed=pack_cublas(batch,mode).ravel()
        for b in range(len(batch)):
            block=packed[b*g['stride_a']:(b+1)*g['stride_a']]
            matrix=block.reshape((g['lda'],-1),order='F')
            left=matrix.T if g['trans_a'] else matrix
            right=matrix.T if g['trans_b'] else matrix
            actual=left @ right
            check([mode_gram(batch[b].tolist(),mode)],actual[None])


@pytest.mark.parametrize('mode',[True,False,np.bool_(True),-1,3,1.0,'1'])
def test_mode_rejected(mode):
    with pytest.raises(ValueError):
        checked_input(np.ones((1,2,3,4)),mode)


@pytest.mark.parametrize('value',[True,1+2j,'1',float('nan'),float('inf'),-float('inf')])
def test_invalid_entries(value):
    with pytest.raises(ValueError):
        checked_input([[[[value]]]],0)


@pytest.mark.parametrize('shape',[(0,2,2,2),(1,0,2,2),(1,2,2),(True,2,2,2),
    (2**31,1,1,1),(1,2**31,2**31,1),(33,128,128,128)])
def test_size_limits_without_allocating(shape):
    with pytest.raises(ValueError):
        bounded_shape(shape,0)


def test_conversion_and_shapes():
    source=np.arange(48.).reshape(2,2,3,4)
    for value in (source[...,::-1],source.astype('>f8'),np.asfortranarray(source)):
        converted=checked_input(value,1)
        assert converted.flags.c_contiguous and converted.flags.aligned
        np.testing.assert_array_equal(converted,value)
    assert checked_input([[[[Fraction(1,3)]]]],0)[0,0,0,0] == float(Fraction(1,3))
    with pytest.raises(ValueError):
        checked_input(source,1,(1,2,3,4))
    with pytest.raises(ValueError):
        checked_input([[[[1],[1,2]]]],0)


def test_disagreement_nonfinite_and_truncation_fail_closed():
    for actual in (np.array([[[2.]]]),np.array([[[np.nan]]]),np.ones((2,1,1))):
        with pytest.raises((ValueError,FloatingPointError)):
            check([[[Fraction(1)]]],actual)


@pytest.fixture(scope='module')
def compiled(tmp_path_factory):
    compiler=os.environ.get('MOJO','mojo')
    if not shutil.which(compiler):
        if os.environ.get('MOJO_CUDA_REQUIRED')=='1':
            pytest.fail('MOJO_CUDA_REQUIRED=1 but compiler unavailable')
        pytest.skip('offline Mojo compiler not installed')
    output=tmp_path_factory.mktemp('cuda-build')/'libmode_gram_cuda.so'
    build(output,compiler)
    checked_build(output)
    return output


@pytest.mark.parametrize('mode',range(3))
def test_compiled_host_contraction_against_exact(compiled,mode):
    lib=ctypes.CDLL(str(compiled))
    fn=lib.mode_gram_cuda_host_probe
    fn.argtypes=[ctypes.c_ssize_t]*7
    fn.restype=None
    for batch in batches():
        rows=batch.shape[mode+1]
        output=np.full((len(batch),rows,rows),np.nan)
        fn(batch.ctypes.data,output.ctypes.data,*batch.shape[1:],mode,output.size)
        check([mode_gram(x.tolist(),mode) for x in batch],output)


def test_native_descriptors_without_gpu(compiled):
    native=NativeMojo(compiled)
    assert native.create(0,0)==0
    assert native.launch(0,1,2,1,2,3,4,3)==1
    assert native.launch(0,1,2,1,0,3,4,0)==2
    assert native.launch(0,1,2,1,2**31,3,4,0)==2
    assert native.launch(0,1,2,33,128,128,128,0)==2
    assert native.launch(0,1,2,1,2,3,4,0)==3
    native.destroy(0)


def test_provenance_tamper_rejected(compiled,tmp_path):
    target=tmp_path/'tampered.so'
    shutil.copyfile(compiled,target)
    shutil.copyfile(compiled.with_suffix('.build.json'),target.with_suffix('.build.json'))
    target.write_bytes(target.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='binary hash'):
        checked_build(target)


def test_invalid_benchmark_counts_rejected_before_gpu():
    with pytest.raises(ValueError):
        benchmark('missing.so',iterations=0)


def host_workspace():
    # Exercise the actual checked boundary without pretending to execute CUDA.
    w=object.__new__(CudaWorkspace)
    w.shape,w.mode,w.device=(1,1,1,1),0,0
    w._closed=w._poisoned=False
    w._lock=Lock()
    w._output=np.array([[[9.]]])
    w.cp=SimpleNamespace(cuda=SimpleNamespace(Device=lambda _:nullcontext()))
    w.stream=SimpleNamespace(synchronize=lambda:None)
    return w


def test_host_guards_precede_any_upload():
    w=host_workspace()
    def forbidden(*_):
        pytest.fail('bad input reached device upload')
    w._upload=forbidden
    for bad in (np.full(w.shape,np.nan),np.ones((2,1,1,1)),w._output.reshape(w.shape)):
        with pytest.raises(ValueError):
            w.compute(bad)
        assert w._output.item()==9
        assert not w._poisoned
    with w._lock:
        with pytest.raises(RuntimeError,match='already in use'):
            w.compute(np.ones(w.shape))


def test_runtime_failure_poisons_and_releases_lock():
    w=host_workspace()
    w._upload=lambda _:None
    def fail():
        raise RuntimeError('asynchronous CUDA failure')
    w._launch=fail
    with pytest.raises(RuntimeError,match='asynchronous'):
        w.compute(np.ones(w.shape))
    assert w._poisoned and not w._lock.locked()
    with pytest.raises(RuntimeError,match='poisoned'):
        w.compute(np.ones(w.shape))


@pytest.mark.parametrize('mode',range(3))
def test_actual_vendor_call_arguments_with_host_interpreter(mode):
    batch=next(batches())
    # Use a nonscalar rectangular batch to detect bad leading dimensions/strides.
    batch=np.arange(48.,dtype=np.float64).reshape(2,2,3,4)/8-.5
    packed=pack_cublas(batch,mode).ravel()
    g=cublas_geometry(batch.shape,mode)
    output=np.full((len(batch),g['m'],g['m']),np.nan)
    def dgemm(handle,ta,tb,m,n,k,alpha,a,lda,sa,b,ldb,sb,beta,c,ldc,sc,nb):
        assert (handle,ldc,sc,nb)==(123,m,m*m,len(batch))
        assert ctypes.c_double.from_address(alpha).value==1
        assert ctypes.c_double.from_address(beta).value==0
        assert a==b==packed.ctypes.data and c==output.ctypes.data
        assert ta!=tb
        for index in range(nb):
            aa=np.ctypeslib.as_array((ctypes.c_double*(m*k)).from_address(a+8*index*sa)).reshape((lda,-1),order='F')
            bb=np.ctypeslib.as_array((ctypes.c_double*(n*k)).from_address(b+8*index*sb)).reshape((ldb,-1),order='F')
            output[index]=(aa.T if ta else aa) @ (bb.T if tb else bb)
    w=host_workspace()
    w._uploaded=True
    w.lane='cublas'
    w.geometry=g
    w._handle=123
    w._alpha,w._beta=np.array(1.),np.array(0.)
    w._x=SimpleNamespace(data=SimpleNamespace(ptr=packed.ctypes.data))
    w._y=SimpleNamespace(data=SimpleNamespace(ptr=output.ctypes.data))
    w.stream=nullcontext()
    w.cp.cuda.cublas=SimpleNamespace(dgemmStridedBatched=dgemm)
    w._launch()
    check([mode_gram(x.tolist(),mode) for x in batch],output)


@pytest.fixture(scope='module')
def gpu_library():
    if os.environ.get('CUDA_REQUIRED')!='1':
        pytest.skip('opt-in hardware lane; set CUDA_REQUIRED=1 to REQUIRE it')
    # No importorskip or catch: missing driver/library/hardware is a failure.
    import cupy as cp
    assert not cp.cuda.runtime.is_hip
    assert cp.cuda.runtime.getDeviceCount()>0
    library=Path(os.environ.get('MODE_GRAM_CUDA_LIBRARY',str(DEFAULT_OUTPUT)))
    checked_build(library)
    return library


@pytest.mark.parametrize('lane',['mojo','cublas'])
@pytest.mark.parametrize('mode',range(3))
def test_required_gpu_corpus_and_failures(gpu_library,lane,mode):
    for batch in batches():
        refs=[mode_gram(x.tolist(),mode) for x in batch]
        with CudaWorkspace(batch.shape,mode,lane,gpu_library) as w:
            result=w.compute(batch)
            check(refs,result)
            saved=result.copy()
            bad=batch.copy()
            bad[-1,-1,-1,-1]=np.nan
            with pytest.raises(ValueError):
                w.compute(bad)
            np.testing.assert_array_equal(result,saved)
            for view in (batch[...,::-1],batch.astype('>f8'),np.asfortranarray(batch)):
                check([mode_gram(x.tolist(),mode) for x in view],w.compute(view))
            with w._lock:
                with pytest.raises(RuntimeError,match='already in use'):
                    w.compute(batch)
            with pytest.raises(FloatingPointError):
                w.compute(np.full(batch.shape,1e308))
            with pytest.raises(RuntimeError,match='poisoned'):
                w.compute(batch)
        with pytest.raises(RuntimeError,match='closed'):
            w.compute(batch)
