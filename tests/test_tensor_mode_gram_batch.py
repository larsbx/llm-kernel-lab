"""Batch conformance, failure atomicity, and reusable-buffer ownership."""
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from tensor_lab.mojo_mode_gram_batch import MojoModeGramBatch, validate_shape
from tensor_lab.numpy_baseline import mode_gram_allclose, mode_gram_numpy
from tensor_lab.tensors import mode_gram
from tools.build_mode_gram_batch import build

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / 'benchmarks/tensor_frontier/mode_gram_cases.json').read_text())['cases']


@pytest.fixture(scope='module')
def kernel(tmp_path_factory):
    if not shutil.which(os.environ.get('MOJO', 'mojo')):
        if os.environ.get('MOJO_REQUIRED') == '1':
            pytest.fail('Mojo required')
        pytest.skip('Mojo unavailable')
    return MojoModeGramBatch(build(tmp_path_factory.mktemp('batch') / 'batch.so'))


@pytest.mark.parametrize('case', CASES, ids=lambda c:c['id'])
@pytest.mark.parametrize('mode', range(3))
@pytest.mark.parametrize('fused', [False, True])
def test_corpus_batch(kernel, case, mode, fused):
    x = np.asarray(case['tensor'], dtype=np.float64)
    batch = np.stack([x, x + .125, x * -.5])
    before = batch.copy()
    actual = kernel(batch, mode, fused=fused)
    for i, item in enumerate(batch):
        assert mode_gram_allclose(mode_gram(item.tolist(), mode), actual[i])
        assert mode_gram_allclose(mode_gram_numpy(item, mode), actual[i])
    assert np.array_equal(before, batch)


@pytest.mark.parametrize('layout', ['fortran','reverse','swapped','unaligned','fraction'])
@pytest.mark.parametrize('mode', range(3))
def test_layouts(kernel, layout, mode):
    batch = (np.arange(120).reshape(2,3,4,5) - 41.) / 8
    if layout == 'fortran': batch = np.asfortranarray(batch)
    elif layout == 'reverse': batch = batch[::-1,:,:,::-1]
    elif layout == 'swapped': batch = batch.astype('>f8')
    elif layout == 'unaligned':
        b = np.ndarray(batch.shape, dtype=np.float64, buffer=bytearray(batch.nbytes+1), offset=1)
        b[:] = batch; batch = b
    elif layout == 'fraction':
        batch = np.array([Fraction(int(n),7) for n in range(120)], dtype=object).reshape(2,3,4,5)
    actual = kernel(batch,mode)
    for i,item in enumerate(batch):
        assert mode_gram_allclose(mode_gram(item.tolist(),mode),actual[i])


@pytest.mark.parametrize('fused', [False,True])
@pytest.mark.parametrize('mode',range(3))
def test_invalid_last_item_leaves_entire_output_untouched(kernel,fused,mode):
    batch = np.ones((3,2,3,4)); batch[-1,-1,-1,-1] = np.nan
    ws = kernel.workspace(batch.shape,mode)
    ws._output.fill(123.)
    with pytest.raises(ValueError,match='finite'):
        ws.compute(batch,fused=fused)
    assert np.all(ws._output == 123.)
    batch[-1,-1,-1,-1] = 2.
    assert np.isfinite(ws.compute(batch,fused=fused)).all() # scratch reusable after failure


def test_output_overflow_rejected(kernel):
    with pytest.raises(FloatingPointError):
        kernel(np.full((2,1,1,1),1e308),0)


def test_reuse_aliasing_and_concurrency(kernel):
    batch = np.arange(8.).reshape(2,2,2,1)
    ws = kernel.workspace(batch.shape,0)
    output = ws.compute(batch)
    address = output.ctypes.data
    assert ws.compute(batch+1).ctypes.data == address
    with pytest.raises(ValueError,match='alias'):
        ws.compute(output.reshape(batch.shape))
    ws._lock.acquire()
    try:
        with pytest.raises(RuntimeError,match='in use'): ws.compute(batch)
    finally: ws._lock.release()
    with pytest.raises(ValueError,match='shape'): ws.compute(np.ones((2,3,2,1)))
    with pytest.raises(ValueError,match='bool'): ws.compute(batch,fused=1)
    assert ws.compute(batch).ctypes.data == address


@pytest.mark.parametrize('bad',[[],[[[1.]]],np.ones((0,2,3,4)),[[[[True]]]],[[[['1']]]],[[[[1j]]]]])
def test_invalid_inputs(kernel,bad):
    with pytest.raises(ValueError): kernel(bad,0)


@pytest.mark.parametrize('mode',[-1,3,False,'1',1.0])
def test_invalid_mode(kernel,mode):
    with pytest.raises(ValueError): kernel(np.ones((1,2,3,4)),mode)


@pytest.mark.parametrize('shape',[(0,1,1,1),(1,2**40,1,1),(2,46340,1,1),(2**40,2**40,2**40,2**40)])
def test_size_guards_without_allocation(shape):
    with pytest.raises(ValueError):validate_shape(shape,0)


@pytest.mark.parametrize('args,status',[
    ((0,1,1,1,1,3,1,0,1,0,0,1),1),
    ((0,1,1,1,1,0,1,0,1,0,0,2),6),
    ((0,2**62,2,2,2,0,0,0,0,0,0,1),2),
    ((0,2,65536,65536,1,0,0,0,0,0,0,1),2),
    ((0,1,1,1,1,0,2,0,1,0,0,1),3),
    ((0,1,1,1,1,1,1,0,1,0,1,1),3),
])
def test_native_descriptor_guards(kernel,args,status):
    assert kernel.native(*args)==status
