"""Fail-closed numerical/provenance checks for the controlled CPU probes."""
import ctypes
import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.tensor_frontier import profile_mode_gram as p
from tensor_lab.tensors import mode_gram


@pytest.fixture(scope='module', params=['lp64', 'numpy-ilp64', 'numpy-bridge'])
def blas(request):
    try:
        if request.param == 'numpy-bridge':
            from tools.build_mode_gram_numpy_bridge import build
            return p.Blas(build())
        return p.Blas('libblas.so.3' if request.param == 'lp64' else p.numpy_openblas(), request.param)
    except (OSError, AttributeError, RuntimeError) as exc:
        pytest.skip(f'provider unavailable on this platform: {exc}')


@pytest.mark.parametrize('mode', range(3))
def test_provider_layouts_and_upper_triangle(blas, mode):
    tensor = (np.arange(60).reshape(3, 4, 5) - 23.) / 8
    unfolding = np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)
    matrix = np.ascontiguousarray(unfolding) if mode == 1 else tensor
    out = np.full((tensor.shape[mode],) * 2, np.nan)
    reference = mode_gram(tensor.tolist(), mode)
    gemm, syrk, quick = blas.calls(matrix, out, mode == 2)
    p.residual(reference, gemm())
    out.fill(np.nan)
    result = syrk()
    upper = np.triu_indices(len(out))
    p.residual(np.asarray(reference)[upper].reshape(1, -1), result[upper].reshape(1, -1))
    assert np.isnan(result[np.tril_indices(len(out), -1)]).all()
    before = out.copy()
    quick()
    assert np.array_equal(out, before, equal_nan=True)
    assert p.symbol_path(blas.gemm).is_file()


@pytest.mark.parametrize('bad', [1.1, float('nan'), float('inf')])
def test_every_sample_is_checked(bad):
    calls = iter([np.array([[1.]]), np.array([[bad]])])
    with pytest.raises(ValueError, match='tolerance'):
        p.sample(lambda: next(calls), lambda x: p.residual([[1]], x), 1, 1)


@pytest.mark.parametrize('warmup,iterations,rounds', [(-1, 1, 1), (0, 0, 1), (0, 1, 0)])
def test_counts_fail_before_loading(warmup, iterations, rounds):
    with pytest.raises(ValueError, match='invalid'):
        p.profile(warmup=warmup, iterations=iterations, rounds=rounds)


def test_thread_environment_required(monkeypatch):
    monkeypatch.delenv('OPENBLAS_NUM_THREADS', raising=False)
    with pytest.raises(ValueError, match='before launch'):
        p.profile()


def test_smoke_report_and_hash_guard(monkeypatch, tmp_path):
    for key in p.THREAD_KEYS:
        monkeypatch.setenv(key, '1')
    try:
        p.numpy_openblas()
        ctypes.CDLL('libblas.so.3')
    except (RuntimeError, OSError) as exc:
        pytest.skip(str(exc))
    monkeypatch.setattr(p, 'cases', lambda: [{'id': 'probe', 'tensor': np.arange(24).reshape(2, 3, 4).tolist()}])
    report = p.profile(warmup=0, iterations=1, rounds=1, large=False)
    assert len(report['records']) == 46
    assert report['native_status'].startswith('not requested')
    assert report['providers']['numpy_blas']['actual_threads'] == 1
    assert all(r['performance_authority'] == 'E0' for r in report['records'])
    assert all(r['max_absolute_residual'] == 0 for r in report['records'] if r['check_scope'].startswith('full Gram'))
    assert not any(r['lane'].startswith('mojo') for r in report['records'])
    json.dumps(report, allow_nan=False)
    binary = tmp_path / 'invalid.so'
    binary.write_bytes(b'not a shared library')
    binary.with_suffix('.build.json').write_text(json.dumps({'binary_sha256': 'wrong'}))
    with pytest.raises(ValueError, match='hash mismatch'):
        p.profile(library=binary, large=False)


@pytest.mark.parametrize('key', p.THREAD_KEYS)
@pytest.mark.parametrize('value', [None, '2', '0'])
def test_all_provider_thread_limits_required(monkeypatch, key, value):
    for name in p.THREAD_KEYS:
        monkeypatch.setenv(name, '1')
    if value is None:
        monkeypatch.delenv(key)
    else:
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(p, 'numpy_openblas', lambda: pytest.fail('loaded provider before guard'))
    with pytest.raises(ValueError, match=key):
        p.profile()


@pytest.fixture
def fake_blas(monkeypatch, tmp_path):
    """Exercise metadata without depending on the host's BLAS implementation."""
    def gemm():
        pass
    def syrk():
        pass
    library = SimpleNamespace(cblas_dgemm=gemm, cblas_dsyrk=syrk)
    binary = tmp_path / 'libfake.so'
    binary.write_bytes(b'fake BLAS for metadata tests')
    monkeypatch.setattr(p.ct, 'CDLL', lambda _: library)
    monkeypatch.setattr(p, 'symbol_path', lambda _: binary)
    monkeypatch.setattr(p.subprocess, 'check_output', lambda *a, **k: 'fake dependencies')
    return library


def test_environment_limits_do_not_claim_verified_runtime_threads(fake_blas, monkeypatch):
    for key in p.THREAD_KEYS:
        monkeypatch.setenv(key, '1')
    provider = p.Blas('libfake.so')
    assert provider.thread_getter is None
    assert provider.metadata['actual_threads'] is None
    assert provider.metadata['thread_control'] == (
        'unavailable; environment is recorded, not proof of thread count')


@pytest.mark.parametrize('enforced', [True, False])
def test_openblas_metadata_requires_runtime_enforcement(fake_blas, enforced):
    threads = 8
    requested = []
    def setter(value):
        nonlocal threads
        requested.append(value)
        if enforced:
            threads = value
    def getter():
        return threads
    fake_blas.openblas_set_num_threads = setter
    fake_blas.openblas_get_num_threads = getter
    fake_blas.openblas_get_config = lambda: b'fake OpenBLAS'
    fake_blas.openblas_get_corename = lambda: b'fake CPU'
    if enforced:
        provider = p.Blas('libfake.so')
        assert provider.thread_getter() == 1
        assert provider.metadata['actual_threads'] == 1
        assert provider.metadata['thread_control'] == 'OpenBLAS runtime API'
    else:
        with pytest.raises(RuntimeError, match='failed to enforce single-thread OpenBLAS'):
            p.Blas('libfake.so')
    assert requested == [1]
