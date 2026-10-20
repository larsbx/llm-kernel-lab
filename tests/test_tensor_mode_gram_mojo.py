"""Cross-language replication; numerical agreement has at most E1 authority."""
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from tensor_lab.mojo_mode_gram import MojoModeGram
from tensor_lab.numpy_baseline import mode_gram_allclose, mode_gram_numpy
from tensor_lab.tensors import mode_gram
from tools.build_mode_gram import build

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "benchmarks/tensor_frontier/mode_gram_cases.json").read_text())["cases"]


@pytest.fixture(scope="session")
def kernel(tmp_path_factory):
    if not shutil.which(os.environ.get("MOJO", "mojo")):
        if os.environ.get("MOJO_REQUIRED") == "1":
            pytest.fail("Mojo is mandatory for this conformance lane")
        pytest.skip("Mojo unavailable; use pixi run tensor-mode-gram-test to require it")
    return MojoModeGram(build(tmp_path_factory.mktemp("mojo") / "libmode_gram.so"))


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("mode", range(3))
def test_committed_corpus(kernel, case, mode):
    reference = mode_gram(case["tensor"], mode)
    actual = kernel(case["tensor"], mode)
    baseline = mode_gram_numpy(case["tensor"], mode)
    assert mode_gram_allclose(reference, actual)
    assert mode_gram_allclose(reference, baseline)
    assert mode_gram_allclose(baseline, actual)


@pytest.mark.parametrize("mode", range(3))
@pytest.mark.parametrize("layout", ["c", "fortran", "strided", "reversed", "swapped", "unaligned"])
def test_layouts_and_fractional_values(kernel, mode, layout):
    x = (np.arange(60, dtype=np.float64).reshape(3, 4, 5) - 23) / 8
    if layout == "fortran":
        x = np.asfortranarray(x)
    elif layout == "strided":
        x = x.transpose(2, 0, 1)
    elif layout == "reversed":
        x = x[::-1, :, ::-1]
    elif layout == "swapped":
        x = x.astype(">f8")
    elif layout == "unaligned":
        y = np.ndarray(x.shape, dtype=np.float64, buffer=bytearray(x.nbytes + 1), offset=1)
        y[...] = x
        x = y
    before = x.copy()
    reference = mode_gram(x.tolist(), mode)
    actual = kernel(x, mode)
    assert mode_gram_allclose(reference, actual)
    assert np.array_equal(x, before)
    assert actual.flags.c_contiguous and actual.dtype == np.float64


@pytest.mark.parametrize("mode", range(3))
def test_non_dyadic_rationals(kernel, mode):
    x = [[[Fraction((-1)**(i+j+k) * (i+j+k+1), 7) for k in range(5)]
          for j in range(3)] for i in range(2)]
    assert mode_gram_allclose(mode_gram(x, mode), kernel(x, mode))


@pytest.mark.parametrize("tensor", [[], [[]], [[[]]], [[1, 2]], [[[[1]]]],
    [[[1]], [[1, 2]]], [[[1j]]], [[["1"]]], [[[True]]],
    [[[float("nan")]]], [[[float("inf")]]], [[[float("-inf")]]]])
def test_invalid_tensor(kernel, tensor):
    with pytest.raises(ValueError):
        kernel(tensor, 0)


@pytest.mark.parametrize("mode", [-1, 3, 0.0, True, "0", None])
def test_invalid_mode(kernel, mode):
    with pytest.raises(ValueError):
        kernel([[[1]]], mode)


def test_overflow_rejected(kernel):
    with pytest.raises(FloatingPointError):
        kernel([[[1e308]]], 0)


@pytest.mark.parametrize("dims,mode,n,out_n,status", [
    ((1, 1, 1), 3, 1, 1, 1), ((0, 1, 1), 0, 0, 0, 2),
    ((-1, 1, 1), 0, 1, 1, 2), ((2**62, 2**62, 4), 0, 1, 1, 2),
    ((65536, 65536, 1), 0, 1, 1, 2), ((46341, 1, 1), 0, 46341, 1, 2),
    ((1, 1, 1), 0, 2, 1, 3), ((1, 1, 1), 0, 1, 0, 3),
])
def test_native_guards_before_memory_access(kernel, dims, mode, n, out_n, status):
    # Even null addresses must not be dereferenced for invalid descriptors.
    assert kernel.native(0, *dims, mode, n, 0, out_n) == status


def test_native_null_and_nonfinite_guards(kernel):
    assert kernel.native(0, 1, 1, 1, 0, 1, 0, 1) == 3
    x = np.array([float("nan")]); out = np.array([42.])
    assert kernel.native(x.ctypes.data, 1, 1, 1, 0, 1, out.ctypes.data, 1) == 4
    assert out[0] == 42  # Fail before writing any result.


def test_tolerance_is_entrywise_and_fail_closed():
    reference = ((0., 1.), (-1., 1e6))
    for row in range(2):
        for col in range(2):
            bound = 1e-12 + 1e-12 * abs(reference[row][col])
            within = np.array(reference); within[row, col] += bound * 0.5
            beyond = np.array(reference); beyond[row, col] += bound * 2
            assert mode_gram_allclose(reference, within)
            assert not mode_gram_allclose(reference, beyond)
    for invalid in [np.full((2, 2), np.nan), np.full((2, 2), np.inf), np.zeros((1, 2))]:
        assert not mode_gram_allclose(reference, invalid)


def test_shared_native_profile_helpers(kernel):
    import ctypes as ct
    scan = kernel.library.mode_gram_probe_scan
    scan.argtypes = [ct.c_ssize_t] * 2
    scan.restype = ct.c_int32
    pack = kernel.library.mode_gram_probe_pack
    pack.argtypes = [ct.c_ssize_t] * 5
    pack.restype = ct.c_int32
    noop = kernel.library.mode_gram_probe_noop
    noop.argtypes = [ct.c_ssize_t] * 8
    noop.restype = ct.c_int32
    assert noop(*([0] * 8)) == 0
    assert scan(0, 1) == 3
    assert scan(0, 2**40) == 3
    for invalid in [float('nan'), float('inf'), float('-inf')]:
        x = np.array([1., invalid, 2.])
        assert scan(x.ctypes.data, x.size) == 4
    x = (np.arange(60).reshape(3, 4, 5) - 23.) / 8
    packed = np.empty((4, 15), dtype=np.float64)
    assert scan(x.ctypes.data, x.size) == 0
    assert pack(x.ctypes.data, *x.shape, packed.ctypes.data) == 0
    assert np.array_equal(packed, np.moveaxis(x, 1, 0).reshape(4, 15))
    assert pack(0, 3, 4, 5, 0) == 3
    assert pack(0, 2**40, 4, 5, 0) == 2


def test_native_profile_end_to_end(kernel, monkeypatch):
    from benchmarks.tensor_frontier import profile_mode_gram as profile
    for key in profile.THREAD_KEYS[:3]:
        monkeypatch.setenv(key, '1')
    try:
        profile.numpy_openblas()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    monkeypatch.setattr(profile, 'cases', lambda: [{'id': 'native_probe', 'tensor': np.arange(24).reshape(2, 3, 4).tolist()}])
    report = profile.profile(kernel.library_path, os.environ.get('MODE_GRAM_BLAS', 'libblas.so.3'),
                             warmup=0, iterations=1, rounds=1, large=False)
    assert report['native_status'] == 'measured'
    assert len(report['records']) == 62
    assert all(r['max_absolute_residual'] == 0 for r in report['records'] if r['check_scope'].startswith('full Gram'))
