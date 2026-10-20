"""Checked homogeneous CPU batches with an explicitly reusable workspace.

Workspace.compute returns a borrowed output overwritten by the next call.
Concurrent use and input/output/scratch aliasing are rejected. Inputs must not
be concurrently modified while native code is running. No numerical fallback.
"""
from __future__ import annotations
import ctypes
from numbers import Integral, Real
from pathlib import Path
from threading import Lock

import numpy as np
from tensor_lab.mojo_mode_gram import LP64_LIMIT

DEFAULT_LIBRARY = Path(__file__).resolve().parents[1] / '.build/tensor_frontier/libmode_gram_batch.so'


def validate_shape(shape, mode):
    if isinstance(mode, (bool, np.bool_)) or not isinstance(mode, Integral) or mode not in (0, 1, 2):
        raise ValueError('mode must be an integer 0, 1, or 2')
    if len(shape) != 4 or any(isinstance(d, (bool, np.bool_)) or not isinstance(d, Integral) or d <= 0 for d in shape):
        raise ValueError('batch must have nonempty shape (batch, d0, d1, d2)')
    shape = tuple(int(d) for d in shape)
    count = 1
    for d in shape:
        count *= d  # Python integers cannot overflow.
    if count > LP64_LIMIT or shape[0] * shape[mode + 1] ** 2 > LP64_LIMIT:
        raise ValueError('batch exceeds the LP64 element limit')
    return shape


def prepare_batch(batch, mode):
    try:
        array = np.asarray(batch)
    except (ValueError, TypeError) as exc:
        raise ValueError('batch must be rectangular rank-4') from exc
    validate_shape(array.shape, mode)
    if array.dtype.kind not in 'iuf':
        if array.dtype.kind != 'O' or not all(isinstance(x, Real) and not isinstance(x, (bool, np.bool_)) for x in array.flat):
            raise ValueError('batch entries must be real numbers')
    return np.require(array, dtype=np.float64, requirements=['C', 'A'])


class MojoModeGramBatch:
    def __init__(self, library=DEFAULT_LIBRARY):
        self.library_path = Path(library).resolve()
        self.library = ctypes.CDLL(str(self.library_path))
        self.native = self.library.mode_gram_batch_f64
        self.native.argtypes = [ctypes.c_ssize_t] * 12
        self.native.restype = ctypes.c_int32

    def workspace(self, shape, mode):
        return BatchWorkspace(self, shape, mode)

    def __call__(self, batch, mode, *, fused=True):
        array = prepare_batch(batch, mode)
        workspace = self.workspace(array.shape, mode)
        return workspace.compute(array, fused=fused)


class BatchWorkspace:
    def __init__(self, kernel, shape, mode):
        self.shape = validate_shape(shape, mode)
        self.mode = int(mode)
        self.kernel = kernel
        self._output = np.empty((self.shape[0], self.shape[mode + 1], self.shape[mode + 1]), dtype=np.float64)
        self._scratch = np.empty(self.shape if mode == 1 else (0,), dtype=np.float64)
        self._lock = Lock()

    @property
    def workspace_bytes(self):
        return self._output.nbytes + self._scratch.nbytes

    def compute(self, batch, *, fused=True):
        if type(fused) is not bool:
            raise ValueError('fused must be a bool')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('workspace is already in use')
        try:
            array = prepare_batch(batch, self.mode)
            if array.shape != self.shape:
                raise ValueError('batch shape differs from workspace')
            if np.shares_memory(array, self._output) or np.shares_memory(array, self._scratch):
                raise ValueError('input must not alias workspace buffers')
            status = self.kernel.native(array.ctypes.data, *self.shape, self.mode,
                array.size, self._output.ctypes.data, self._output.size,
                self._scratch.ctypes.data if self.mode == 1 else 0,
                self._scratch.size, int(fused))
            if status == 4:
                raise ValueError('batch entries must be finite FP64 values')
            if status == 5:
                raise FloatingPointError('batch Gram output is nonfinite; discard output buffer')
            if status:
                raise ValueError(f'Mojo batch rejected input: status {status}')
            return self._output
        finally:
            self._lock.release()
