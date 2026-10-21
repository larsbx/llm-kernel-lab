"""Opt-in bounded CUDA experiment. No numerical fallback or theorem authority.

CuPy supplies CUDA allocation/streams/events and direct cuBLAS bindings only.
Neither lane uses CuPy matmul. Device pointers stay internal to a workspace.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
from threading import Lock

import numpy as np

from tensor_lab.mojo_mode_gram_batch import prepare_batch, validate_shape

DEFAULT_LIBRARY = Path(__file__).resolve().parents[1] / '.build/tensor_frontier/libmode_gram_cuda.so'
MAX_TERMS = 2**28


def bounded_shape(shape, mode):
    shape = validate_shape(shape, mode)
    count = 1
    for d in shape:
        count *= d
    if count * shape[mode + 1] > MAX_TERMS:
        raise ValueError('prototype exceeds 2**28 multiply-add terms')
    return shape


def checked_input(batch, mode, shape=None):
    array = prepare_batch(batch, mode)
    bounded_shape(array.shape, mode)
    if shape is not None and array.shape != shape:
        raise ValueError('batch shape differs from workspace')
    if not np.isfinite(array).all():
        raise ValueError('batch entries must be finite FP64 values')
    return array


def pack_cublas(array, mode):
    """Independent NumPy unfolding, C-order (batch, rows, reduction).

    cuBLAS sees each row-major matrix as column-major (reduction, rows).
    DGEMM(T,N) produces its symmetric Gram; no triangle completion is used.
    """
    if mode == 2:
        return array  # C-order tensor is already column-major (rows, reduction).
    return np.ascontiguousarray(np.moveaxis(array, mode + 1, 1).reshape(
        len(array), array.shape[mode + 1], -1))


def cublas_geometry(shape, mode):
    shape = bounded_shape(shape, mode)
    batch, *dims = shape
    rows = dims[mode]
    width = int(np.prod(dims)) // rows
    return dict(m=rows, n=rows, k=width, lda=rows if mode == 2 else width,
                ldb=rows if mode == 2 else width, trans_a=0 if mode == 2 else 1,
                trans_b=1 if mode == 2 else 0,
                ldc=rows, stride_a=rows*width, stride_b=rows*width,
                stride_c=rows*rows, batch=batch)


class NativeMojo:
    def __init__(self, library=DEFAULT_LIBRARY):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.create = self.library.mode_gram_cuda_create
        self.create.argtypes = [ctypes.c_ssize_t] * 2
        self.create.restype = ctypes.c_ssize_t
        self.launch = self.library.mode_gram_cuda_launch
        self.launch.argtypes = [ctypes.c_ssize_t] * 8
        self.launch.restype = ctypes.c_int32
        self.destroy = self.library.mode_gram_cuda_destroy
        self.destroy.argtypes = [ctypes.c_ssize_t]
        self.destroy.restype = None


class CudaWorkspace:
    """One private nonblocking stream, reusable buffers, synchronous checked API.

    compute() returns a borrowed host result. Late bad input leaves it unchanged.
    A runtime failure poisons the workspace; close it and create another.
    Internal _upload/_launch/_download support isolated benchmark stages only.
    Callers must not mutate inputs or reuse borrowed results concurrently.
    """
    def __init__(self, shape, mode, lane, library=DEFAULT_LIBRARY, device=0):
        self.shape = bounded_shape(shape, mode)
        if lane not in ('mojo', 'cublas'):
            raise ValueError('lane must be mojo or cublas')
        if type(device) is not int or device < 0:
            raise ValueError('device must be a nonnegative integer')
        import cupy as cp
        if cp.cuda.runtime.is_hip:
            raise RuntimeError('this comparison requires NVIDIA CUDA')
        self.cp, self.device = cp, device
        self.mode, self.lane = int(mode), lane
        self._lock = Lock()
        self._closed, self._poisoned, self._uploaded = False, False, False
        self._session = 0
        self._handle = None
        self._native = None
        self.geometry = cublas_geometry(self.shape, self.mode)
        self._alpha, self._beta = np.array(1., dtype=np.float64), np.array(0., dtype=np.float64)
        rows = self.shape[mode+1]
        self._output = np.empty((self.shape[0], rows, rows), dtype=np.float64)
        try:
            with cp.cuda.Device(device):
                self.stream = cp.cuda.Stream(non_blocking=True)
                # Bypass CuPy's pool: report actual device allocation costs.
                with self.stream, cp.cuda.using_allocator(None):
                    self._x = cp.empty(int(np.prod(self.shape)), dtype=cp.float64)
                    self._y = cp.empty(self._output.shape, dtype=cp.float64)
                if lane == 'mojo':
                    self._native = NativeMojo(library)
                    self._session = self._native.create(self.stream.ptr, device)
                    if not self._session:
                        raise RuntimeError('Mojo CUDA context/stream initialization failed')
                else:
                    cb = cp.cuda.cublas
                    self._handle = cb.create()
                    cb.setPointerMode(self._handle, cb.CUBLAS_POINTER_MODE_HOST)
                    # CUDA cublasMath_t value 2; CuPy 13.6 accepts the value but
                    # exports only DEFAULT/TENSOR_OP constants. No reduced precision.
                    cb.setMathMode(self._handle, 2)
                    if cb.getMathMode(self._handle) != 2 or cb.getPointerMode(self._handle) != 0:
                        raise RuntimeError('cuBLAS configuration was not applied')
                self.stream.synchronize()
        except Exception:
            self.close()
            raise

    def _ready(self):
        if self._closed or self._poisoned:
            raise RuntimeError('workspace closed or poisoned')

    def _upload(self, array):
        self._ready()
        packed = array if self.lane == 'mojo' else pack_cublas(array, self.mode)
        self._x.set(packed.reshape(-1), stream=self.stream)
        # Pageable host input: include staging and synchronization in transfer cost.
        self.stream.synchronize()
        self._uploaded = True

    def _launch(self):
        self._ready()
        if not self._uploaded:
            raise RuntimeError('upload a checked input before launch')
        with self.cp.cuda.Device(self.device), self.stream:
            if self.lane == 'mojo':
                status = self._native.launch(self._session, self._x.data.ptr,
                    self._y.data.ptr, *self.shape, self.mode)
                if status:
                    raise RuntimeError(f'Mojo CUDA launch rejected: status {status}')
            else:
                g = self.geometry
                cb = self.cp.cuda.cublas
                cb.dgemmStridedBatched(self._handle, g['trans_a'], g['trans_b'],
                    g['m'], g['n'], g['k'], self._alpha.ctypes.data,
                    self._x.data.ptr, g['lda'], g['stride_a'],
                    self._x.data.ptr, g['ldb'], g['stride_b'], self._beta.ctypes.data,
                    self._y.data.ptr, g['ldc'], g['stride_c'], g['batch'])

    def _download(self):
        self._ready()
        self._y.get(stream=self.stream, out=self._output, blocking=True)
        if not np.isfinite(self._output).all():
            raise FloatingPointError('nonfinite Gram output; discard borrowed result')
        return self._output

    def compute(self, batch):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('workspace is already in use')
        try:
            self._ready()
            array = checked_input(batch, self.mode, self.shape)
            if np.shares_memory(array, self._output):
                raise ValueError('input must not alias workspace output')
            try:
                with self.cp.cuda.Device(self.device):
                    self._upload(array)
                    self._launch()
                    self.stream.synchronize()
                    return self._download()
            except Exception:
                self._poisoned = True
                raise
        finally:
            self._lock.release()

    def close(self):
        if self._closed:
            return
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('workspace is already in use')
        try:
            with self.cp.cuda.Device(self.device):
                try:
                    if hasattr(self, 'stream'):
                        self.stream.synchronize()
                finally:
                    if self._session:
                        self._native.destroy(self._session)
                        self._session = 0
                    if self._handle is not None:
                        self.cp.cuda.cublas.destroy(self._handle)
                        self._handle = None
                    self._x = self._y = None
                    self._closed = True
        finally:
            self._lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
