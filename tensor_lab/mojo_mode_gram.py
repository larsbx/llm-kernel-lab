"""Checked Python boundary to the compiled Mojo/CBLAS CPU experiment.

No computation fallback: a missing or incompatible native library is an error.
The exact oracle and NumPy baseline remain independent implementations.
"""
from __future__ import annotations

import ctypes
from numbers import Integral, Real
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY = ROOT / ".build" / "tensor_frontier" / "libmode_gram.so"
LP64_LIMIT = 2**31 - 1


def prepare_tensor(tensor, mode: int) -> np.ndarray:
    if isinstance(mode, (bool, np.bool_)) or not isinstance(mode, Integral) or mode not in (0, 1, 2):
        raise ValueError("mode must be an integer 0, 1, or 2")
    try:
        array = np.asarray(tensor)
    except (ValueError, TypeError) as exc:
        raise ValueError("tensor must be rectangular rank-3") from exc
    if array.ndim != 3 or any(d == 0 for d in array.shape):
        raise ValueError("tensor must be nonempty rectangular rank-3")
    if array.dtype.kind not in "iuf":
        if array.dtype.kind != "O" or not all(
            isinstance(x, Real) and not isinstance(x, (bool, np.bool_)) for x in array.flat
        ):
            raise ValueError("tensor entries must be real numbers")
    if array.size > LP64_LIMIT or array.shape[mode] ** 2 > LP64_LIMIT:
        raise ValueError("tensor exceeds the LP64 element limit")
    # Native code checks finiteness after conversion, including overflow to inf.
    # require() also handles unaligned, byte-swapped, Fortran and strided input.
    return np.require(array, dtype=np.float64, requirements=["C", "A"])


class MojoModeGram:
    def __init__(self, library: str | Path = DEFAULT_LIBRARY):
        self.library_path = Path(library).resolve()
        self.library = ctypes.CDLL(str(self.library_path))
        self.native = self.library.mode_gram_f64
        # Linux-64 C ABI: eight intptr_t values, including raw buffer addresses.
        self.native.argtypes = [ctypes.c_ssize_t] * 8
        self.native.restype = ctypes.c_int32

    def __call__(self, tensor, mode: int) -> np.ndarray:
        array = prepare_tensor(tensor, mode)
        result = np.empty((array.shape[mode],) * 2, dtype=np.float64)
        status = self.native(
            array.ctypes.data, *array.shape, int(mode), array.size,
            result.ctypes.data, result.size,
        )
        if status == 4:
            raise ValueError("tensor entries must be finite FP64 values")
        if status == 5:
            raise FloatingPointError("mode Gram output is nonfinite (FP64 overflow)")
        if status:
            raise ValueError(f"Mojo mode_gram rejected input: status {status}")
        return result
