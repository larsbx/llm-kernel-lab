"""Conventional floating-point baselines for tensor-frontier kernels."""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

import numpy as np

MODE_GRAM_FP64_ATOL = 1e-12
MODE_GRAM_FP64_RTOL = 1e-12


def mode_gram_numpy(
    tensor,
    mode: int,
    *,
    dtype=np.float64,
) -> np.ndarray:
    """Compute a mode Gram matrix with NumPy/BLAS.

    This is a conventional inexact baseline, not a mathematical authority.
    """
    if mode not in (0, 1, 2):
        raise ValueError("mode must be 0, 1, or 2")

    try:
        array = np.asarray(tensor, dtype=dtype)
    except ValueError as exc:
        raise ValueError("tensor must be rectangular rank-3") from exc

    if array.ndim != 3:
        raise ValueError("tensor must have rank 3")
    if any(dimension == 0 for dimension in array.shape):
        raise ValueError("tensor dimensions must be non-empty")

    unfolding = np.moveaxis(array, mode, 0).reshape(array.shape[mode], -1)
    return unfolding @ unfolding.T


def mode_gram_allclose(
    reference: Sequence[Sequence[int | Fraction]],
    observed,
    *,
    atol: float = MODE_GRAM_FP64_ATOL,
    rtol: float = MODE_GRAM_FP64_RTOL,
) -> bool:
    """Check the declared FP64 comparison relation against an exact matrix."""
    expected = np.asarray(
        [[float(value) for value in row] for row in reference],
        dtype=np.float64,
    )
    actual = np.asarray(observed, dtype=np.float64)
    if actual.shape != expected.shape:
        return False
    return bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
