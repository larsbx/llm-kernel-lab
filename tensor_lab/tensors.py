"""Exact tensor reference predicates for accelerator differential tests.

The first implementation is deliberately narrow: dense rank-3 tensors over
integers/rationals. It is an oracle for bounded corpora, not a performance
implementation.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

ScalarInput = int | Fraction
Rank3Tensor = Sequence[Sequence[Sequence[ScalarInput]]]
Matrix = tuple[tuple[Fraction, ...], ...]


def _q(value: ScalarInput) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def _rank3_shape(tensor: Rank3Tensor) -> tuple[int, int, int]:
    """Validate a non-empty rectangular rank-3 tensor and return its shape."""
    d0 = len(tensor)
    if d0 == 0:
        raise ValueError("tensor dimensions must be non-empty")

    d1 = len(tensor[0])
    if d1 == 0:
        raise ValueError("tensor dimensions must be non-empty")

    d2 = len(tensor[0][0])
    if d2 == 0:
        raise ValueError("tensor dimensions must be non-empty")

    for slab in tensor:
        if len(slab) != d1:
            raise ValueError("tensor must be rectangular")
        for row in slab:
            if len(row) != d2:
                raise ValueError("tensor must be rectangular")

    return d0, d1, d2


def mode_gram(tensor: Rank3Tensor, mode: int) -> Matrix:
    """Return the exact Gram matrix of a mode unfolding.

    For a rank-3 tensor T with shape (d0, d1, d2), this computes

        G_mode = T_(mode) T_(mode)^T

    over exact rational arithmetic. The result has shape
    d_mode x d_mode. This real/rational oracle is intentionally smaller than
    the eventual complex/Hermitian accelerator surface.
    """
    if mode not in (0, 1, 2):
        raise ValueError("mode must be 0, 1, or 2")

    d0, d1, d2 = _rank3_shape(tensor)
    dims = (d0, d1, d2)
    size = dims[mode]
    gram = [[Fraction(0) for _ in range(size)] for _ in range(size)]

    for left in range(size):
        for right in range(left, size):
            total = Fraction(0)

            if mode == 0:
                for j in range(d1):
                    for k in range(d2):
                        total += _q(tensor[left][j][k]) * _q(tensor[right][j][k])
            elif mode == 1:
                for i in range(d0):
                    for k in range(d2):
                        total += _q(tensor[i][left][k]) * _q(tensor[i][right][k])
            else:
                for i in range(d0):
                    for j in range(d1):
                        total += _q(tensor[i][j][left]) * _q(tensor[i][j][right])

            gram[left][right] = total
            gram[right][left] = total

    return tuple(tuple(row) for row in gram)
