"""Radical-free exact reference predicates for the tensor frontier lab.

This module is intentionally small and exact.  Accelerator implementations are
expected to agree with these predicates on bounded corpora before their output
is used as research evidence.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Sequence

Scalar = Fraction
Vector = Sequence[Scalar]
Matrix = Sequence[Sequence[Scalar]]
Rotor = tuple[Scalar, Scalar]


def _q(value: int | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def _identity_gram(n: int) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(Fraction(int(i == j)) for j in range(n))
        for i in range(n)
    )


def bilinear(x: Vector, y: Vector, gram: Matrix | None = None) -> Fraction:
    """Return x^T G y exactly."""
    if len(x) != len(y):
        raise ValueError("dimension mismatch")
    g = gram if gram is not None else _identity_gram(len(x))
    if len(g) != len(x) or any(len(row) != len(x) for row in g):
        raise ValueError("Gram matrix must be square with vector dimension")
    return sum(
        (_q(x[i]) * _q(g[i][j]) * _q(y[j]) for i in range(len(x)) for j in range(len(x))),
        Fraction(0),
    )


def quadratic(x: Vector, gram: Matrix | None = None) -> Fraction:
    """Return Q(x)=B(x,x) exactly."""
    return bilinear(x, x, gram)


def gram_determinant(x: Vector, y: Vector, gram: Matrix | None = None) -> Fraction:
    """Return Q(x)Q(y)-B(x,y)^2 without square roots or angles."""
    qx = quadratic(x, gram)
    qy = quadratic(y, gram)
    bxy = bilinear(x, y, gram)
    return qx * qy - bxy * bxy


def projective_spread(x: Vector, y: Vector, gram: Matrix | None = None) -> Fraction:
    """Return the scale-invariant spread Delta/(Q(x)Q(y)).

    Isotropic inputs are outside this affine ratio chart and are rejected.
    """
    qx = quadratic(x, gram)
    qy = quadratic(y, gram)
    denom = qx * qy
    if denom == 0:
        raise ZeroDivisionError("spread undefined for isotropic input")
    return gram_determinant(x, y, gram) / denom


def rotor_from_parameter(t: int | Fraction) -> Rotor:
    """Rational parametrization of c^2+s^2=1."""
    t = _q(t)
    denom = Fraction(1) + t * t
    if denom == 0:
        raise ZeroDivisionError("rotor chart denominator is zero")
    return ((Fraction(1) - t * t) / denom, (2 * t) / denom)


def rotor_compose(left: Rotor, right: Rotor) -> Rotor:
    """Compose two algebraic rotors with no angle representation."""
    c1, s1 = map(_q, left)
    c2, s2 = map(_q, right)
    return (c1 * c2 - s1 * s2, s1 * c2 + c1 * s2)


def rotor_apply(rotor: Rotor, v: Sequence[int | Fraction]) -> tuple[Fraction, Fraction]:
    """Apply the 2x2 rotor [[c,-s],[s,c]]."""
    if len(v) != 2:
        raise ValueError("rotor_apply requires a 2-vector")
    c, s = map(_q, rotor)
    x, y = map(_q, v)
    return (c * x - s * y, s * x + c * y)


def normalize_weights(weights: Iterable[int | Fraction]) -> tuple[Fraction, ...]:
    """Normalize multiplicative weights directly; no logits/exp/softmax."""
    ws = tuple(_q(w) for w in weights)
    if not ws:
        raise ValueError("at least one weight is required")
    total = sum(ws, Fraction(0))
    if total == 0:
        raise ZeroDivisionError("weight sum is zero")
    return tuple(w / total for w in ws)


def power_sum(weights: Iterable[int | Fraction], order: int) -> Fraction:
    """Polynomial probability statistic sum_i p_i^order.

    This is not labeled as Shannon entropy.
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    ps = normalize_weights(weights)
    return sum((p**order for p in ps), Fraction(0))
