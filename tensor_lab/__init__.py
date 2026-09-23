"""Exact algebraic reference layer for tensor-frontier experiments."""

from .algebraic import (
    bilinear,
    gram_determinant,
    normalize_weights,
    power_sum,
    projective_spread,
    quadratic,
    rotor_apply,
    rotor_compose,
    rotor_from_parameter,
)

__all__ = [
    "bilinear",
    "gram_determinant",
    "normalize_weights",
    "power_sum",
    "projective_spread",
    "quadratic",
    "rotor_apply",
    "rotor_compose",
    "rotor_from_parameter",
]
