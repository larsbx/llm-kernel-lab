from fractions import Fraction

import pytest

from tensor_lab.algebraic import (
    normalize_weights,
    power_sum,
    projective_spread,
    quadratic,
    rotor_apply,
    rotor_compose,
    rotor_from_parameter,
)


def test_rotor_parameterization_is_on_conic():
    c, s = rotor_from_parameter(Fraction(3, 5))
    assert c * c + s * s == 1


def test_rotor_composition_stays_on_conic():
    r = rotor_compose(rotor_from_parameter(Fraction(1, 2)), rotor_from_parameter(Fraction(2, 3)))
    c, s = r
    assert c * c + s * s == 1


def test_rotor_preserves_quadrance():
    r = rotor_from_parameter(Fraction(4, 7))
    v = (Fraction(5), Fraction(-2))
    assert quadratic(rotor_apply(r, v)) == quadratic(v)


def test_spread_is_projectively_scale_invariant():
    x = (Fraction(1), Fraction(2), Fraction(3))
    y = (Fraction(2), Fraction(-1), Fraction(4))
    base = projective_spread(x, y)
    assert projective_spread(tuple(7 * a for a in x), tuple(-5 * b for b in y)) == base


def test_spread_rejects_isotropic_chart():
    gram = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(-1)))
    with pytest.raises(ZeroDivisionError):
        projective_spread((Fraction(1), Fraction(1)), (Fraction(1), Fraction(0)), gram)


def test_weights_are_normalized_without_logits():
    assert normalize_weights((2, 3, 5)) == (Fraction(1, 5), Fraction(3, 10), Fraction(1, 2))
    assert power_sum((2, 3, 5), 2) == Fraction(19, 50)


@pytest.mark.parametrize("weights", [(2, -1), (2, 0), (0,), (-1,)])
def test_nonpositive_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="strictly positive"):
        normalize_weights(weights)
    with pytest.raises(ValueError, match="strictly positive"):
        power_sum(weights, 2)
