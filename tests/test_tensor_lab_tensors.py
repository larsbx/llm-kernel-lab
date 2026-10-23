from fractions import Fraction

import pytest

from tensor_lab.tensors import mode_gram


TENSOR_2X2X2 = (
    ((1, 2), (3, 4)),
    ((5, 6), (7, 8)),
)


def test_mode_gram_matches_known_unfoldings():
    assert mode_gram(TENSOR_2X2X2, 0) == (
        (Fraction(30), Fraction(70)),
        (Fraction(70), Fraction(174)),
    )
    assert mode_gram(TENSOR_2X2X2, 1) == (
        (Fraction(66), Fraction(94)),
        (Fraction(94), Fraction(138)),
    )
    assert mode_gram(TENSOR_2X2X2, 2) == (
        (Fraction(84), Fraction(100)),
        (Fraction(100), Fraction(120)),
    )


def test_mode_gram_uses_exact_rational_arithmetic():
    tensor = (((Fraction(1, 2), Fraction(1, 3)),),)
    assert mode_gram(tensor, 2) == (
        (Fraction(1, 4), Fraction(1, 6)),
        (Fraction(1, 6), Fraction(1, 9)),
    )


def test_mode_gram_dimensions_follow_selected_mode():
    tensor = (
        ((1, 2, 3), (4, 5, 6)),
        ((7, 8, 9), (10, 11, 12)),
    )
    assert len(mode_gram(tensor, 0)) == 2
    assert len(mode_gram(tensor, 1)) == 2
    assert len(mode_gram(tensor, 2)) == 3


@pytest.mark.parametrize("mode", [-1, 3, 10])
def test_mode_gram_rejects_invalid_mode(mode):
    with pytest.raises(ValueError, match="mode must be"):
        mode_gram(TENSOR_2X2X2, mode)


@pytest.mark.parametrize(
    "tensor",
    [
        (),
        ((),),
        (((1,), (2,)), ((3,),)),
        (((1, 2),), ((3,),)),
    ],
)
def test_mode_gram_rejects_empty_or_ragged_tensors(tensor):
    with pytest.raises(ValueError):
        mode_gram(tensor, 0)
