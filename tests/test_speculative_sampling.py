"""Exact finite-distribution checks, independent of model or RNG statistics."""

from collections import Counter
from dataclasses import FrozenInstanceError
from fractions import Fraction
from itertools import product

import pytest

from sampling_lab.speculative import prepare


INVALID = (TypeError, ValueError)


class TicketTape:
    """A deterministic bounded-integer source that also checks requested bounds."""

    def __init__(self, *draws):
        self.draws = list(draws)
        self.calls = []

    def __call__(self, bound):
        self.calls.append(bound)
        assert self.draws, "unexpected extra random draw"
        expected_bound, ticket = self.draws.pop(0)
        assert bound == expected_bound
        return ticket

    def assert_exhausted(self):
        assert not self.draws, "the algorithm skipped a required random draw"


def assert_plan_law(target, draft):
    """Enumerate every reachable ticket; compare to the normalized target PMF."""
    plan = prepare(target, draft)
    p_total, q_total = sum(target), sum(draft)
    denominator = p_total * q_total
    expected = [Fraction(weight, p_total) for weight in target]
    proposal = [Fraction(weight, q_total) for weight in draft]
    overlap = sum(map(min, expected, proposal), Fraction(0))
    assert plan.denominator == denominator
    assert Fraction(plan.rejection_total, denominator) == 1 - overlap
    assert list(map(lambda a: Fraction(a, denominator), plan.accepted)) == [
        min(p, q) for p, q in zip(expected, proposal)
    ]
    assert list(map(lambda r: Fraction(r, denominator), plan.residual)) == [
        max(p - q, Fraction(0)) for p, q in zip(expected, proposal)
    ]

    direct = Counter()
    corrected = Counter()
    start = 0
    for token, (p, q) in enumerate(zip(target, draft)):
        width = q * p_total
        accepted_width = min(p * q_total, width)
        for local in range(width):
            global_ticket = start + local
            if local < accepted_width:
                assert plan.resolve(global_ticket) == token
                direct[plan.resolve(global_ticket)] += Fraction(1, denominator)
                tape = TicketTape((width, local))
                outcome = plan.correct(token, tape)
                tape.assert_exhausted()
                assert outcome == token
                corrected[outcome] += proposal[token] / width
            else:
                with pytest.raises(INVALID):
                    plan.resolve(global_ticket)
                for residual_ticket in range(plan.rejection_total):
                    probability = Fraction(1, denominator * plan.rejection_total)
                    direct[plan.resolve(global_ticket, residual_ticket)] += probability
                    tape = TicketTape(
                        (width, local), (plan.rejection_total, residual_ticket)
                    )
                    outcome = plan.correct(token, tape)
                    tape.assert_exhausted()
                    corrected[outcome] += proposal[token] / width / plan.rejection_total
        start += width

    assert start == denominator
    assert set(direct).issubset(range(len(target)))
    assert set(corrected).issubset(range(len(target)))
    assert [direct[i] for i in range(len(target))] == expected
    assert [corrected[i] for i in range(len(target))] == expected
    assert sum(direct.values()) == sum(corrected.values()) == 1


def test_exhaustive_small_distributions_preserve_target_exactly():
    # All 744 ordered pairs over vocabulary sizes 1..3 and weights 0..2.
    cases = 0
    for size in (1, 2, 3):
        weights = [w for w in product(range(3), repeat=size) if sum(w)]
        for target, draft in product(weights, repeat=2):
            assert_plan_law(target, draft)
            cases += 1
    assert cases == 744


@pytest.mark.parametrize(
    ("target", "draft"),
    [
        ((1, 4, 2), (3, 1, 1)),  # unequal totals and mixed acceptance
        ((0, 2, 0, 3), (4, 0, 1, 0)),  # disjoint supports
        ((0, 2, 0, 1), (0, 6, 0, 3)),  # equal PMFs, unequal weights
        ((1, 0, 2, 0), (0, 1, 0, 1)),
        ((0, 0, 1), (1, 0, 0)),
    ],
)
def test_edge_distributions_have_exact_marginals(target, draft):
    assert_plan_law(target, draft)


def test_input_rescaling_preserves_normalized_acceptance_and_residual():
    original = prepare((1, 2, 3), (3, 0, 2))
    scaled = prepare((7, 14, 21), (33, 0, 22))
    for field in ("accepted", "residual"):
        assert [Fraction(x, original.denominator) for x in getattr(original, field)] == [
            Fraction(x, scaled.denominator) for x in getattr(scaled, field)
        ]
    assert Fraction(original.rejection_total, original.denominator) == Fraction(
        scaled.rejection_total, scaled.denominator
    )


def test_plan_copies_input_and_is_frozen():
    target, draft = [1, 2], [2, 1]
    plan = prepare(target, draft)
    target[0] = 99
    draft.append(99)
    assert plan.target == (1, 2)
    assert plan.draft == (2, 1)
    assert plan.target_total == plan.draft_total == 3
    assert plan.accepted == (3, 3)
    assert plan.residual == (0, 3)
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        plan.denominator = 999


@pytest.mark.parametrize(
    ("target", "draft"),
    [
        ((), ()),
        ((1,), ()),
        ((), (1,)),
        ((1, 2), (3,)),
        ((0, 0), (1, 0)),
        ((1, 0), (0, 0)),
        ((-1, 2), (1, 1)),
        ((1, 1), (2, -1)),
    ],
)
def test_invalid_supports_are_refused(target, draft):
    with pytest.raises(INVALID):
        prepare(target, draft)


@pytest.mark.parametrize("bad", [True, False, 1.0, 0.0, float("nan"), float("inf"), "1", None, Fraction(1)])
@pytest.mark.parametrize("side", ["target", "draft"])
def test_weights_must_be_strict_python_integers(bad, side):
    inputs = {"target": (1, 1), "draft": (1, 1)}
    inputs[side] = (bad, 1)
    with pytest.raises(INVALID):
        prepare(inputs["target"], inputs["draft"])


def test_arbitrary_precision_weights_do_not_wrap_or_round():
    big = 2**257 + 123
    plan = prepare((big, 1), (1, big))
    assert plan.denominator == (big + 1) ** 2
    assert plan.accepted == (big + 1, big + 1)
    assert plan.residual == (big * big - 1, 0)
    assert plan.rejection_total == big * big - 1
    assert plan.resolve(0) == 0
    assert plan.resolve(big + 1) == 1
    assert plan.resolve(plan.denominator - 1, plan.rejection_total - 1) == 0


@pytest.mark.parametrize("ticket", [-1, 9, 10, True, False, 0.0, 1.0, float("nan"), "0", None])
def test_global_ticket_validation(ticket):
    plan = prepare((1, 2), (2, 1))
    with pytest.raises(INVALID):
        plan.resolve(ticket)


@pytest.mark.parametrize("ticket", [-1, 3, 4, True, False, 0.0, float("nan"), "0", None])
def test_required_residual_ticket_validation(ticket):
    plan = prepare((1, 2), (2, 1))
    with pytest.raises(INVALID):
        plan.resolve(3, ticket)


@pytest.mark.parametrize("ticket", [0, -1, 3, True, 0.0, "0"])
def test_acceptance_refuses_any_unused_residual_ticket(ticket):
    plan = prepare((1, 2), (2, 1))
    with pytest.raises(INVALID):
        plan.resolve(0, ticket)


@pytest.mark.parametrize("token", [-1, 3, True, False, 0.0, "0", None])
def test_correct_refuses_invalid_proposal_before_drawing(token):
    plan = prepare((1, 0, 2), (2, 0, 1))
    tape = TicketTape()
    with pytest.raises(INVALID):
        plan.correct(token, tape)
    assert tape.calls == []


def test_correct_refuses_zero_draft_support_before_drawing():
    plan = prepare((0, 2, 1), (2, 0, 1))
    tape = TicketTape()
    with pytest.raises(INVALID):
        plan.correct(1, tape)
    assert tape.calls == []


@pytest.mark.parametrize(("ticket", "outcome"), [(0, 0), (2, 0), (6, 1), (8, 1)])
def test_draw_acceptance_uses_only_one_uniform_ticket(ticket, outcome):
    plan = prepare((1, 2), (2, 1))
    tape = TicketTape((9, ticket))
    assert plan.draw(tape) == outcome
    tape.assert_exhausted()


@pytest.mark.parametrize(("ticket", "residual_ticket"), [(3, 0), (4, 1), (5, 2)])
def test_draw_rejection_uses_one_additional_residual_ticket(ticket, residual_ticket):
    plan = prepare((1, 2), (2, 1))
    tape = TicketTape((9, ticket), (3, residual_ticket))
    assert plan.draw(tape) == 1
    tape.assert_exhausted()


@pytest.mark.parametrize("bad", [-1, 9, True, False, 0.0, float("nan"), "0", None])
def test_draw_validates_random_source_output(bad):
    plan = prepare((1, 2), (2, 1))
    tape = TicketTape((9, bad))
    with pytest.raises(INVALID):
        plan.draw(tape)
    tape.assert_exhausted()


@pytest.mark.parametrize("bad", [-1, 3, True, False, 0.0, float("nan"), "0", None])
def test_draw_validates_residual_random_source_output(bad):
    plan = prepare((1, 2), (2, 1))
    tape = TicketTape((9, 3), (3, bad))
    with pytest.raises(INVALID):
        plan.draw(tape)
    tape.assert_exhausted()


@pytest.mark.parametrize("bad", [-1, 6, True, False, 0.0, float("nan"), "0", None])
def test_correct_validates_acceptance_random_source_output(bad):
    plan = prepare((1, 2), (2, 1))
    tape = TicketTape((6, bad))
    with pytest.raises(INVALID):
        plan.correct(0, tape)
    tape.assert_exhausted()


@pytest.mark.parametrize("bad", [-1, 3, True, False, 0.0, float("nan"), "0", None])
def test_correct_validates_residual_random_source_output(bad):
    plan = prepare((1, 2), (2, 1))
    tape = TicketTape((6, 3), (3, bad))
    with pytest.raises(INVALID):
        plan.correct(0, tape)
    tape.assert_exhausted()


def test_identical_distributions_never_request_zero_bound():
    plan = prepare((0, 2, 1), (0, 6, 3))
    assert plan.rejection_total == 0
    for ticket in range(plan.denominator):
        tape = TicketTape((plan.denominator, ticket))
        assert plan.draw(tape) in (1, 2)
        tape.assert_exhausted()


def test_callback_errors_are_not_silently_replaced_with_an_output():
    plan = prepare((1, 2), (2, 1))

    def unavailable(_bound):
        raise RuntimeError("random source unavailable")

    with pytest.raises(RuntimeError, match="random source unavailable"):
        plan.draw(unavailable)
    with pytest.raises(RuntimeError, match="random source unavailable"):
        plan.correct(0, unavailable)
