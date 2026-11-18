"""Exact one-token speculative correction over finite integer distributions.

Reference: Leviathan, Kalman and Matias, ICML 2023, section 2.3:
https://proceedings.mlr.press/v202/leviathan23a.html

Weights are unbounded, nonnegative Python integers, not logits or floats.
For target total P and draft total Q, work on the common denominator P*Q.
An external ``randbelow(n)`` must return an independent uniform integer in
``range(n)``. Its range is checked; its independence and uniformity are caller
assumptions. The Python implementation is reference/conformance evidence;
the separate Lean mass theorem does not verify this interpreter or its RNG.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import accumulate


def _weights(values: Sequence[int], name: str) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{name} weights must be a nonempty sequence of integers")
    result = tuple(values)
    if not result or any(type(value) is not int or value < 0 for value in result):
        raise ValueError(f"{name} weights must be nonempty nonnegative integers; bool is invalid")
    if sum(result) == 0:
        raise ValueError(f"{name} total must be positive")
    return result


def _ticket(value: int, bound: int, name: str) -> int:
    if type(value) is not int or not 0 <= value < bound:
        raise ValueError(f"{name} must be an integer in [0, {bound}); bool is invalid")
    return value


@dataclass(frozen=True, slots=True, init=False)
class SpeculativePlan:
    """Validated immutable correction plan for one aligned vocabulary.

    Preparation is linear in vocabulary size and uses linear storage. After
    preparation, ``draw`` uses binary searches and at most two random draws;
    ``correct`` needs a residual search only on rejection. These are operation
    counts, not a model-serving acceleration claim. Large integer arithmetic
    and random-bit generation have their own costs.
    """

    target: tuple[int, ...]
    draft: tuple[int, ...]
    target_total: int
    draft_total: int
    denominator: int
    accepted: tuple[int, ...]
    residual: tuple[int, ...]
    rejection_total: int
    _draft_ends: tuple[int, ...]
    _residual_ends: tuple[int, ...]

    def __init__(self, target: Sequence[int], draft: Sequence[int]) -> None:
        p = _weights(target, "target")
        q = _weights(draft, "draft")
        if len(p) != len(q):
            raise ValueError("target and draft must have the same vocabulary length")
        p_total, q_total = sum(p), sum(q)
        target_mass = tuple(value * q_total for value in p)
        draft_mass = tuple(value * p_total for value in q)
        accepted = tuple(min(t, d) for t, d in zip(target_mass, draft_mass))
        residual = tuple(t - a for t, a in zip(target_mass, accepted))
        fields = {
            "target": p,
            "draft": q,
            "target_total": p_total,
            "draft_total": q_total,
            "denominator": p_total * q_total,
            "accepted": accepted,
            "residual": residual,
            "rejection_total": sum(residual),
            "_draft_ends": tuple(accumulate(draft_mass)),
            "_residual_ends": tuple(accumulate(residual)),
        }
        for name, value in fields.items():
            object.__setattr__(self, name, value)

    def _proposal(self, ticket: int) -> tuple[int, int]:
        _ticket(ticket, self.denominator, "proposal ticket")
        token = bisect_right(self._draft_ends, ticket)
        start = self._draft_ends[token - 1] if token else 0
        return token, ticket - start

    def resolve(self, ticket: int, residual_ticket: int | None = None) -> int:
        """Decode explicit tickets; no randomness and no hidden normalization.

        A proposal interval has width ``draft[token] * target_total``. Its
        first ``accepted[token]`` tickets accept that token; other tickets
        require a second ticket in the residual distribution. Supplying an
        unused residual ticket is rejected, making branch use explicit.
        """
        token, local = self._proposal(ticket)
        if local < self.accepted[token]:
            if residual_ticket is not None:
                raise ValueError("accepted proposal must not supply an unused residual ticket")
            return token
        _ticket(residual_ticket, self.rejection_total, "residual ticket")
        return bisect_right(self._residual_ends, residual_ticket)

    def draw(self, randbelow: Callable[[int], int]) -> int:
        """Draw the proposal and apply exact acceptance/residual correction."""
        ticket = _ticket(randbelow(self.denominator), self.denominator, "proposal ticket")
        token, local = self._proposal(ticket)
        if local < self.accepted[token]:
            return token
        residual_ticket = _ticket(
            randbelow(self.rejection_total), self.rejection_total, "residual ticket"
        )
        return bisect_right(self._residual_ends, residual_ticket)

    def correct(self, proposed_token: int, randbelow: Callable[[int], int]) -> int:
        """Correct an externally proposed token drawn from this plan's draft.

        The caller must provide a fresh proposal from ``draft / draft_total``
        at the same conditioning prefix as ``target``. This method can check
        support and integer ranges, but cannot authenticate that provenance.
        """
        token = _ticket(proposed_token, len(self.draft), "proposed token")
        bound = self.draft[token] * self.target_total
        if bound == 0:
            raise ValueError("proposed token has zero draft probability")
        acceptance_ticket = _ticket(randbelow(bound), bound, "acceptance ticket")
        if acceptance_ticket < self.accepted[token]:
            return token
        residual_ticket = _ticket(
            randbelow(self.rejection_total), self.rejection_total, "residual ticket"
        )
        return bisect_right(self._residual_ends, residual_ticket)


def prepare(target_weights: Sequence[int], draft_weights: Sequence[int]) -> SpeculativePlan:
    """Prepare aligned, nonempty integer distributions with positive totals."""
    return SpeculativePlan(target_weights, draft_weights)
