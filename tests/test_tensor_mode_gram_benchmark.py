"""Benchmark evidence must fail closed on invalid counts or numeric drift."""
import numpy as np
import pytest

from benchmarks.tensor_frontier.benchmark_mode_gram import benchmark, measure


@pytest.mark.parametrize("warmup,iterations,rounds", [(-1, 1, 1), (0, 0, 1), (0, 1, 0)])
def test_invalid_counts_fail_before_library_loading(warmup, iterations, rounds):
    with pytest.raises(ValueError, match="warmup"):
        benchmark("missing.so", warmup, iterations, rounds)


@pytest.mark.parametrize("bad", [1.0 + 1e-6, float("nan"), float("inf")])
def test_bad_measured_result_cannot_become_evidence(bad):
    calls = iter([np.array([[1.0]]), np.array([[bad]])])
    with pytest.raises(ValueError, match="tolerance failed"):
        measure(lambda *_: next(calls), np.array([[[1.]]]), 0, ((1,),), 1, 1)
