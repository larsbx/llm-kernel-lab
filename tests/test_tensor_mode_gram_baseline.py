import json
from pathlib import Path

import numpy as np
import pytest

from tensor_lab.numpy_baseline import mode_gram_allclose, mode_gram_numpy
from tensor_lab.tensors import mode_gram


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "benchmarks" / "tensor_frontier" / "mode_gram_cases.json"


def load_cases():
    return json.loads(CORPUS.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", load_cases(), ids=lambda case: case["id"])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_numpy_baseline_agrees_with_exact_oracle(case, mode):
    reference = mode_gram(case["tensor"], mode)
    observed = mode_gram_numpy(case["tensor"], mode)
    assert mode_gram_allclose(reference, observed)


def test_numpy_baseline_uses_selected_mode_dimension():
    tensor = (
        ((1, 2, 3), (4, 5, 6)),
        ((7, 8, 9), (10, 11, 12)),
    )
    assert mode_gram_numpy(tensor, 0).shape == (2, 2)
    assert mode_gram_numpy(tensor, 1).shape == (2, 2)
    assert mode_gram_numpy(tensor, 2).shape == (3, 3)


def test_declared_relation_rejects_material_error():
    reference = ((1, 0), (0, 1))
    observed = np.asarray(((1.0, 0.0), (0.0, 1.0 + 1e-6)))
    assert not mode_gram_allclose(reference, observed)


@pytest.mark.parametrize("mode", [-1, 3])
def test_numpy_baseline_rejects_invalid_mode(mode):
    with pytest.raises(ValueError, match="mode must be"):
        mode_gram_numpy([[[1]]], mode)


@pytest.mark.parametrize("tensor", [[], [[]], [[[]]], [[1, 2]], [[[1]], [[1, 2]]]])
def test_numpy_baseline_rejects_invalid_tensor_shape(tensor):
    with pytest.raises(ValueError):
        mode_gram_numpy(tensor, 0)
