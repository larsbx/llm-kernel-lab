import copy
import importlib.util
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "research" / "tensor_frontier" / "algebraization-contracts.toml"
CHECKER = ROOT / "tools" / "check_algebraization_contracts.py"

spec = importlib.util.spec_from_file_location("algebraization_checker", CHECKER)
assert spec and spec.loader
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def load_registry():
    with REGISTRY.open("rb") as handle:
        return tomllib.load(handle)


def test_registry_is_valid():
    assert checker.validate(load_registry()) == []


def test_a_class_requires_preserved_target_predicate():
    data = load_registry()
    broken = copy.deepcopy(data)
    broken["algebraization"][0]["target_predicate_status"] = "changed"
    errors = checker.validate(broken)
    assert any(
        "target_predicate_status must be 'preserved' for class A" in error
        for error in errors
    )


def test_surrogate_cannot_claim_preserved_target_predicate():
    data = load_registry()
    broken = copy.deepcopy(data)
    surrogate = next(
        item for item in broken["algebraization"] if item["class"] == "S"
    )
    surrogate["target_predicate_status"] = "preserved"
    errors = checker.validate(broken)
    assert any(
        "target_predicate_status must be 'changed' for class S" in error
        for error in errors
    )


def test_loss_profile_must_name_every_dimension():
    data = load_registry()
    broken = copy.deepcopy(data)
    del broken["algebraization"][0]["loss"]["topology"]
    errors = checker.validate(broken)
    assert any("loss missing dimensions: topology" in error for error in errors)


def test_unknown_loss_status_is_rejected():
    data = load_registry()
    broken = copy.deepcopy(data)
    broken["algebraization"][0]["loss"]["smoothness"] = "probably"
    errors = checker.validate(broken)
    assert any("has invalid status: 'probably'" in error for error in errors)


def test_diagnostic_must_remain_unreplaced():
    data = load_registry()
    broken = copy.deepcopy(data)
    diagnostic = next(
        item for item in broken["algebraization"] if item["class"] == "D"
    )
    diagnostic["target_predicate_status"] = "preserved"
    errors = checker.validate(broken)
    assert any(
        "target_predicate_status must be 'not_replaced' for class D" in error
        for error in errors
    )
