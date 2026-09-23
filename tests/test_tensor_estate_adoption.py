import copy
import importlib.util
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "research" / "tensor_frontier" / "estate-adoption.toml"
CHECKER = ROOT / "tools" / "check_tensor_estate_adoption.py"

spec = importlib.util.spec_from_file_location("tensor_estate_checker", CHECKER)
assert spec and spec.loader
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def load_manifest():
    with MANIFEST.open("rb") as handle:
        return tomllib.load(handle)


def test_estate_manifest_is_valid():
    assert checker.validate(load_manifest()) == []


def test_no_route_can_be_given_domain_acceptance_authority():
    data = load_manifest()
    broken = copy.deepcopy(data)
    broken["route"][0]["may_accept_domain_claim"] = True
    errors = checker.validate(broken)
    assert any("may_accept_domain_claim must be false" in error for error in errors)


def test_claim_and_proof_authority_are_fixed():
    data = load_manifest()
    broken = copy.deepcopy(data)
    broken["authority"]["claim_owner"] = "larsbx/llm-kernel-lab"
    broken["authority"]["proof_authority"] = "max_mojo"
    errors = checker.validate(broken)
    assert "claim_owner must remain domain_repository" in errors
    assert "proof_authority must remain domain_checker_or_lean" in errors
