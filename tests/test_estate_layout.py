from pathlib import Path
import importlib.util
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def load_audit_module():
    path = ROOT / "tools" / "audit_estate_layout.py"
    spec = importlib.util.spec_from_file_location("audit_estate_layout", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_manifest():
    return tomllib.loads((ROOT / "estate.toml").read_text(encoding="utf-8"))


def test_estate_manifest_is_valid():
    audit = load_audit_module()
    audit.validate(audit.load())


def test_estate_manifest_names_canonical_repository():
    data = load_manifest()
    assert data["repository"]["id"] == "larsbx/llm-kernel-lab"
    assert data["principles"]["ordering"] == ["authority", "domain", "language"]


def test_only_lean_has_acceptance_authority():
    data = load_manifest()
    accepted = [x["name"] for x in data["language"] if x["acceptance_authority"]]
    assert accepted == ["Lean"]


def test_supporting_languages_cannot_accept_proofs():
    data = load_manifest()
    supporting = [x for x in data["language"] if x["authority"] == "supporting"]
    assert supporting
    assert all(x["acceptance_authority"] is False for x in supporting)


def test_template_forbids_empty_silos_and_mass_move():
    data = load_manifest()
    assert data["principles"]["empty_silos"] == "forbidden"
    assert data["migration"]["mass_move"] is False
