from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_REVISION = "5ea47dec8086eab6a1dad6fd239688d20ea89891"
GOVERNANCE_PIN = "sha256:412a64dea58d709aae21e5a0315b8e7e96488c310a9d999267927ba3d8146a30"


def load_manifest():
    return tomllib.loads((ROOT / "ESTATE.toml").read_text(encoding="utf-8"))


def test_estate_manifest_is_v2_and_names_repository():
    data = load_manifest()
    assert data["version"] == 2
    assert data["template"] == "estate-repository-v2"
    assert data["repo"]["id"] == "llm-kernel-lab"
    assert data["repo"]["slug"] == "larsbx/llm-kernel-lab"
    assert data["principles"]["ordering"] == ["authority", "domain", "language"]


def test_estate_governance_is_immutably_pinned():
    data = load_manifest()
    dep = next(d for d in data["dep"] if d["id"] == "estate-governance")
    assert dep["rev"] == GOVERNANCE_REVISION
    assert dep["pin"] == GOVERNANCE_PIN


def test_only_lean_has_acceptance_authority():
    data = load_manifest()
    accepted = [x["name"] for x in data["language"] if x.get("acceptance_authority") is True]
    assert accepted == ["Lean"]


def test_supporting_languages_cannot_accept_proofs():
    data = load_manifest()
    supporting = [x for x in data["language"] if x["authority"] == "supporting"]
    assert supporting
    assert all(x.get("acceptance_authority") is False for x in supporting)


def test_template_forbids_empty_silos_and_mass_move():
    data = load_manifest()
    assert data["principles"]["empty_silos"] == "forbidden"
    assert data["migration"]["mass_move"] is False


def test_v1_manifest_is_retired():
    assert not (ROOT / "estate.toml").exists()
    assert not (ROOT / "tools" / "audit_estate_layout.py").exists()
