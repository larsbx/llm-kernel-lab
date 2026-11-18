"""Negative controls for the evidence parser, independent of a Lean install."""

import pytest

from tools.check_speculative_sampling import audit_report
from tools import check_speculative_sampling as gate


@pytest.mark.parametrize("report", [
    "",
    "'other' does not depend on any axioms\n",
    "prefix 'proof' does not depend on any axioms\n",
    "'proof' does not depend on any axioms trailing\n",
    "'proof' depends on axioms: [sorryAx]\n",
    "'proof' depends on axioms: [propext, Lean.ofReduceBool]\n",
    "'proof' depends on axioms: [unprovedAssumption]\n",
    "'proof' does not depend on any axioms\n'proof' depends on axioms: [sorryAx]\n",
    "'proof' does not depend on any axioms\n'proof' does not depend on any axioms\n",
])
def test_missing_malformed_unsafe_or_ambiguous_reports_are_refused(report):
    with pytest.raises(ValueError):
        audit_report(report, "proof")


def test_exact_standard_report_is_accepted():
    assert audit_report("'proof' depends on axioms: [propext, Quot.sound]\n", "proof") == [
        "Quot.sound", "propext"
    ]
    assert audit_report("'proof' does not depend on any axioms\n", "proof") == []


def test_failed_rerun_removes_previous_success_receipt(tmp_path, monkeypatch):
    receipt = tmp_path / "acceptance.json"
    receipt.write_text('{"lean_mass_theorems_accepted": true}')

    def refused(*_args):
        raise ValueError("required proof did not compile")

    monkeypatch.setattr(gate, "accept", refused)
    monkeypatch.setattr("sys.argv", ["gate", "--out", str(receipt)])
    assert gate.main() == 1
    assert not receipt.exists()


def test_receipt_cannot_overwrite_a_proof_input(monkeypatch):
    source = gate.ROOT / "lean/SpeculativeSampling.lean"
    before = source.read_bytes()
    monkeypatch.setattr("sys.argv", ["gate", "--out", str(source)])
    assert gate.main() == 1
    assert source.read_bytes() == before
