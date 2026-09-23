import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "tools" / "audit_transcendentals.py"


def test_authoritative_tensor_layer_passes_firewall():
    run = subprocess.run(
        [sys.executable, str(AUDIT)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr


def test_firewall_detects_trigonometric_call(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import math\nvalue = math.sin(1)\n", encoding="utf-8")
    run = subprocess.run(
        [sys.executable, str(AUDIT), str(bad)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 1
    assert "sin" in run.stderr
