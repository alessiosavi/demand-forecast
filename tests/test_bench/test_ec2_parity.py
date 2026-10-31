"""The EC2 probe has an offline path through exactly the same parity checks."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/ec2/check_real_parity.py"


def test_parity_probe_self_test(tmp_path):
    output = tmp_path / "parity.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--self-test", "--batches", "3", "--out", str(output)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["self_test"] is True
    assert report["device"] == "cpu"
    assert len(report["checks"]) >= 5
    assert all(check["status"] == "pass" for check in report["checks"])
    assert {c["check"] for c in report["checks"]} >= {
        "training batches",
        "prepared validation",
        "inference batches",
        "evaluation",
        "origin facts",
    }
