"""Colab image differences are harmless; exact installed-closure pins are mandatory."""

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ec2 import closure_parity as script


@pytest.fixture
def run_parity(monkeypatch, tmp_path, capsys):
    def run(colab, ec2, names=("torch",), extras=("bench",)):
        def closure(root, requested_extras):
            assert root == "demand-forecast"
            assert requested_extras == list(extras)
            return {"demand-forecast", *names}

        monkeypatch.setattr(script, "closure", closure)
        lock = tmp_path / "colab.txt"
        freeze = tmp_path / "ec2.txt"
        output = tmp_path / "reports" / "parity.txt"
        lock.write_text(colab)
        freeze.write_text(ec2)
        args = ["--colab-lock", str(lock), "--freeze", str(freeze), "--out", str(output)]
        if extras:
            args.extend(["--extras", *extras])
        status = script.main(args)
        report = output.read_text()
        assert capsys.readouterr().out == report
        return status, report

    return run


def test_equal_versions_and_editable_root_pass(run_parity):
    status, report = run_parity(
        "# Colab\ntorch==2.11.0+cu128\ndemand-forecast==1.0\n\n",
        "torch==2.11.0+cu128\n# Editable project\n-e /home/ubuntu/fev/code\n",
    )
    assert status == 0
    assert "1 packages, 0 failures" in report
    assert "| torch | 2.11.0+cu128 | 2.11.0+cu128 | equal |" in report
    assert "| demand-forecast |" not in report


@pytest.mark.parametrize(
    "colab,ec2",
    [("2.10.0", "2.11.0"), ("2.11.0+cu128", "2.11.0"), ("2.11", "2.11.0")],
)
def test_version_strings_must_match_exactly(run_parity, colab, ec2):
    status, report = run_parity(f"torch=={colab}\n", f"torch=={ec2}\n")
    assert status == 2
    assert f"| torch | {colab} | {ec2} | version-mismatch |" in report


@pytest.mark.parametrize(
    "colab,ec2,missing",
    [("", "torch==2.11.0", "Colab"), ("torch==2.11.0", "", "EC2"), ("", "", "both")],
)
def test_missing_closure_package_fails(run_parity, colab, ec2, missing):
    status, report = run_parity(colab, ec2)
    assert status == 2
    for side in ("Colab", "EC2") if missing == "both" else (missing,):
        assert f"missing-in-{side}" in report


def test_normalises_names_and_forwards_multiple_extras(run_parity):
    status, report = run_parity(
        "Foo_Bar==1.0\n", "foo-bar==1.0\n", names=("Foo.Bar",), extras=("bench", "dev")
    )
    assert status == 0
    assert "| foo-bar | 1.0 | 1.0 | equal |" in report


def test_non_closure_differences_are_informational(run_parity):
    status, report = run_parity(
        "torch==2.11.0\njax==0.1\ngoogle-colab @ file:///colab-only.whl\n",
        "torch==2.11.0\njax==0.2\nlocal-tool==1.0\n",
        extras=(),
    )
    assert status == 0
    informational = report.split("Outside the closure (informational only):\n")[1]
    assert "| torch |" not in informational
    assert "| jax | 0.1 | 0.2 | informational: version-mismatch |" in informational
    assert "| google-colab | (unpinned) | (missing) | informational:" in informational
    assert "| local-tool | (missing) | 1.0 | informational: missing-in-Colab |" in informational


@pytest.mark.parametrize("unpinned", ["torch @ file:///torch.whl", "torch>=2.11", "torch==2.*"])
def test_unpinned_closure_package_cannot_establish_parity(run_parity, unpinned):
    status, report = run_parity(unpinned, unpinned)
    assert status == 2
    assert "unpinned-in-Colab, unpinned-in-EC2" in report


def test_conflicting_normalised_pins_fail(run_parity):
    status, report = run_parity("torch==2.11.0\nTorch==2.10.0\n", "torch==2.11.0\n")
    assert status == 2
    assert "| torch | (unpinned) | 2.11.0 | unpinned-in-Colab |" in report


@pytest.mark.parametrize("ignored", ["jax", "demand-forecast"])
def test_conflicting_pins_outside_compared_closure_are_ignored(run_parity, ignored):
    status, _ = run_parity(f"torch==2.11.0\n{ignored}==1.0\n{ignored}==2.0\n", "torch==2.11.0\n")
    assert status == 0


def test_missing_installed_root_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(script, "closure", lambda *_: set())
    lock = tmp_path / "empty.txt"
    lock.write_text("")
    report = tmp_path / "parity.txt"
    assert (
        script.main(["--colab-lock", str(lock), "--freeze", str(lock), "--out", str(report)]) == 2
    )
    assert "demand-forecast is not installed" in report.read_text()


def test_direct_script_works_outside_checkout_in_isolated_mode(tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", str(Path(script.__file__).resolve()), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--colab-lock" in result.stdout
