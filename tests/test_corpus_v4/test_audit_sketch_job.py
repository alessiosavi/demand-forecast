"""Provisional guest isolation and refusal by the existing audit consumer."""

import argparse
import json
import subprocess
import sys

import pytest

from scripts.corpus_v4.make_corpus_job import render as render_audit
from scripts.corpus_v4.make_sketch_job import render
from tests.test_corpus_v4 import test_job as audit_job_tests


def test_job_is_separate_pinned_and_syntax_valid(tmp_path):
    source = render("a" * 40, "gs://bucket/corpus-v4", "gs://bucket/reports/sketch", workers=4)
    path = tmp_path / "job.sh"
    path.write_text(source)
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "verify_published" in source and "audit_sketch" in source
    assert "-m scripts.corpus_v4.fetch_native" not in source
    assert "sketch-report.json" in source and "sketch-report.md" in source
    assert "PROVISIONAL" in source
    assert '"$CORPUS_URI/attempt-status.json"' not in source
    assert '"$CORPUS_URI/audit.json"' not in source
    assert "--workers" in source and "--memory-budget-gib" in source
    assert "trap finish EXIT" in source
    assert "status=3" in source


@pytest.mark.parametrize(
    "report_prefix",
    [
        "gs://bucket/corpus-v4",
        "gs://bucket/corpus-v4/reports",
        "gs://bucket",
        "gs://bucket/run/state",
    ],
)
def test_report_destination_must_not_overlap_corpus_or_state(report_prefix):
    with pytest.raises(ValueError):
        render(
            "a" * 40, "gs://bucket/corpus-v4", report_prefix, state_prefix="gs://bucket/run/state"
        )


@pytest.mark.parametrize(
    "changes",
    [
        dict(repo_sha="main"),
        dict(corpus_prefix="gs://bucket/x;evil"),
        dict(workers=0),
        dict(memory_budget_gib=-1),
    ],
)
def test_job_rejects_invalid_parameters(changes):
    args = dict(
        repo_sha="a" * 40,
        corpus_prefix="gs://bucket/corpus-v4",
        report_prefix="gs://bucket/reports/sketch",
    )
    args.update(changes)
    with pytest.raises((ValueError, TypeError, argparse.ArgumentTypeError)):
        render(**args)


def test_existing_audit_consumer_refuses_provisional_even_with_attempt_ids(tmp_path):
    source = render_audit("a" * 40, "gs://bucket/corpus-v4", audit_only=True)
    contract = source.split("<<'CORPUS_AUDIT_CONTRACT'\n", 1)[1].split(
        "\nCORPUS_AUDIT_CONTRACT", 1
    )[0]
    report = dict(
        artifact_type="corpus-sketch-triage",
        provisional=True,
        release_eligible=False,
        execution_ok=True,
        attempt_id="attempt",
        repo_sha="a" * 40,
        fetch_attempt_id="fetch",
    )
    path = tmp_path / "renamed-audit.json"
    path.write_text(json.dumps(report))
    result = subprocess.run(
        [sys.executable, "-c", contract, str(path), "attempt", "a" * 40, "fetch"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "does not certify" in result.stderr


@pytest.mark.parametrize("failure", [None, "scan", "upload", "status_state", "status_report"])
def test_guest_execution_and_failed_publication(tmp_path, monkeypatch, failure):
    guest, _, invoke = audit_job_tests.environment_guest.__wrapped__(tmp_path, monkeypatch)
    tool = tmp_path / "bin/python3"
    text = tool.read_text()
    text = text.replace(
        'elif name == "gcloud":\n',
        """elif name == "gcloud":
    if os.environ.get("FAIL_UPLOAD") == "1" and args[:2] == ["storage", "rsync"] and args[-1].startswith("gs://"):
        sys.exit(21)
    if args[:2] == ["storage", "cp"] and args[2].endswith("sketch-status.json"):
        destination = "state" if args[-1].startswith("gs://fake-state") else "report"
        if os.environ.get("FAIL_STATUS_COPY") == destination and json.loads(pathlib.Path(args[2]).read_text())["process_exit_code"] == 0:
            sys.exit(22)
        (pathlib.Path(os.environ["RUN_DIR"]) / ("uploaded-" + destination + ".json")).write_text(pathlib.Path(args[2]).read_text())
""",
    )
    text = text.replace(
        'elif args[:2] == ["-m", "scripts.corpus_v4.audit_fev_overlap"]:',
        """elif args[:2] == ["-m", "scripts.corpus_v4.audit_sketch"]:
    if os.environ.get("FAIL_SKETCH") == "1":
        sys.exit(19)
    target = pathlib.Path(args[3])
    (target / "sketch-report.json").write_text(json.dumps(dict(
        artifact_type="corpus-sketch-triage", provisional=True, release_eligible=False,
        execution_ok=True)))
    (target / "sketch-report.md").write_text("PROVISIONAL")
elif args[:2] == ["-m", "scripts.corpus_v4.audit_fev_overlap"]:""",
    )
    tool.write_text(text)
    (tmp_path / "bin/gcloud").write_text(text)
    (tmp_path / "job.sh").write_text(
        render("a" * 40, "gs://fake-corpus/job", "gs://fake-report/job")
    )
    if failure == "scan":
        monkeypatch.setenv("FAIL_SKETCH", "1")
    if failure == "upload":
        monkeypatch.setenv("FAIL_UPLOAD", "1")
    if failure in {"status_state", "status_report"}:
        monkeypatch.setenv("FAIL_STATUS_COPY", failure.removeprefix("status_"))
    result, calls = invoke()
    status = json.loads((guest / "state/sketch/sketch-status.json").read_text())
    assert result.returncode == (19 if failure == "scan" else 3 if failure else 0), result.stderr
    assert status["execution_ok"] is (failure is None)
    assert status["release_eligible"] is False
    for published in guest.glob("uploaded-*.json"):
        assert json.loads(published.read_text())["execution_ok"] is (failure is None)
    assert not (guest / "state/attempt-status.json").exists()
    assert all(
        not (c["name"] == "gcloud" and c["args"][-1].startswith("gs://fake-corpus")) for c in calls
    )


def test_guest_passes_subset_flags_and_publishes_progress(tmp_path, monkeypatch):
    guest, _, invoke = audit_job_tests.environment_guest.__wrapped__(tmp_path, monkeypatch)
    tool = tmp_path / "bin/python3"
    source = tool.read_text().replace(
        'elif args[:2] == ["-m", "scripts.corpus_v4.audit_fev_overlap"]:',
        """elif args[:2] == ["-m", "scripts.corpus_v4.audit_sketch"]:
    assert args[args.index("--only-folders") + 1:] == ["univariate/gpm-mergir", "univariate/era5"]
    assert "--skip-finished" in args
    target = pathlib.Path(args[3])
    state = pathlib.Path(args[args.index("--state") + 1])
    (state / "sketch-progress.json").write_text('{"units_done": 1, "units_total": 2}')
    (target / "sketch-report.json").write_text(json.dumps(dict(artifact_type="corpus-sketch-triage", provisional=True, release_eligible=False, execution_ok=True)))
    (target / "sketch-report.md").write_text("PROVISIONAL")
elif args[:2] == ["-m", "scripts.corpus_v4.audit_fev_overlap"]:""",
    )
    tool.write_text(source)
    (tmp_path / "job.sh").write_text(
        render(
            "a" * 40,
            "gs://fake-corpus/job",
            "gs://fake-report/job",
            only_folders=["univariate/gpm-mergir", "univariate/era5"],
            skip_finished=True,
        )
    )
    result, calls = invoke()
    assert result.returncode == 0, result.stderr
    assert any(
        c["name"] == "gcloud"
        and c["args"][:2] == ["storage", "cp"]
        and c["args"][-1] == "gs://fake-report/job/sketch-progress.json"
        for c in calls
    )
    assert (guest / "state/sketch-progress.json").is_file()
