"""The generated cloud guest job is pinned, restartable and shell-parseable."""

import subprocess

import pytest

from scripts.corpus_v4.make_corpus_job import render


def test_generated_job_syntax_and_contract(tmp_path):
    script = render("a" * 40, "gs://research/corpus-v4")
    path = tmp_path / "job.sh"
    path.write_text(script)
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
    for text in (
        "RUN_DIR",
        "STATE_URI",
        '"$RUN_DIR/state"',
        "sleep 600",
        "fetch_native",
        "audit_fev_overlap",
        "audit.json",
        "stats.json",
        "git checkout --detach",
        "exit",
    ):
        assert text in script
    assert "a" * 40 in script and "gs://research/corpus-v4" in script
    assert "HF_TOKEN=" not in script


@pytest.mark.parametrize(
    "sha,prefix",
    [
        ("main", "gs://research/corpus"),
        ("a" * 40, "file:///tmp/corpus"),
        ("a" * 40, "gs://x/$(id)"),
    ],
)
def test_bad_job_parameters_refused(sha, prefix):
    with pytest.raises(ValueError):
        render(sha, prefix)


def test_failed_attempt_does_not_publish_restored_success(tmp_path, monkeypatch):
    import json
    import os

    guest = tmp_path / "guest"
    audit = guest / "state/audit/audit.json"
    audit.parent.mkdir(parents=True)
    audit.write_text('{"exit_code":0,"valid":true}')
    binary = tmp_path / "bin"
    binary.mkdir()
    for name, source in {
        "git": "#!/bin/bash\nexit 7\n",
        "sleep": "#!/bin/bash\nexit 1\n",
        "gcloud": (
            '#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n'
            'if [[ $2 == cp && $3 == */attempt-status.json ]]; then cp "$3" "$STATUS_COPY"; fi\nexit 0\n'
        ),
    }.items():
        path = binary / name
        path.write_text(source)
        path.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary}:{os.environ['PATH']}")
    monkeypatch.setenv("RUN_DIR", str(guest))
    monkeypatch.setenv("STATE_URI", "gs://fake-state/job")
    monkeypatch.setenv("CALL_LOG", str(tmp_path / "calls"))
    monkeypatch.setenv("STATUS_COPY", str(tmp_path / "status.json"))
    path = tmp_path / "job.sh"
    path.write_text(render("a" * 40, "gs://fake-corpus/job"))
    result = subprocess.run(["bash", str(path)], capture_output=True, timeout=10)
    assert result.returncode == 7
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["exit_code"] == 7 and not status["complete"]
    assert status["repo_sha"] == "a" * 40
    calls = (tmp_path / "calls").read_text()
    assert not any(" cp " in f" {line} " and "/audit.json" in line for line in calls.splitlines())
