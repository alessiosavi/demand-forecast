"""Execute the guest script offline with real bash, jobs and GNU timeout."""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/gcp/guest-run.sh"


def assert_marker_identity(path, label="run-a", job_uri="gs://test-bucket/job.sh"):
    fields = dict(line.split("=", 1) for line in path.read_text().splitlines())
    assert fields["label"] == label
    assert fields["job_uri"] == job_uri
    assert fields["time"].endswith("Z")
    datetime.fromisoformat(fields["time"].replace("Z", "+00:00"))


STUB = r"""
import json, os, shutil, sys, time
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
with Path(os.environ['CALLS']).open('a') as out: out.write(json.dumps([name, *args])+'\n')
store = Path(os.environ['STORE'])
if name == 'curl':
    key = args[-1].rsplit('/', 1)[1]
    if key == os.environ.get('MISSING_METADATA'):
        print('HTTP 404: metadata not found', file=sys.stderr)
        sys.exit(22)
    print(json.loads(os.environ['METADATA'])[key])
elif name == 'shutdown': pass
elif name == 'install': pass
elif name == 'sleep': time.sleep(0.05)
elif name == 'timeout':
    os.execv(os.environ['REAL_TIMEOUT'], [os.environ['REAL_TIMEOUT'], *args])
elif name == 'gcloud':
    assert '--format=json' in args
    if args[:3] == ['storage', 'objects', 'describe']:
        if (store/args[3].rsplit('/', 1)[1]).exists(): print('{}')
        else:
            print('HTTPError 404: No such object', file=sys.stderr)
            sys.exit(1)
    elif args[:2] == ['storage', 'rsync']:
        src, dst = args[2:4]
        if src.startswith('gs://'):
            time.sleep(float(os.environ.get('DELAY_RESTORE', '0')))
            if not any(store.rglob('*')):
                print('InvalidUrlError: Source URL matched no objects', file=sys.stderr)
                sys.exit(1)
            if os.environ.get('RESTORE_ERROR'):
                print('403 Permission denied restoring state', file=sys.stderr)
                sys.exit(1)
            Path(dst).mkdir(parents=True, exist_ok=True)
            import re
            exclude = next((a.split('=', 1)[1] for a in args if a.startswith('--exclude=')), '')
            for p in store.rglob('*'):
                relative = p.relative_to(store)
                if p.is_file() and not (exclude and re.search(exclude, str(relative))):
                    target = Path(dst)/relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(p, target)
        else:
            count = Path(os.environ['COUNT'])
            n = int(count.read_text())+1 if count.exists() else 1
            count.write_text(str(n))
            time.sleep(float(os.environ.get('DELAY_UPLOAD', '0')))
            if os.environ.get('HANG_COPY') == 'first' and n == 1: time.sleep(30)
            if os.environ.get('FAIL_COPY') == 'all' or (os.environ.get('FAIL_COPY') == 'first' and n == 1):
                print('synthetic copy failure', file=sys.stderr)
                sys.exit(1)
            shutil.copytree(src, store, dirs_exist_ok=True)
    elif args[:2] == ['storage', 'cp']:
        src, dst = args[2:4]
        if os.environ.get('HANG_MARKER') and dst.endswith('/PREEMPTED'): time.sleep(30)
        if src.startswith('gs://') and dst.startswith('gs://'):
            if src.endswith('/'+os.environ.get('FAIL_ARCHIVE', 'NO_MARKER')):
                print('synthetic archive copy failure', file=sys.stderr)
                sys.exit(1)
            target = store/dst.split('/state/', 1)[1]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(store/src.split('/state/', 1)[1], target)
        elif src.startswith('gs://'): shutil.copy(os.environ['JOB'], dst)
        else:
            target = store/dst.split('/state/', 1)[1]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, target)
    elif args[:2] == ['storage', 'cat']:
        target = store/args[2].split('/state/', 1)[1]
        if os.environ.get('MARKER_READ_ERROR'):
            print('403 Permission denied', file=sys.stderr)
            sys.exit(1)
        if not target.exists():
            # SDK 593 storage cat uses NameExpansionIterator, unlike describe.
            print('The following URLs matched no objects or files:\n'+args[2], file=sys.stderr)
            sys.exit(1)
        sys.stdout.write(target.read_text())
    elif args[:2] == ['storage', 'rm']:
        (store/args[2].split('/state/', 1)[1]).unlink()
    else: raise AssertionError(args)
else: raise AssertionError(name)
"""


@pytest.fixture
def guest(tmp_path):
    date = shutil.which("gdate") or shutil.which("date")
    timeout = shutil.which("timeout") or shutil.which("gtimeout")
    if (
        not date
        or not timeout
        or subprocess.run([date, "-d", "@0", "+%s"], capture_output=True).returncode
    ):
        pytest.skip("GNU date and timeout required")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("gcloud", "curl", "shutdown", "sleep", "install", "timeout"):
        exe = bin_dir / name
        exe.write_text(f"#!{sys.executable}\n" + STUB)
        exe.chmod(0o755)
    (bin_dir / "date").symlink_to(date)
    store = tmp_path / "store"
    store.mkdir()
    calls = tmp_path / "calls"
    job = tmp_path / "job.sh"
    job.write_text(
        'printf checkpoint > "$RUN_DIR/state/checkpoint"\n'
        + f"{sys.executable} -c 'import time; time.sleep(0.4)'\n"
    )
    metadata = {
        "guard-label": "run-a",
        "guard-state-uri": "gs://test-bucket/runs/run-a/state",
        "guard-job-uri": "gs://test-bucket/job.sh",
        "guard-reserved-end": (datetime.now(timezone.utc) + timedelta(seconds=610)).isoformat(),
    }
    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        CALLS=str(calls),
        STORE=str(store),
        JOB=str(job),
        COUNT=str(tmp_path / "count"),
        METADATA=json.dumps(metadata),
        GCP_RUN_ROOT=str(tmp_path / "runs"),
        REAL_TIMEOUT=timeout,
    )

    class Guest:
        def run(
            self,
            *args,
            fail="",
            seconds=610,
            hang="",
            hang_marker="",
            label=None,
            job_uri=None,
            **overrides,
        ):
            if label is not None:
                metadata["guard-label"] = label
            if job_uri is not None:
                metadata["guard-job-uri"] = job_uri
            metadata["guard-reserved-end"] = (
                datetime.now(timezone.utc) + timedelta(seconds=seconds)
            ).isoformat()
            return subprocess.run(
                ["bash", str(SCRIPT), *args],
                env=dict(
                    env,
                    METADATA=json.dumps(metadata),
                    FAIL_COPY=fail,
                    HANG_COPY=hang,
                    HANG_MARKER=hang_marker,
                    **overrides,
                ),
                capture_output=True,
                text=True,
                timeout=15,
            )

        def calls(self):
            return [json.loads(line) for line in calls.read_text().splitlines()]

    g = Guest()
    g.start = lambda: subprocess.Popen(
        ["bash", str(SCRIPT)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    g.store, g.job, g.tmp = store, job, tmp_path
    return g


def test_failed_periodic_copy_does_not_kill_job_and_next_copy_recovers(guest):
    result = guest.run(fail="first")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (guest.store / "DONE").exists()
    assert (guest.store / "checkpoint").read_text() == "checkpoint"
    assert (guest.store / "HEARTBEAT").exists()
    assert "copy failed" in result.stderr
    assert (guest.store / "logs/run-a.log").exists()
    assert any(c[0] == "shutdown" for c in guest.calls())
    for key in ("STARTED", "HEARTBEAT", "DONE"):
        assert_marker_identity(guest.store / key)


def test_missing_metadata_powers_off_before_restore_or_job_fetch(guest):
    result = guest.run(MISSING_METADATA="guard-label")
    assert result.returncode != 0
    assert any(c[0] == "shutdown" for c in guest.calls())
    assert not any(c[0] == "gcloud" for c in guest.calls())
    assert not (guest.tmp / "runs/run-a/job.sh").exists()


def test_job_timeout_writes_timeout_and_powers_off(guest):
    guest.job.write_text(f"{sys.executable} -c 'import time; time.sleep(20)'\n")
    result = guest.run(seconds=602)
    assert result.returncode != 0
    assert (guest.store / "TIMEOUT").exists(), result.stderr
    assert not (guest.store / "DONE").exists()
    assert any(c[0] == "shutdown" for c in guest.calls())
    assert_marker_identity(guest.store / "TIMEOUT")


def test_done_never_fetches_or_runs_job(guest):
    (guest.store / "DONE").touch()
    result = guest.run()
    assert result.returncode == 0, result.stderr
    assert not any(c[:3] == ["gcloud", "storage", "cp"] for c in guest.calls())
    assert any(c[0] == "shutdown" for c in guest.calls())
    assert "no job URI" in result.stderr
    assert "archive" in result.stderr and "new state prefix" in result.stderr


def test_failed_final_copy_never_writes_done(guest):
    result = guest.run(fail="all")
    assert result.returncode != 0
    assert not (guest.store / "DONE").exists()
    assert (guest.store / "FAILED").exists()


def test_preemption_copies_state_once_and_marks_interruption(guest):
    local = guest.tmp / "runs/run-a/state"
    local.mkdir(parents=True)
    (local / "checkpoint").write_text("resume me")
    result = guest.run("--preempt")
    assert result.returncode == 0, result.stderr
    assert (guest.store / "checkpoint").read_text() == "resume me"
    assert (guest.store / "PREEMPTED").exists()
    assert_marker_identity(guest.store / "PREEMPTED")
    assert not any(c[0] == "shutdown" for c in guest.calls())


def test_job_failure_preserves_status_and_preemption_hook_ignores_normal_end(guest):
    guest.job.write_text("echo failing-job\nexit 7\n")
    result = guest.run()
    assert result.returncode == 7, result.stderr
    assert "exit_status=7" in (guest.store / "FAILED").read_text()
    assert_marker_identity(guest.store / "FAILED")
    guest.run("--preempt")
    assert not (guest.store / "PREEMPTED").exists()


def test_stuck_periodic_copy_is_stopped_before_final_sync(guest):
    import time

    start = time.monotonic()
    result = guest.run(hang="first")
    assert result.returncode == 0, result.stderr
    assert time.monotonic() - start < 8
    assert (guest.store / "DONE").exists()
    assert (guest.store / "checkpoint").exists()


def test_external_term_stops_stubborn_job_before_final_copy(guest):
    import signal
    import time

    guest.job.write_text(
        f"exec {sys.executable} - <<'PY'\n"
        + """import os, signal, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
state = Path(os.environ['RUN_DIR']) / 'state'
(state / 'pid').write_text(str(os.getpid()))
while True:
    (state / 'tick').write_text(str(time.time()))
    time.sleep(0.02)
PY
"""
    )
    process = guest.start()
    pidfile = guest.tmp / "runs/run-a/state/pid"
    deadline = time.monotonic() + 4
    pid = None
    try:
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pidfile.exists()
        pid = int(pidfile.read_text())
        process.terminate()
        process.communicate(timeout=8)
        assert process.returncode == 143
        tick = pidfile.with_name("tick")
        before = tick.read_text()
        time.sleep(0.15)
        assert tick.read_text() == before, "job still writes after final copy"
        assert (guest.store / "FAILED").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
        if pid:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_preemption_marker_upload_is_bounded(guest):
    import time

    local = guest.tmp / "runs/run-a/state"
    local.mkdir(parents=True)
    (local / "checkpoint").write_text("latest")
    start = time.monotonic()
    result = guest.run("--preempt", hang_marker="yes")
    assert result.returncode == 0, result.stderr
    assert time.monotonic() - start < 8
    assert (guest.store / "checkpoint").read_text() == "latest"
    assert "preemption marker failed" in result.stderr


def test_job_restarts_from_restored_state(guest):
    (guest.store / "checkpoint-resume").write_text("saved")
    guest.job.write_text(
        '[ "$(cat "$RUN_DIR/state/checkpoint-resume")" = saved ] || exit 9\n'
        'printf resumed > "$RUN_DIR/state/checkpoint-next"\n'
    )
    result = guest.run()
    assert result.returncode == 0, result.stderr
    assert (guest.store / "checkpoint-next").read_text() == "resumed"
    assert (guest.store / "DONE").exists()


def test_timeout_marker_survives_failed_final_state_copy(guest):
    guest.job.write_text(f"{sys.executable} -c 'import time; time.sleep(20)'\n")
    result = guest.run(seconds=602, fail="all")
    assert result.returncode == 124, result.stderr
    assert (guest.store / "TIMEOUT").exists()
    assert not (guest.store / "DONE").exists()


def test_first_run_seeds_prefix_before_restore_and_excludes_started(guest):
    guest.job.write_text(
        '[ ! -e "$RUN_DIR/state/STARTED" ] || exit 9\nprintf first > "$RUN_DIR/state/checkpoint"\n'
    )
    result = guest.run()
    assert result.returncode == 0, result.stderr
    started = (guest.store / "STARTED").read_text()
    assert "label=run-a" in started
    boot_time = next(
        line.split("=", 1)[1] for line in started.splitlines() if line.startswith("boot_time=")
    )
    assert boot_time.endswith("Z")
    datetime.fromisoformat(boot_time.replace("Z", "+00:00"))
    assert (guest.store / "DONE").exists()
    assert (guest.store / "checkpoint").read_text() == "first"
    calls = guest.calls()
    seed = next(
        i
        for i, c in enumerate(calls)
        if c[:3] == ["gcloud", "storage", "cp"] and c[4].endswith("/STARTED")
    )
    restore = next(
        i
        for i, c in enumerate(calls)
        if c[:3] == ["gcloud", "storage", "rsync"] and c[3].startswith("gs://")
    )
    assert seed < restore


def test_restore_permission_error_never_runs_job_or_writes_done(guest):
    (guest.store / "checkpoint").write_text("previous")
    guest.job.write_text('touch "$RUN_DIR/ran-job"\n')
    result = guest.run(RESTORE_ERROR="yes")
    assert result.returncode != 0
    assert not (guest.tmp / "runs/run-a/ran-job").exists()
    assert not (guest.store / "DONE").exists()


def test_started_is_not_a_terminal_marker(guest):
    (guest.store / "STARTED").write_text("label=earlier\n")
    result = guest.run()
    assert result.returncode == 0, result.stderr
    assert (guest.store / "DONE").exists()


def test_storage_operations_have_separate_default_budgets(guest):
    result = guest.run()
    assert result.returncode == 0, result.stderr
    calls = [c for c in guest.calls() if c[0] == "timeout" and "gcloud" in c]
    restore = [
        c
        for c in calls
        if "rsync" in c
        and any(a.startswith("gs://") for a in c[c.index("rsync") + 1 : c.index("rsync") + 2])
    ]
    assert len(restore) == 1 and "1800s" in restore[0]
    uploads = [c for c in calls if "rsync" in c and c not in restore]
    assert "480s" in uploads[-1]
    assert any("540s" in c for c in uploads[:-1])
    fetch = next(c for c in calls if "cp" in c and c[c.index("cp") + 1].startswith("gs://"))
    assert "120s" in fetch
    small = [c for c in calls if c not in restore + uploads + [fetch]]
    assert small and all("60s" in c for c in small)


def test_restore_budget_can_exceed_short_marker_budget(guest):
    result = guest.run(
        DELAY_RESTORE="0.6",
        GCP_RESTORE_TIMEOUT="2s",
        GCP_MARKER_TIMEOUT="0.4s",
        GCP_JOB_FETCH_TIMEOUT="0.5s",
        GCP_PERIODIC_COPY_TIMEOUT="0.7s",
        GCP_FINAL_COPY_TIMEOUT="0.8s",
        GCP_LOG_TIMEOUT="0.9s",
    )
    assert result.returncode == 0, result.stderr
    assert (guest.store / "DONE").exists()
    calls = [c for c in guest.calls() if c[0] == "timeout"]
    for limit in ("2s", "0.4s", "0.5s", "0.7s", "0.8s", "0.9s"):
        assert any(limit in c for c in calls), limit


def test_final_copy_enforces_its_override_and_refuses_done_on_timeout(guest):
    guest.job.write_text('printf checkpoint > "$RUN_DIR/state/checkpoint"\n')
    result = guest.run(DELAY_UPLOAD="2", GCP_FINAL_COPY_TIMEOUT="0.2s")
    assert result.returncode != 0, result.stderr
    assert not (guest.store / "DONE").exists()
    assert (guest.store / "FAILED").exists()


def test_different_job_archives_terminal_markers_before_started_and_restores_checkpoint(guest):
    assert guest.run().returncode == 0
    old_done = (guest.store / "DONE").read_text()
    previous = "label=run-a\njob_uri=gs://test-bucket/job.sh\ntime=2026-11-13T00:00:00Z\n"
    for key in ("FAILED", "TIMEOUT", "PREEMPTED"):
        (guest.store / key).write_text(previous)
    guest.job.write_text(
        '[ "$(cat "$RUN_DIR/state/checkpoint")" = checkpoint ] || exit 9\n'
        'printf next > "$RUN_DIR/state/checkpoint"\n'
    )
    before = len(guest.calls())
    result = guest.run(label="run-b", job_uri="gs://test-bucket/leg-2.sh")
    assert result.returncode == 0, result.stderr
    assert (guest.store / "checkpoint").read_text() == "next"
    assert_marker_identity(guest.store / "DONE", "run-b", "gs://test-bucket/leg-2.sh")
    calls = guest.calls()[before:]
    started = next(
        i
        for i, c in enumerate(calls)
        if c[:3] == ["gcloud", "storage", "cp"] and c[4].endswith("/STARTED")
    )
    for key in ("DONE", "FAILED", "TIMEOUT", "PREEMPTED"):
        archived = list((guest.store / "history").glob(f"*-run-b-{key}"))
        assert len(archived) == 1
        assert archived[0].read_text() == (old_done if key == "DONE" else previous)
        timestamp = archived[0].name.removesuffix(f"-run-b-{key}")
        assert timestamp.endswith("Z")
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        copied = next(
            i
            for i, c in enumerate(calls)
            if c[:3] == ["gcloud", "storage", "cp"]
            and c[3].endswith("/" + key)
            and "/history/" in c[4]
        )
        deleted = next(
            i
            for i, c in enumerate(calls)
            if c[:3] == ["gcloud", "storage", "rm"] and c[3].endswith("/" + key)
        )
        assert copied < deleted < started
        if key != "DONE":
            assert not (guest.store / key).exists()


@pytest.mark.parametrize("key", ["DONE", "FAILED", "TIMEOUT", "PREEMPTED"])
def test_archive_copy_failure_preserves_marker_and_never_runs_job(guest, key):
    previous = "label=old\njob_uri=gs://test-bucket/old.sh\ntime=2026-11-13T00:00:00Z\n"
    (guest.store / key).write_text(previous)
    (guest.store / "checkpoint").write_text("saved")
    guest.job.write_text('touch "$RUN_DIR/ran-job"\n')
    result = guest.run(FAIL_ARCHIVE=key)
    assert result.returncode != 0, result.stderr
    assert (guest.store / key).read_text() == previous
    assert (guest.store / "checkpoint").read_text() == "saved"
    assert not (guest.store / "STARTED").exists()
    assert not (guest.tmp / "runs/run-a/ran-job").exists()
    assert not any(c[:3] == ["gcloud", "storage", "rm"] for c in guest.calls())
    assert not any(c[:3] == ["gcloud", "storage", "rsync"] for c in guest.calls())
    assert any(c[0] == "shutdown" for c in guest.calls())
    # A platform hook following this power-off must not publish another result.
    before = len(guest.calls())
    assert guest.run("--preempt").returncode == 0
    assert not any(c[0] == "gcloud" for c in guest.calls()[before:])


def test_same_job_done_powers_off_without_fetching_or_archiving(guest):
    previous = "label=old\njob_uri=gs://test-bucket/job.sh\ntime=2026-11-13T00:00:00Z\n"
    (guest.store / "DONE").write_text(previous)
    result = guest.run()
    assert result.returncode == 0, result.stderr
    assert (guest.store / "DONE").read_text() == previous
    assert not (guest.store / "STARTED").exists()
    assert not any(c[:3] == ["gcloud", "storage", "cp"] for c in guest.calls())
    assert any(c[0] == "shutdown" for c in guest.calls())


def test_same_job_preempted_marker_is_archived_on_resume(guest):
    local = guest.tmp / "runs/run-a/state"
    local.mkdir(parents=True)
    (local / "checkpoint").write_text("preempted")
    assert guest.run("--preempt").returncode == 0
    previous = (guest.store / "PREEMPTED").read_text()
    guest.job.write_text(
        '[ "$(cat "$RUN_DIR/state/checkpoint")" = preempted ] || exit 9\n'
        'printf resumed > "$RUN_DIR/state/checkpoint"\n'
    )
    result = guest.run(label="run-b")
    assert result.returncode == 0, result.stderr
    archived = list((guest.store / "history").glob("*-run-b-PREEMPTED"))
    assert len(archived) == 1 and archived[0].read_text() == previous
    assert not (guest.store / "PREEMPTED").exists()
    assert (guest.store / "checkpoint").read_text() == "resumed"
    assert_marker_identity(guest.store / "DONE", "run-b")


def test_restore_excludes_history_logs_and_all_marker_objects(guest):
    (guest.store / "history").mkdir()
    (guest.store / "history/old-DONE").write_text("archived result")
    (guest.store / "logs").mkdir()
    (guest.store / "logs/old.log").write_text("old log")
    for key in ("STARTED", "HEARTBEAT", "DONE", "FAILED", "TIMEOUT", "PREEMPTED"):
        (guest.store / key).write_text("job_uri=gs://test-bucket/old.sh\n")
    (guest.store / "checkpoints").mkdir()
    (guest.store / "checkpoints/final").write_text("saved")
    guest.job.write_text(
        '[ "$(cat "$RUN_DIR/state/checkpoints/final")" = saved ] || exit 9\n'
        'printf ran > "$RUN_DIR/ran-job"\n'
    )
    result = guest.run()
    assert result.returncode == 0, result.stderr
    local = guest.tmp / "runs/run-a/state"
    assert sorted(str(p.relative_to(local)) for p in local.rglob("*")) == [
        "checkpoints",
        "checkpoints/final",
    ]
    assert (guest.tmp / "runs/run-a/ran-job").read_text() == "ran"


def test_unreadable_done_fails_closed_without_running_job(guest):
    previous = "job_uri=gs://test-bucket/old.sh\n"
    (guest.store / "DONE").write_text(previous)
    result = guest.run(MARKER_READ_ERROR="yes")
    assert result.returncode != 0
    assert (guest.store / "DONE").read_text() == previous
    assert not (guest.store / "STARTED").exists()
    assert not any(c[:3] == ["gcloud", "storage", "cp"] for c in guest.calls())
