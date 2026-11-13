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

STUB = r"""
import json, os, shutil, sys, time
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
with Path(os.environ['CALLS']).open('a') as out: out.write(json.dumps([name, *args])+'\n')
store = Path(os.environ['STORE'])
if name == 'curl':
    key = args[-1].rsplit('/', 1)[1]
    print(json.loads(os.environ['METADATA'])[key])
elif name == 'shutdown': pass
elif name == 'install': pass
elif name == 'sleep': time.sleep(0.05)
elif name == 'timeout':
    os.execv(os.environ['REAL_TIMEOUT'], [os.environ['REAL_TIMEOUT'], *args])
elif name == 'gcloud':
    assert '--format=json' in args
    if args[:3] == ['storage', 'objects', 'describe']:
        if (store/'DONE').exists(): print('{}')
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
            for p in store.iterdir():
                if p.is_file() and not (exclude and re.search(exclude, p.name)):
                    shutil.copy(p, Path(dst)/p.name)
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
        if src.startswith('gs://'): shutil.copy(os.environ['JOB'], dst)
        else:
            target = store/dst.split('/state/', 1)[1]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, target)
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
        def run(self, *args, fail="", seconds=610, hang="", hang_marker="", **overrides):
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


def test_job_timeout_writes_timeout_and_powers_off(guest):
    guest.job.write_text(f"{sys.executable} -c 'import time; time.sleep(20)'\n")
    result = guest.run(seconds=602)
    assert result.returncode != 0
    assert (guest.store / "TIMEOUT").exists(), result.stderr
    assert not (guest.store / "DONE").exists()
    assert any(c[0] == "shutdown" for c in guest.calls())


def test_done_never_fetches_or_runs_job(guest):
    (guest.store / "DONE").touch()
    result = guest.run()
    assert result.returncode == 0, result.stderr
    assert not any(c[:3] == ["gcloud", "storage", "cp"] for c in guest.calls())
    assert any(c[0] == "shutdown" for c in guest.calls())


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
    assert not any(c[0] == "shutdown" for c in guest.calls())


def test_job_failure_preserves_status_and_preemption_hook_ignores_normal_end(guest):
    guest.job.write_text("echo failing-job\nexit 7\n")
    result = guest.run()
    assert result.returncode == 7, result.stderr
    assert "exit_status=7" in (guest.store / "FAILED").read_text()
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
