"""Execute the guest script offline with real bash, jobs and GNU timeout."""

import json
import os
import shutil
import subprocess
import sys
import time
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
def lock_upload():
    import fcntl
    lock = Path(os.environ['COUNT']+'.lock').open('a')
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        Path(os.environ['COUNT']+'.overlap').touch()
        raise
    return lock
if name == 'curl':
    url = args[-1]
    if '/upload/storage/v1/' in url:
        if os.environ.get('HANG_API_MARKER'): time.sleep(30)
        if os.environ.get('FAIL_API_MARKER'): sys.exit(22)
        if os.environ.get('API_REDIRECT'):
            sys.stdout.write('302')
            sys.exit(0)
        source = next(a[1:] for a in args if a.startswith('@'))
        from urllib.parse import parse_qs, urlparse
        key = parse_qs(urlparse(url).query)['name'][0].split('/state/', 1)[1]
        target = store/key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)
        if '--write-out' in args: sys.stdout.write('200')
        sys.exit(0)
    if url.endswith('/service-accounts/default/token'):
        print('{"access_token":"offline-token"}')
        sys.exit(0)
    key = url.rsplit('/', 1)[1]
    if key == 'preempted':
        if os.environ.get('REQUIRE_PERIODIC') and not Path(os.environ['COUNT']).exists():
            print('FALSE'); sys.exit(0)
        if os.environ.get('PREEMPT_GATE') and not Path(os.environ['PREEMPT_GATE']).exists():
            print('FALSE'); sys.exit(0)
        if not (Path(os.environ['GCP_RUN_ROOT'])/'run-a/.job-running').exists():
            print('FALSE'); sys.exit(0)
        counter = Path(os.environ['COUNT']+'.metadata')
        n = int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(n+1))
        responses = json.loads(os.environ.get('PREEMPT_RESPONSES', '["FALSE"]'))
        response = responses[min(n, len(responses)-1)]
        if response == 'timeout': time.sleep(30)
        if response == 'error': sys.exit(22)
        print(response)
        sys.exit(0)
    if key == os.environ.get('MISSING_METADATA'):
        print('HTTP 404: metadata not found', file=sys.stderr)
        sys.exit(22)
    print(json.loads(os.environ['METADATA'])[key])
elif name == 'shutdown': pass
elif name == 'install': pass
elif name == 'sleep': time.sleep(0.05)
elif name == 'timeout':
    if os.environ.get('FORCE_JOB_TIMEOUT') and 'bash' in args: sys.exit(124)
    if os.environ.get('FORCE_FINAL_TIMEOUT') and 'rsync' in args and not args[args.index('rsync')+1].startswith('gs://'): sys.exit(124)
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
            upload_lock = lock_upload()
            count = Path(os.environ['COUNT'])
            n = int(count.read_text())+1 if count.exists() else 1
            temporary = count.with_suffix('.tmp')
            temporary.write_text(str(n))
            temporary.replace(count)
            time.sleep(float(os.environ.get('DELAY_UPLOAD', '0')))
            if os.environ.get('HANG_COPY') == 'first' and n == 1:
                import signal
                Path(os.environ['COUNT']+'.blocked').touch()
                while True: signal.pause()
            if os.environ.get('FAIL_COPY') == 'all' or (os.environ.get('FAIL_COPY') == 'first' and n == 1):
                print('synthetic copy failure', file=sys.stderr)
                sys.exit(1)
            shutil.copytree(src, store, dirs_exist_ok=True)
    elif args[:2] == ['storage', 'cp']:
        src, dst = args[2:4]
        if not src.startswith('gs://') and '/state/' in src:
            upload_lock = lock_upload()
            if os.environ.get('HANG_STATE_CP'): time.sleep(30)
            if os.environ.get('REQUIRE_PREEMPT_MARKER'): assert (store/'PREEMPTED').exists()
        if os.environ.get('HANG_LOG') and '/logs/' in dst: time.sleep(30)
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
        'printf checkpoint > "$RUN_DIR/state/checkpoint"\n' + f"{sys.executable} - <<'JOB'\n"
        "import os, time\nfrom pathlib import Path\n"
        "while not Path(os.environ['COUNT']).exists(): time.sleep(0.01)\nJOB\n"
    )
    metadata = {
        "guard-label": "run-a",
        "guard-state-uri": "gs://test-bucket/runs/run-a/state",
        "guard-job-uri": "gs://test-bucket/job.sh",
        "guard-copy-interval": "600",
        "guard-reserved-end": (datetime.now(timezone.utc) + timedelta(seconds=3600)).isoformat(),
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
            seconds=3600,
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
                timeout=60,
            )

        def calls(self):
            return [json.loads(line) for line in calls.read_text().splitlines()]

    g = Guest()

    def start(seconds=3600, **overrides):
        values = dict(metadata)
        values["guard-reserved-end"] = (
            datetime.now(timezone.utc) + timedelta(seconds=seconds)
        ).isoformat()
        return subprocess.Popen(
            ["bash", str(SCRIPT)],
            env=dict(env, METADATA=json.dumps(values), **overrides),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    g.start = start
    g.store, g.job, g.tmp = store, job, tmp_path
    return g


def test_failed_periodic_copy_does_not_kill_job_and_next_copy_recovers(guest):
    # Keep the job active until recovery is observable, independent of scheduling.
    guest.job.write_text(
        'printf checkpoint > "$RUN_DIR/state/checkpoint"\n'
        f"exec {sys.executable} - <<'JOB'\n"
        "import os, time\nfrom pathlib import Path\n"
        "heartbeat = Path(os.environ['STORE']) / 'HEARTBEAT'\n"
        "while not heartbeat.exists(): time.sleep(0.01)\n"
        "assert heartbeat.exists(), 'periodic copy never recovered'\nJOB\n"
    )
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
    result = guest.run(FORCE_JOB_TIMEOUT="1")
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


def test_hook_only_marks_interruption_without_copying_state(guest):
    local = guest.tmp / "runs/run-a/state"
    local.mkdir(parents=True)
    (local / "checkpoint").write_text("resume me")
    result = guest.run("--preempt")
    assert result.returncode == 0, result.stderr
    assert not (guest.store / "checkpoint").exists()
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
    # Publish readiness after the counter write, with the upload lock still held.
    # Exiting on count.exists() could cancel write_text before it wrote any bytes.
    guest.job.write_text(
        'printf checkpoint > "$RUN_DIR/state/checkpoint"\n'
        f"exec {sys.executable} - <<'JOB'\n"
        "import os, time\nfrom pathlib import Path\n"
        "blocked = Path(os.environ['COUNT'] + '.blocked')\n"
        "while not blocked.exists(): time.sleep(0.01)\nJOB\n"
    )
    result = guest.run(hang="first")
    assert result.returncode == 0, result.stderr
    assert (guest.tmp / "count.blocked").exists(), "periodic upload never blocked"
    assert (guest.tmp / "count").read_text() == "2", "final sync did not run after cancellation"
    assert not (guest.tmp / "count.overlap").exists(), "final sync overlapped the blocked upload"
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
(state / 'tick').write_text(str(time.time()))
(state / '.pid').write_text(str(os.getpid()))
(state / '.pid').replace(state / 'pid')
while True:
    (state / 'tick').write_text(str(time.time()))
    time.sleep(0.02)
PY
"""
    )
    process = guest.start()
    pidfile = guest.tmp / "runs/run-a/state/pid"
    deadline = time.monotonic() + 60
    pid = None
    try:
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pidfile.exists()
        pid = int(pidfile.read_text())
        process.terminate()
        process.communicate(timeout=60)
        assert process.returncode == 143
        tick = pidfile.with_name("tick")
        before = tick.read_text()
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert tick.read_text() == before, "job still writes after final copy"
        assert (guest.store / "FAILED").exists()
    finally:
        if process.poll() is None:
            process.terminate()
        if pid:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=60)


def test_preemption_marker_upload_is_bounded(control):
    ns = control.ns
    budgets = []

    def command(argv, budget, *_, **__):
        budgets.append(budget)
        control.clock.value += budget
        return None

    ns["command"] = command
    assert not ns["marker"]("PREEMPTED", "hook")
    assert sum(budgets) == 3
    assert not (control.root / "state/checkpoint").exists()


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
    result = guest.run(FORCE_JOB_TIMEOUT="1", fail="all")
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
        DELAY_RESTORE="0",
        GCP_RESTORE_TIMEOUT="20s",
        GCP_MARKER_TIMEOUT="10s",
        GCP_JOB_FETCH_TIMEOUT="11s",
        GCP_PERIODIC_COPY_TIMEOUT="12s",
        GCP_FINAL_COPY_TIMEOUT="13s",
        GCP_LOG_TIMEOUT="14s",
    )
    assert result.returncode == 0, result.stderr
    assert (guest.store / "DONE").exists()
    calls = [c for c in guest.calls() if c[0] == "timeout"]
    for limit in ("20s", "10s", "11s", "12s", "13s", "14s"):
        assert any(limit in c for c in calls), limit


def test_final_copy_enforces_its_override_and_refuses_done_on_timeout(guest):
    guest.job.write_text('printf checkpoint > "$RUN_DIR/state/checkpoint"\n')
    result = guest.run(FORCE_FINAL_TIMEOUT="1", GCP_FINAL_COPY_TIMEOUT="13s")
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
    (guest.store / "checkpoint").write_text("preempted")
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


def watcher_job(guest, cooperate=True, finish="saved"):
    guest.job.write_text(
        f"exec {sys.executable} - <<'JOB'\n"
        + r"""
import os, time
from pathlib import Path
root = Path(os.environ['RUN_DIR'])
state = root/'state'
(state/'old.log').write_text('old')
os.utime(state/'old.log', (1, 1))
print('job running', flush=True)
(root/'.job-running').touch()
while True:
    request = root/'checkpoint-request'
    if COOPERATE and request.exists() and not (root/'checkpoint-done').exists():
        nonce = request.read_text()
        (state/'fresh.ckpt').write_text(nonce)
        (root/'checkpoint-done').write_text(nonce)
    if FINISH == 'saved' and (root/'.preempt-done').exists(): break
    if FINISH == 'request' and request.exists(): break
    if FINISH == 'metadata' and Path(os.environ['COUNT']+'.metadata').exists(): break
    if FINISH == 'periodic' and (Path(os.environ['STORE'])/'HEARTBEAT').exists(): break
    time.sleep(0.01)
print('job completed', flush=True)
JOB
""".replace("COOPERATE", str(cooperate)).replace("FINISH", repr(finish))
    )


def test_watcher_marker_nonce_and_newest_first_once(guest):
    watcher_job(guest)
    result = guest.run(PREEMPT_RESPONSES='["TRUE"]', REQUIRE_PREEMPT_MARKER="1")
    assert result.returncode == 0, result.stderr
    marker = (guest.store / "PREEMPTED").read_text()
    assert "source=watcher" in marker
    assert_marker_identity(guest.store / "PREEMPTED")
    nonce = (guest.tmp / "runs/run-a/checkpoint-request").read_text()
    assert nonce and (guest.tmp / "runs/run-a/checkpoint-done").read_text() == nonce
    assert (guest.store / "fresh.ckpt").read_text() == nonce
    copies = [
        c[3] for c in guest.calls() if c[:3] == ["gcloud", "storage", "cp"] and "/state/" in c[3]
    ]
    assert copies[0].endswith("fresh.ckpt") and copies[1].endswith("old.log")
    assert not (guest.store / "DONE").exists() and not (guest.store / "FAILED").exists()
    uploads = [c for c in guest.calls() if c[0] == "curl" and "/upload/storage/v1/" in c[-1]]
    assert any("PREEMPTED" in c[-1] for c in uploads)
    assert (guest.store / "logs/run-a.log").exists()


@pytest.mark.parametrize("response", ["FALSE", "", "error", "timeout", "true", "TRUE "])
def test_watcher_requires_literal_true_and_survives_metadata_errors(guest, response):
    watcher_job(guest, finish="metadata")
    result = guest.run(PREEMPT_RESPONSES=json.dumps([response]))
    assert result.returncode == 0, result.stderr
    assert not (guest.store / "PREEMPTED").exists()
    assert (guest.store / "DONE").exists()


def test_watcher_reconnects_after_hiccups_and_ignores_stale_nonce(guest):
    watcher_job(guest, cooperate=False)
    result = guest.run(
        PREEMPT_RESPONSES='["error", "", "FALSE", "TRUE"]', GCP_CHECKPOINT_WAIT="0.2"
    )
    assert result.returncode == 0, result.stderr
    assert (guest.store / "PREEMPTED").exists()
    assert (guest.store / "old.log").exists()
    assert not (guest.tmp / "runs/run-a/checkpoint-done").exists()


def test_watcher_saves_marker_before_hanging_copy_and_still_attempts_log(guest):
    watcher_job(guest, cooperate=False)
    result = guest.run(PREEMPT_RESPONSES='["TRUE"]', GCP_CHECKPOINT_WAIT="0.1", HANG_STATE_CP="1")
    assert result.returncode == 0, result.stderr
    assert (guest.store / "PREEMPTED").exists()
    assert not (guest.store / "DONE").exists()
    assert (guest.store / "logs/run-a.log").exists()


def test_watcher_all_storage_calls_hang_but_total_save_is_bounded(guest):
    watcher_job(guest, cooperate=False, finish="request")
    result = guest.run(
        PREEMPT_RESPONSES='["TRUE"]',
        GCP_CHECKPOINT_WAIT="0",
        HANG_API_MARKER="1",
        hang_marker="1",
        HANG_STATE_CP="1",
    )
    assert result.returncode == 0, result.stderr
    assert not (guest.store / "DONE").exists()


def test_finished_job_is_not_mislabelled_by_late_metadata(guest):
    guest.job.write_text('touch "$RUN_DIR/.job-running"\nexit 0\n')
    result = guest.run(PREEMPT_RESPONSES='["TRUE"]')
    assert result.returncode == 0, result.stderr
    assert (guest.store / "DONE").exists()
    assert not (guest.store / "PREEMPTED").exists()


def test_job_finishing_during_checkpoint_wait_publishes_actual_outcome(guest):
    watcher_job(guest, cooperate=False, finish="request")
    result = guest.run(PREEMPT_RESPONSES='["TRUE"]')
    assert result.returncode == 0, result.stderr
    assert (guest.store / "DONE").exists(), result.stderr
    assert "source=watcher" in (guest.store / "DONE").read_text()
    assert not (guest.store / "PREEMPTED").exists()
    assert len(list((guest.store / "history").glob("*PREEMPTED"))) == 1
    assert (guest.store / "old.log").read_text() == "old"


def test_hook_cannot_overwrite_a_watcher_marker(guest):
    watcher_job(guest)
    result = guest.run(PREEMPT_RESPONSES='["TRUE"]')
    assert result.returncode == 0, result.stderr
    before = (guest.store / "PREEMPTED").read_text()
    assert guest.run("--preempt").returncode == 0
    assert (guest.store / "PREEMPTED").read_text() == before


def test_stale_checkpoint_acknowledgement_does_not_skip_wait(control, monkeypatch):
    ns, root = control.ns, control.root
    (root / "checkpoint-done").write_text("stale")
    (root / "state/old.log").write_text("old")
    copied_at = []
    monkeypatch.setenv("GCP_CHECKPOINT_WAIT", "0.3")
    ns["marker"] = lambda *_: True
    ns["upload"] = lambda *_: True

    def cloud(*_):
        copied_at.append(control.clock.value)
        return True

    ns["cloud"] = cloud
    ns["preempt"](24)
    assert copied_at == [pytest.approx(0.3)]
    assert (root / "checkpoint-request").read_text().strip() != "stale"


@pytest.mark.parametrize("preempt", [False, True])
def test_periodic_copies_never_overlap_each_other_or_final_save(guest, preempt):
    watcher_job(guest, cooperate=False, finish="saved" if preempt else "periodic")
    responses = '["FALSE", "FALSE", "TRUE"]' if preempt else '["FALSE"]'
    result = guest.run(PREEMPT_RESPONSES=responses, GCP_CHECKPOINT_WAIT="0", REQUIRE_PERIODIC="1")
    assert result.returncode == 0, result.stderr
    assert not (guest.tmp / "count.overlap").exists()
    assert (guest.tmp / "count").exists(), "periodic copy was never exercised"
    assert (guest.store / "old.log").exists()


def test_total_save_bound_with_every_preemption_upload_hung_and_job_still_running(
    control, monkeypatch
):
    ns, root = control.ns, control.root
    (root / "state/large.ckpt").write_text("checkpoint")
    (root / "job.log").write_text("running")
    monkeypatch.setenv("GCP_CHECKPOINT_WAIT", "8")

    def command(argv, budget, *_, **__):
        control.clock.value += budget
        return None

    ns["command"] = command
    ns["preempt"](24)
    assert control.alarms[0][1] == 24
    assert control.alarms[-1][1] == 0
    assert control.clock.value <= 24
    assert (root / ".preempt-done").exists()
    assert not (root / ".job-ended").exists()


def test_job_exit_status_is_published_atomically_before_reconciliation(control):
    ns, root = control.ns, control.root
    (root / "state/checkpoint").write_text("saved")
    (root / ".job-ended.tmp").write_text("")
    ns["marker"] = lambda *_: True
    ns["upload"] = lambda *_: True
    ns["cloud"] = lambda *_: True
    reconciled = []
    ns["reconcile"] = reconciled.append
    ns["preempt"](24)
    assert reconciled == [], "a partial temporary status must never publish FAILED"
    (root / ".job-ended.tmp").write_text("0\n")
    (root / ".job-ended.tmp").replace(root / ".job-ended")
    ns["preempt"](24)
    assert reconciled == ["0"]


@pytest.mark.parametrize("fault", ["API_REDIRECT", "FAIL_API_MARKER"])
def test_fast_marker_requires_http_200_and_falls_back_to_gcloud(guest, fault):
    watcher_job(guest)
    result = guest.run(PREEMPT_RESPONSES='["TRUE"]', **{fault: "1"})
    assert result.returncode == 0, result.stderr
    assert (guest.store / "PREEMPTED").exists()
    assert "source=watcher" in (guest.store / "PREEMPTED").read_text()
    assert any(
        c[:3] == ["gcloud", "storage", "cp"] and c[4].endswith("/PREEMPTED") for c in guest.calls()
    )


def test_hook_first_cannot_prevent_watcher_checkpoint_and_state_save(guest):
    watcher_job(guest)
    gate = guest.tmp / "preempt-gate"
    process = guest.start(PREEMPT_RESPONSES='["TRUE"]', PREEMPT_GATE=str(gate))
    run = guest.tmp / "runs/run-a"
    try:
        deadline = time.monotonic() + 60
        while not (run / ".job-running").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (run / ".job-running").exists()
        hook = guest.run("--preempt")
        assert hook.returncode == 0, hook.stderr
        assert "source=hook" in (guest.store / "PREEMPTED").read_text()
        gate.touch()
        _, stderr = process.communicate(timeout=60)
        assert process.returncode == 0, stderr
        assert (run / "checkpoint-request").exists(), stderr
        assert (guest.store / "fresh.ckpt").read_text() == (run / "checkpoint-request").read_text()
        assert "source=watcher" in (guest.store / "PREEMPTED").read_text()
        assert (guest.store / "logs/run-a.log").exists()
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=60)


@pytest.fixture
def control(guest, monkeypatch):
    """Load the guest-local helper with isolated clocks and no monitor thread."""
    import signal
    import types

    root = guest.tmp / "runs/run-a"
    (root / "state").mkdir(parents=True)
    for key, value in {
        "RUN_DIR": str(root),
        "STATE_URI": "gs://test-bucket/runs/run-a/state",
        "LABEL": "run-a",
        "JOB_URI": "gs://test-bucket/job.sh",
        "GCP_CHECKPOINT_WAIT": "0",
    }.items():
        monkeypatch.setenv(key, value)
    alarms = []
    monkeypatch.setattr(signal, "signal", lambda *_: None)
    monkeypatch.setattr(signal, "setitimer", lambda *args: alarms.append(args))
    source = SCRIPT.read_text().split("<<'GCP_CONTROL_PY'\n", 1)[1].split("\nGCP_CONTROL_PY", 1)[0]
    namespace = {}
    exec(compile(source.split("\nmode = sys.argv[1]", 1)[0], "control.py", "exec"), namespace)

    class Clock:
        value = 0

        def monotonic(self):
            return self.value

        def sleep(self, seconds):
            self.value += seconds

    clock = Clock()
    namespace["time"] = clock
    return types.SimpleNamespace(ns=namespace, root=root, clock=clock, alarms=alarms)


def test_preemption_log_and_trace_survive_exhausted_state_budget(control):
    ns, root = control.ns, control.root
    (root / "job.log").write_text("checkpoint request handled\n")
    (root / "state/large.ckpt").write_text("checkpoint")
    uploads, copies, events = [], [], []

    def command(argv, budget, capture=False, **_):
        if argv[-1].endswith("/token"):
            return b'{"access_token":"offline-token"}'
        from urllib.parse import parse_qs, urlparse

        key = parse_qs(urlparse(argv[-1]).query)["name"][0]
        source = Path(next(a[1:] for a in argv if a.startswith("@")))
        events.append(key)
        uploads.append((key, source.read_text(), control.clock.value, budget))
        return b"200"

    def cloud(args, budget):
        events.append("state-copy")
        copies.append((args, control.clock.value, budget))
        (root / "job.log").write_text("last seconds after state copy\n")
        control.clock.value += budget
        return False

    ns["command"], ns["cloud"] = command, cloud
    ns["preempt"](24)
    log = next(u for u in uploads if u[0].endswith("/logs/run-a.log"))
    assert log[1] == "checkpoint request handled\n"
    assert events.index(log[0]) < events.index("state-copy"), (
        "large state must not starve the first log upload"
    )
    trace = [u for u in uploads if u[0].endswith("/logs/run-a.preempt.jsonl")]
    assert trace
    assert [u for u in uploads if u[0].endswith("/logs/run-a.log")][-1][
        1
    ] == "last seconds after state copy\n"
    records = [json.loads(line) for line in trace[-1][1].splitlines()]
    assert all("time" in r and "elapsed_s" in r for r in records)
    assert any(r["step"] == "state_copy" and r["outcome"] == "incomplete" for r in records)
    assert any(r["step"] == "job_log" and r["outcome"] == "uploaded" for r in records)
    assert (root / ".preempt-done").exists()
    assert control.clock.value <= 24
