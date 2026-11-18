"""Exercise monitor signals in real processes, without GNU date or timeout."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/gcp/guest-run.sh"
DEADLOCK_GUARD = 120  # Only turns a hang into a failure; no elapsed-time assertions.

DOUBLE = r"""
import os
import signal
import sys
import time
from pathlib import Path

name = Path(sys.argv[0]).name
if name == "timeout":
    args = sys.argv[1:]
    while args[0].startswith("-"):
        args.pop(0)
    os.execvp(args[1], args[1:])
elif name == "curl":
    url = sys.argv[-1]
    injection = os.environ["INJECTION"]
    if (
        (injection in ("wait", "double-term") and url.endswith("preempted"))
        or (injection in ("alarm", "term-alarm", "hook") and url.endswith("PREEMPTED"))
    ):
        # Let communicate() reach waitpid while this child is still alive.
        os.close(1)
        while True:
            signal.pause()
    elif url.endswith("preempted"):
        if injection in ("event", "cleanup-killpg"):
            # Keep the poll loop alive until there is a periodic child to kill.
            ready = Path(os.environ["RUN_DIR"]).parent / "periodic-ready"
            while not ready.exists():
                time.sleep(0.001)
        print("TRUE")
    elif url.endswith("token"):
        print('{"access_token":"test-token"}')
    else:
        print("200", end="")
elif name == "sleep":
    (Path(os.environ["RUN_DIR"]).parent / "periodic-ready").touch()
    while True:
        signal.pause()
elif name == "gcloud":
    if os.environ["INJECTION"] == "wait-acquire":
        while True:
            signal.pause()
else:
    raise AssertionError(name)
"""

RUNNER = r"""
import os
import runpy
import signal
import subprocess
import sys
import threading
import traceback
from pathlib import Path

work = Path(sys.argv[1])
injection = os.environ["INJECTION"]
fired = False
original_init = subprocess.Popen.__init__
original_wait = subprocess.Popen._try_wait
original_set = threading.Event.set
original_killpg = os.killpg
original_setitimer = signal.setitimer

def record_injection():
    global fired
    fired = True
    (work / "injected").touch()

def interrupt(signum, followup=None):
    record_injection()
    try:
        os.kill(os.getpid(), signum)
    finally:
        if followup is not None:
            os.kill(os.getpid(), followup)
    (work / "handler-returned").touch()

class InterruptAfterAcquire:
    def __init__(self, lock):
        self.lock = lock

    def acquire(self, blocking=True):
        acquired = self.lock.acquire(blocking)
        if acquired and not blocking and not fired:
            assert threading.current_thread() is threading.main_thread()
            interrupt(signal.SIGTERM)
        return acquired

    def __getattr__(self, name):
        return getattr(self.lock, name)

def remember_child(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    (work / "pids" / str(self.pid)).touch()
    if injection == "wait-acquire" and self.args[0] == "gcloud":
        self._waitpid_lock = InterruptAfterAcquire(self._waitpid_lock)

def interrupt_wait(self, flags):
    if threading.current_thread() is threading.main_thread() and not fired:
        url = self.args[-1]
        if (
            (injection in ("wait", "double-term") and url.endswith("preempted"))
            or (injection in ("alarm", "term-alarm", "hook") and url.endswith("PREEMPTED"))
        ):
            assert self._waitpid_lock.locked()
            if injection in ("alarm", "term-alarm"):
                assert signal.getsignal(signal.SIGALRM).__name__ == "expired"
                if injection == "alarm":
                    interrupt(signal.SIGALRM)
                else:
                    interrupt(signal.SIGTERM, signal.SIGALRM)
            elif injection == "double-term":
                interrupt(signal.SIGTERM, signal.SIGTERM)
            else:
                interrupt(signal.SIGTERM)
    return original_wait(self, flags)

def interrupt_set(self):
    caller = sys._getframe(1)
    in_preempt = caller.f_code.co_name == "preempt"
    if (
        ((injection == "event" and not in_preempt) or
         (injection == "preempt-event" and in_preempt))
        and self is caller.f_globals.get("stop")
        and threading.current_thread() is threading.main_thread()
        and not fired
    ):
        with self._cond:
            # Match Event.set's non-reentrant critical section exactly.
            interrupt(signal.SIGTERM)
            self._flag = True
            self._cond.notify_all()
        return
    return original_set(self)

def interrupt_killpg(pgid, signum):
    if injection == "cleanup-killpg" and not fired:
        assert any(frame.name == "cancel_children" for frame in traceback.extract_stack())
        assert (work / "periodic-ready").exists()
        interrupt(signal.SIGTERM)
    return original_killpg(pgid, signum)

def interrupt_disarm(which, seconds):
    if injection in ("cleanup-term", "cleanup-alarm") and seconds == 0 and not fired:
        # Check both handlers are still installed when cleanup disarms the timer.
        handlers = [signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGALRM)]
        names = [getattr(handler, "__name__", str(handler)) for handler in handlers]
        (work / "cleanup-handlers").write_text(" ".join(names))
        signum = signal.SIGTERM if injection == "cleanup-term" else signal.SIGALRM
        interrupt(signum)
    return original_setitimer(which, seconds)

subprocess.Popen.__init__ = remember_child
subprocess.Popen._try_wait = interrupt_wait
threading.Event.set = interrupt_set
os.killpg = interrupt_killpg
signal.setitimer = interrupt_disarm
sys.argv = [str(work / "control.py"), "hook" if injection == "hook" else "monitor"]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def run_monitor(tmp_path, injection, scale):
    source = (
        SCRIPT.read_text().split("<<'GCP_CONTROL_PY'\n", 1)[1].split("\nGCP_CONTROL_PY\n", 1)[0]
    )
    (tmp_path / "control.py").write_text(source)
    runner = tmp_path / "runner.py"
    runner.write_text(RUNNER)
    (tmp_path / "pids").mkdir()
    bins = tmp_path / "bin"
    bins.mkdir()
    for name in ("curl", "sleep", "timeout", "gcloud"):
        executable = bins / name
        executable.write_text(f"#!{sys.executable}\n{DOUBLE}")
        executable.chmod(0o755)
    root = tmp_path / "run"
    (root / "state").mkdir(parents=True)
    if injection == "wait-acquire":
        (root / "state" / "checkpoint").touch()
    (root / ".job-running").touch()
    if injection in ("event", "cleanup-killpg"):
        # Decline the watcher claim and exercise the end of the monitor loop.
        (root / ".job-ended").write_text("0\n")
    env = dict(
        os.environ,
        PATH=f"{bins}:{os.environ['PATH']}",
        RUN_DIR=str(root),
        STATE_URI="gs://test-bucket/runs/run-a/state",
        LABEL="run-a",
        JOB_URI="gs://test-bucket/job.sh",
        COPY_INTERVAL="600",
        GCP_CHECKPOINT_WAIT="0",
        GCP_TEST_BUDGET_SCALE=str(scale),
        INJECTION=injection,
    )
    process = subprocess.Popen(
        [sys.executable, str(runner), str(tmp_path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        try:
            stdout, stderr = process.communicate(timeout=DEADLOCK_GUARD)
        except subprocess.TimeoutExpired:
            pytest.fail(f"{injection} signal deadlocked the monitor (scale={scale})")
        assert (tmp_path / "injected").exists(), "signal injection was never reached"
        assert process.returncode == 0, stdout + stderr
        assert "Traceback" not in stderr, stderr
        pids = [int(path.name) for path in (tmp_path / "pids").iterdir()]
        assert pids, "no monitor children were exercised"
        for pid in pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        return root, stderr
    finally:
        # A broken monitor must not leave the test's doubles running either.
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=DEADLOCK_GUARD)
        for path in (tmp_path / "pids").iterdir():
            try:
                os.kill(int(path.name), signal.SIGKILL)
            except ProcessLookupError:
                pass


def assert_preempt_done(root, stderr, *, expired=False):
    assert (root / ".preempt-done").exists()
    records = [
        json.loads(line) for line in (root / ".preempt-trace.jsonl").read_text().splitlines()
    ]
    assert any(row["step"] == "deadline" and row["outcome"] == "expired" for row in records) == (
        expired
    )
    assert ("24 s save deadline reached" in stderr) == expired


def assert_handler_returned(tmp_path):
    assert (tmp_path / "handler-returned").exists(), (
        "signal handler raised in the interrupted frame"
    )


@pytest.mark.parametrize("scale", [3600, 1])
def test_term_inside_child_wait_leaves_no_children(tmp_path, scale):
    run_monitor(tmp_path, "wait", scale)
    assert_handler_returned(tmp_path)


@pytest.mark.parametrize("scale", [3600, 1])
def test_term_after_wait_lock_acquisition_leaves_no_children(tmp_path, scale):
    """Deliver TERM after acquiring the wait lock, before Popen can protect it."""
    root, stderr = run_monitor(tmp_path, "wait-acquire", scale)
    assert_preempt_done(root, stderr)
    assert_handler_returned(tmp_path)


def test_term_inside_stop_set_does_not_deadlock(tmp_path):
    """TERM inside the normal monitor cleanup's Event.set leaves no periodic child."""
    run_monitor(tmp_path, "event", 1)
    assert_handler_returned(tmp_path)


def test_save_alarm_inside_upload_wait_finishes_and_records_deadline(tmp_path):
    root, stderr = run_monitor(tmp_path, "alarm", 1)
    assert_preempt_done(root, stderr, expired=True)
    assert_handler_returned(tmp_path)


def test_term_inside_monitor_cleanup_killpg_leaves_no_children(tmp_path):
    """TERM inside the first killpg of normal cleanup cannot interrupt cancellation."""
    run_monitor(tmp_path, "cleanup-killpg", 1)
    assert_handler_returned(tmp_path)


@pytest.mark.parametrize("injection", ["cleanup-term", "cleanup-alarm"])
def test_signal_inside_preempt_timer_disarm_finishes_cleanup(tmp_path, injection):
    """Both handlers stay installed during disarm and the cleanup writes its marker."""
    root, stderr = run_monitor(tmp_path, injection, 1)
    assert (tmp_path / "cleanup-handlers").read_text() == "terminate expired"
    assert_preempt_done(root, stderr, expired=injection == "cleanup-alarm")
    assert_handler_returned(tmp_path)


def test_both_term_handlers_return_inside_child_wait(tmp_path):
    run_monitor(tmp_path, "double-term", 3600)
    assert_handler_returned(tmp_path)


def test_alarm_after_term_inside_upload_wait_records_deadline(tmp_path):
    root, stderr = run_monitor(tmp_path, "term-alarm", 1)
    assert_preempt_done(root, stderr, expired=True)
    assert_handler_returned(tmp_path)


def test_term_inside_hook_upload_wait_leaves_no_children(tmp_path):
    run_monitor(tmp_path, "hook", 1)
    assert_handler_returned(tmp_path)


def test_term_inside_preempt_stop_set_does_not_deadlock(tmp_path):
    """TERM returns with the preempt body's Event.set lock still held."""
    root, stderr = run_monitor(tmp_path, "preempt-event", 1)
    assert_preempt_done(root, stderr)
    assert_handler_returned(tmp_path)


def test_shutdown_after_a_finished_command_does_not_signal_its_reaped_group(tmp_path, monkeypatch):
    """A reaped child's pid is free for reuse: once a command has ended, its group is left alone."""
    source = SCRIPT.read_text().split("<<'GCP_CONTROL_PY'\n", 1)[1].split("\nGCP_CONTROL_PY", 1)[0]
    (tmp_path / "state").mkdir()
    for key, value in {
        "RUN_DIR": str(tmp_path),
        "STATE_URI": "gs://test-bucket/runs/run-a/state",
        "LABEL": "run-a",
        "JOB_URI": "gs://test-bucket/job.sh",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(signal, "signal", lambda *_: None)
    namespace = {}
    exec(compile(source.split("\nmode = sys.argv[1]", 1)[0], "control.py", "exec"), namespace)

    class FlagsShutdownWhenDone(subprocess.Popen):
        def communicate(self, *args, **kwargs):
            result = super().communicate(*args, **kwargs)
            namespace["shutdown_requested"] = True  # TERM lands as the command ends.
            return result

    monkeypatch.setattr(subprocess, "Popen", FlagsShutdownWhenDone)
    signalled = []
    monkeypatch.setattr(os, "killpg", lambda pgid, signum: signalled.append(pgid))
    with pytest.raises(SystemExit) as stopped:
        namespace["command"]([sys.executable, "-c", "pass"], 30)
    assert stopped.value.code == 0
    assert signalled == []
    assert not namespace["children"]
