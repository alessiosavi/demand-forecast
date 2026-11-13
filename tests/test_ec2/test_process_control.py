"""Local cancellation barriers; no cloud commands or transport mocks."""

import ctypes
import importlib.util
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.fixture
def control():
    path = Path(__file__).resolve().parents[2] / "scripts/ec2/ops/process_control.py"
    spec = importlib.util.spec_from_file_location("ec2_process_control", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_drain_retries_unconfirmed_group_kill(control, monkeypatch):
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    killpg = os.killpg
    denied = False

    def deny_once(group, sig):
        nonlocal denied
        if sig == signal.SIGKILL and not denied:
            denied = True
            raise PermissionError("temporary cancellation refusal")
        return killpg(group, sig)

    monkeypatch.setattr(control.os, "killpg", deny_once)
    try:
        control.drain_group(child)
        assert child.poll() is not None
    finally:
        try:
            killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("count", [1, 3])
def test_first_interrupt_during_drain_is_delivered_after_death(control, monkeypatch, signum, count):
    drain = control.drain_group
    previous = {sig: signal.signal(sig, control.interrupted) for sig in control.INTERRUPTS}
    killed = []

    def interrupt_then_drain(child):
        for _ in range(count):
            os.kill(os.getpid(), signum)
        drain(child)
        killed.append(child.poll() is not None)

    monkeypatch.setattr(control, "drain_group", interrupt_then_drain)
    try:
        with pytest.raises(InterruptedError):
            control.run_process([sys.executable, "-c", "pass"], 30)
        assert killed == [True]
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def test_failed_term_preserves_timeout_after_group_death(control, monkeypatch):
    killpg = os.killpg
    groups = []

    def deny_term(group, sig):
        if sig == signal.SIGTERM:
            groups.append(group)
            raise PermissionError("graceful signal refused")
        return killpg(group, sig)

    monkeypatch.setattr(control.os, "killpg", deny_term)
    with pytest.raises(subprocess.TimeoutExpired):
        control.run_process([sys.executable, "-c", "import time; time.sleep(60)"], 0.1)
    assert groups
    with pytest.raises(ProcessLookupError):
        killpg(groups[0], 0)


def test_successful_shell_exit_still_drains_background_descendant(control):
    script = """
import subprocess, sys
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(child.pid)
"""
    result = control.run_process(
        [sys.executable, "-c", script], 30, stdout=subprocess.PIPE, text=True
    )
    pid = int(result.stdout)
    try:
        assert result.returncode == 0
        from test_ops import process_alive

        alive = process_alive(pid)
        assert not alive, f"Background descendant {pid} outlived its operation"
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class ProbeStopError(Exception):
    """Bound a deliberately unresolved drain without releasing a real session."""


def observe_exit(pid, block=False):
    if hasattr(os, "waitid"):
        flags = os.WEXITED | os.WNOWAIT | (0 if block else os.WNOHANG)
        return os.waitid(os.P_PID, pid, flags) is not None
    info = (ctypes.c_int * 64)()
    waitid = ctypes.CDLL(None, use_errno=True).waitid
    if waitid(1, pid, ctypes.byref(info), 36 | (0 if block else 1)):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return info[3] == pid


def zombie_scan(monkeypatch, control, child):
    """ps is injected; exit/signals/reaping still use the real kernel child."""
    run = subprocess.run

    def inspect(argv, **kwargs):
        if argv[0] == "ps":
            observe_exit(child.pid, block=True)
            # Support both snapshot formats so the red run tests behavior.
            output = f"{child.pid} {child.pid} Z\n" if "pid=" in argv else f"{child.pid} Z\n"
            return subprocess.CompletedProcess(argv, 0, output, "")
        return run(argv, **kwargs)

    monkeypatch.setattr(control.subprocess, "run", inspect)

    def members(_group):
        observe_exit(child.pid, block=True)
        return []

    monkeypatch.setattr(control, "group_members", members, raising=False)


@pytest.mark.parametrize("mode", ["success", "term-exit", "grace-expiry"])
def test_group_signals_and_scans_keep_original_leader_unreaped(control, monkeypatch, mode):
    popen, killpg = subprocess.Popen, os.killpg
    children, observations = [], []

    def spawn(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        zombie_scan(monkeypatch, control, child)
        return child

    def pinned_signal(group, sig):
        try:
            observe_exit(group)
        except ChildProcessError:
            pytest.fail(f"Signal {sig} targets a reaped/recyclable group {group}")
        observations.append(sig)
        return killpg(group, sig)

    monkeypatch.setattr(control.subprocess, "Popen", spawn)
    monkeypatch.setattr(control.os, "killpg", pinned_signal)
    script = {
        "success": "pass",
        "term-exit": "import time; time.sleep(60)",
        "grace-expiry": (
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
        ),
    }[mode]
    try:
        if mode != "success":
            with pytest.raises(subprocess.TimeoutExpired):
                control.run_process([sys.executable, "-c", script], 0.1)
        else:
            assert control.run_process([sys.executable, "-c", script], 5).returncode == 0
        assert signal.SIGKILL in observations
        assert children[0].returncode is not None
        with pytest.raises(ChildProcessError):
            observe_exit(children[0].pid)
    finally:
        for child in children:
            if child.returncode is None:
                child.kill()
                child.wait()


@pytest.mark.parametrize("boundary", ["spawn", "raising"])
def test_interrupt_at_deferral_transition_cannot_skip_drain(control, monkeypatch, boundary):
    popen, drain = subprocess.Popen, control.drain_group
    children, drained = [], []
    fired = False
    previous = {sig: signal.signal(sig, control.interrupted) for sig in control.INTERRUPTS}

    def spawn(*args, **kwargs):
        nonlocal fired
        child = popen(*args, **kwargs)
        children.append(child)
        zombie_scan(monkeypatch, control, child)
        if boundary == "spawn":
            fired = True
            os.kill(os.getpid(), signal.SIGINT)
        return child

    def record_drain(child):
        drained.append(child.pid)
        drain(child)

    def trace(frame, event, _arg):
        nonlocal fired
        if (
            not fired
            and boundary == "raising"
            and children
            and event == "call"
            and frame.f_code.co_filename == control.__file__
            and frame.f_code.co_name in ("raising", "deferred_interrupts")
        ):
            fired = True
            os.kill(os.getpid(), signal.SIGINT)
        return trace

    monkeypatch.setattr(control.subprocess, "Popen", spawn)
    monkeypatch.setattr(control, "drain_group", record_drain)
    try:
        sys.settrace(trace)
        with pytest.raises(InterruptedError):
            control.run_process([sys.executable, "-c", "pass"], 5)
        assert fired
        assert drained == [children[0].pid]
        assert children[0].returncode is not None
    finally:
        sys.settrace(None)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        for child in children:
            if child.returncode is None:
                child.kill()
                child.wait()


def test_zombie_only_eperm_reaches_reap(control, monkeypatch):
    child = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    attempts = []
    real_refusals = []
    killpg = os.killpg
    try:
        observe_exit(child.pid, block=True)
        zombie_scan(monkeypatch, control, child)

        def zombie_signal(group, sig):
            if group != child.pid:
                return killpg(group, sig)
            try:
                killpg(group, sig)
            except PermissionError:
                real_refusals.append(sig)
            attempts.append(sig)
            if len(attempts) == 3:
                raise ProbeStopError("Never reached reap of zombie-only group")
            raise PermissionError("zombie-only group")

        monkeypatch.setattr(control.os, "killpg", zombie_signal)
        control.drain_group(child)
        assert attempts == [signal.SIGKILL]
        assert child.returncode == 0
        if sys.platform == "darwin":
            assert real_refusals == [signal.SIGKILL]
    finally:
        child.wait()


def test_stale_scope_cannot_select_nested_path(control, monkeypatch):
    popen, drain = subprocess.Popen, control.drain_group
    children, drained = [], []

    def spawn(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        zombie_scan(monkeypatch, control, child)
        return child

    def record_drain(child):
        drained.append(child.pid)
        drain(child)

    monkeypatch.setenv(control.SCOPE, str(os.getpgrp()))
    monkeypatch.setattr(control.subprocess, "Popen", spawn)
    monkeypatch.setattr(control, "drain_group", record_drain)
    result = control.run_process([sys.executable, "-c", "pass"], 5)
    assert result.returncode == 0
    assert drained == [children[0].pid]


@pytest.mark.parametrize("members", [["456(S)"], None], ids=["live", "unknown"])
def test_unconfirmed_drain_reports_periodically_and_does_not_reap(
    control, monkeypatch, capsys, members
):
    class Child:
        pid = 456
        returncode = None

        def wait(self):
            pytest.fail("Unconfirmed death must retain the session")

    elapsed, sleeps = 0, 0

    def sleep(_seconds):
        nonlocal elapsed, sleeps
        elapsed += 6
        sleeps += 1
        if sleeps == 4:
            raise ProbeStopError()

    def denied(*_args):
        raise PermissionError("signal refused")

    monkeypatch.setattr(control, "exited", lambda _child: False, raising=False)
    monkeypatch.setattr(control.os, "killpg", denied)
    monkeypatch.setattr(control, "group_members", lambda _group: members, raising=False)
    monkeypatch.setattr(control.time, "monotonic", lambda: elapsed)
    monkeypatch.setattr(control.time, "sleep", sleep)
    with pytest.raises(ProbeStopError):
        control.drain_group(Child())
    err = capsys.readouterr().err
    assert err.count("Waiting for local process group 456") == 3
    assert "elapsed=6.0s" in err and "elapsed=18.0s" in err
    assert "session retained" in err
    assert ("456(S)" if members else "unknown (inspection failed)") in err
    assert "signal refused" in err


@pytest.mark.parametrize("text", [False, True])
def test_pinned_capture_drains_both_full_pipes(control, monkeypatch, text):
    popen = subprocess.Popen

    def spawn(*args, **kwargs):
        child = popen(*args, **kwargs)
        zombie_scan(monkeypatch, control, child)
        return child

    monkeypatch.setattr(control.subprocess, "Popen", spawn)
    script = (
        "import sys; "
        "sys.stdout.buffer.write(b'x'*262144+b'\\r\\n'); "
        "sys.stderr.buffer.write(b'y'*262144+b'\\r\\n')"
    )
    result = control.run_process(
        [sys.executable, "-c", script],
        5,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    )
    assert result.returncode == 0
    assert result.stdout == ("x" * 262144 + "\n" if text else b"x" * 262144 + b"\r\n")
    assert result.stderr == ("y" * 262144 + "\n" if text else b"y" * 262144 + b"\r\n")


@pytest.mark.parametrize("shell", [False, True], ids=["exec", "shell-substitution"])
def test_live_lease_keeps_nested_transport_in_same_group(control, monkeypatch, shell):
    popen = subprocess.Popen
    children = []

    def spawn(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        zombie_scan(monkeypatch, control, child)
        return child

    monkeypatch.setattr(control.subprocess, "Popen", spawn)
    script = f"""
import os, subprocess, sys
sys.path.insert(0, {str(Path(control.__file__).parent)!r})
import process_control as control
assert control.scope_fd() is not None, 'Owner lease did not survive exec'
result = control.run_process([sys.executable, '-c', 'import os; print(os.getpgrp())'],
                             5, stdout=subprocess.PIPE, text=True)
assert result.returncode == 0
print(os.getpgrp(), result.stdout.strip())
"""
    command = [sys.executable, "-c", script]
    if shell:
        command = [
            "bash",
            "-c",
            'reply=$("$1" -c "$2") || exit $?; printf "%s\\n" "$reply"',
            "lease-test",
            sys.executable,
            script,
        ]
    result = control.run_process(command, 10, stdout=subprocess.PIPE, text=True)
    assert result.returncode == 0
    assert result.stdout.split() == [str(children[0].pid)] * 2


def test_owner_lease_requires_live_writer_and_matching_pipe(control, monkeypatch):
    read_fd, write_fd = os.pipe()
    try:
        os.set_blocking(read_fd, False)
        info = os.fstat(read_fd)
        marker = f"{os.getpgrp()}:{read_fd}:{info.st_dev}:{info.st_ino}"
        monkeypatch.setenv(control.SCOPE, marker)
        assert control.scope_fd() == read_fd
        monkeypatch.setenv(control.SCOPE, marker + "0")
        assert control.scope_fd() is None  # Recycled/mismatched descriptor.
        monkeypatch.setenv(control.SCOPE, marker)
        os.close(write_fd)
        write_fd = None
        assert control.scope_fd() is None  # Exact pipe, but the owner is gone.
    finally:
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)


@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [
        ("123 123 Z\n124 123 Z+\n999 999 S\n", []),
        ("123 123 Z\n124 123 S+\n", ["124(S+)"]),
        ("124 123 Z\n", None),
        ("123 123 Z\nmalformed\n", None),
        ("", None),
    ],
)
def test_membership_scan_requires_leader_and_valid_snapshot(
    control, monkeypatch, snapshot, expected
):
    def inspect(argv, **kwargs):
        assert argv == ["ps", "-A", "-o", "pid=", "-o", "pgid=", "-o", "stat="]
        assert kwargs["timeout"] == 1
        return subprocess.CompletedProcess(argv, 0, snapshot, "")

    monkeypatch.setattr(control.subprocess, "run", inspect)
    monkeypatch.setattr(control, "native_group_members", lambda _group: None, raising=False)
    assert control.group_members(123) == expected


@pytest.mark.parametrize("capture", [False, True])
def test_signal_during_selector_wait_reaches_caller(control, monkeypatch, capture):
    # selectors catch InterruptedError, even when our signal handler raised it.
    # Deliver inside the kernel wait, rather than before/after the I/O boundary.
    factory = selectors.DefaultSelector
    armed = False

    def selector():
        nonlocal armed
        result = factory()
        select = result.select

        def wait(seconds=None):
            nonlocal armed
            if not armed:
                armed = True
                signal.setitimer(signal.ITIMER_REAL, 0.02)
                return select(0.2)
            return select(seconds)

        result.select = wait
        return result

    popen = subprocess.Popen

    def spawn(*args, **kwargs):
        child = popen(*args, **kwargs)
        zombie_scan(monkeypatch, control, child)
        return child

    previous = {sig: signal.signal(sig, control.interrupted) for sig in control.INTERRUPTS}
    alarm = signal.signal(signal.SIGALRM, lambda *_args: os.kill(os.getpid(), signal.SIGINT))
    monkeypatch.setattr(selectors, "DefaultSelector", selector)
    monkeypatch.setattr(control.subprocess, "Popen", spawn)
    try:
        with pytest.raises(InterruptedError):
            control.run_process(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                0.3,
                stdout=subprocess.PIPE if capture else None,
            )
        assert armed
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, alarm)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def test_native_membership_includes_live_and_excludes_zombie_without_ps(control, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Native membership inspection must not execute ps")

    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    monkeypatch.setattr(control.subprocess, "run", forbidden)
    try:
        members = control.group_members(child.pid)
        assert members is not None
        assert any(member.startswith(f"{child.pid}(") for member in members)
        child.kill()
        observe_exit(child.pid, block=True)
        assert control.group_members(child.pid) == []
        assert child.returncode is None  # Inspection did not reap the pin.
    finally:
        child.kill()
        child.wait()


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ({123: "Z 1 123", 124: "S 123 123", 999: "R 1 999"}, ["124(S)"]),
        ({123: "Z 1 123", 124: "Z 1 123"}, []),
        ({124: "Z 1 123"}, None),
        ({123: "Z 1 123", 124: "malformed"}, None),
    ],
)
def test_linux_membership_handles_comm_and_requires_complete_scan(
    control, monkeypatch, tmp_path, rows, expected
):
    for pid, fields in rows.items():
        entry = tmp_path / str(pid)
        entry.mkdir()
        (entry / "stat").write_text(f"{pid} (name with ) parentheses) {fields}\n")
    monkeypatch.setattr(control.sys, "platform", "linux")
    monkeypatch.setattr(control, "Path", lambda _path: tmp_path)
    assert control.native_group_members(123) == expected


def test_drain_diagnostics_bypass_captured_child_stderr(control, monkeypatch, capsys):
    scan = control.group_members
    started = time.monotonic()

    def temporarily_unknown(group):
        return None if time.monotonic() - started < 0.2 else scan(group)

    monkeypatch.setattr(control, "group_members", temporarily_unknown)
    monkeypatch.setattr(control, "DRAIN_DIAGNOSTIC_SECONDS", 0.02)
    result = control.run_process(
        [sys.executable, "-c", "pass"], 2, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert result.returncode == 0
    assert result.stderr == b""
    err = capsys.readouterr().err
    assert "Waiting for local process group" in err
    assert "unknown (inspection failed)" in err
    assert "session retained" in err


def test_auto_reaping_sigchld_is_rejected_before_spawn(control, monkeypatch):
    def spawn(*_args, **_kwargs):
        pytest.fail("Must reject auto-reaping before owning any child")

    previous = signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    monkeypatch.setattr(control.subprocess, "Popen", spawn)
    try:
        with pytest.raises(ValueError, match="SIGCHLD"):
            control.run_process([sys.executable, "-c", "pass"], 5)
    finally:
        signal.signal(signal.SIGCHLD, previous)


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("count", [1, 3])
def test_interrupt_deferral_stays_active_until_reap(control, monkeypatch, signum, count):
    popen = subprocess.Popen

    def spawn(*args, **kwargs):
        child = popen(*args, **kwargs)
        zombie_scan(monkeypatch, control, child)
        return child

    monkeypatch.setattr(control.subprocess, "Popen", spawn)
    test_first_interrupt_during_drain_is_delivered_after_death(control, monkeypatch, signum, count)


def test_lost_leader_identity_forbids_signals_and_inspection(control, monkeypatch, capsys):
    class Child:
        pid = 456
        returncode = 0  # Already reaped: never touch this group again.

        def wait(self):
            pytest.fail("Lost ownership must retain the session")

    elapsed = 0

    def sleep(_seconds):
        nonlocal elapsed
        elapsed += 6
        if elapsed == 12:
            raise ProbeStopError()

    def forbidden(*_args):
        pytest.fail("Cannot signal or inspect an unpinned group")

    monkeypatch.setattr(control.os, "killpg", forbidden)
    monkeypatch.setattr(control, "group_members", forbidden)
    monkeypatch.setattr(control.time, "monotonic", lambda: elapsed)
    monkeypatch.setattr(control.time, "sleep", sleep)
    with pytest.raises(ProbeStopError):
        control.drain_group(Child())
    err = capsys.readouterr().err
    assert "leader identity unavailable" in err
    assert "session retained" in err


def test_broken_diagnostic_pipe_cannot_release_session(control, monkeypatch):
    class Child:
        pid = 456
        returncode = None

        def wait(self):
            pytest.fail("Unconfirmed death must retain the session")

    class ClosedPipe:
        def write(self, _data):
            raise BrokenPipeError("diagnostic reader exited")

    elapsed = 0

    def sleep(_seconds):
        nonlocal elapsed
        elapsed += 6
        if elapsed == 12:
            raise ProbeStopError()

    monkeypatch.setattr(control, "exited", lambda _child: False)
    monkeypatch.setattr(control.os, "killpg", lambda *_args: None)
    monkeypatch.setattr(control, "group_members", lambda _group: ["456(S)"])
    monkeypatch.setattr(control.time, "monotonic", lambda: elapsed)
    monkeypatch.setattr(control.time, "sleep", sleep)
    monkeypatch.setattr(control.sys, "stderr", ClosedPipe())
    with pytest.raises(ProbeStopError):
        control.drain_group(Child())
