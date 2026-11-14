"""Local cancellation barriers; no cloud commands or transport mocks."""

import ctypes
import importlib.util
import inspect
import os
import select
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from process_identity import kill_owned


@pytest.fixture
def control():
    path = Path(__file__).resolve().parents[2] / "scripts/ec2/ops/process_control.py"
    spec = importlib.util.spec_from_file_location("ec2_process_control", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cleanup_child(child, *, group=False):
    """An unreaped, exclusively waited child pins its PID and group identity."""
    if child.returncode is None:
        try:
            observe_exit(child.pid)
        except ChildProcessError:
            return
        try:
            if group:
                os.killpg(child.pid, signal.SIGKILL)
            else:
                child.kill()
        except ProcessLookupError:
            pass
        type(child).wait(child)


def wait_for_ready(fd):
    assert select.select([fd], [], [], 120)[0], "Child never announced readiness"
    assert os.read(fd, 1) == b"R", "Child exited before announcing readiness"


@pytest.fixture
def readiness():
    pipe = os.pipe()
    try:
        yield pipe
    finally:
        for fd in pipe:
            os.close(fd)


def test_healthy_command_has_no_process_control_stderr(control, capsys):
    result = control.run_process(
        ["/bin/echo", "hello"],
        5,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout == "hello\n"
    assert result.stderr == ""
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("restore", [None, True, False], ids=["default", "restore", "inherit"])
@pytest.mark.parametrize("inherited", [signal.SIG_IGN, signal.SIG_DFL], ids=["ignored", "default"])
def test_exec_wrapper_preserves_popen_sigpipe_behavior(control, restore, inherited):
    argv = ["/bin/bash", "-c", "kill -PIPE $$; printf survived"]
    options = {} if restore is None else {"restore_signals": restore}
    previous = signal.signal(signal.SIGPIPE, inherited)
    try:
        direct = subprocess.run(argv, capture_output=True, **options)
        wrapped = control.run_process(argv, 30, stdout=subprocess.PIPE, **options)
        assert direct.returncode == (
            0 if restore is False and inherited == signal.SIG_IGN else -signal.SIGPIPE
        )
        assert wrapped.returncode == direct.returncode
        assert wrapped.stdout == direct.stdout
    finally:
        signal.signal(signal.SIGPIPE, previous)


def test_nested_killed_child_reap_reports_every_five_seconds(control, monkeypatch, capsys):
    elapsed = 0
    killed = False
    messages = []

    class Child:
        pid = 456
        returncode = None
        stdin = stdout = stderr = None

        def communicate(self, timeout):
            raise subprocess.TimeoutExpired("nested", timeout)

        def terminate(self):
            pass

        def kill(self):
            nonlocal killed
            killed = True

        def wait(self, timeout=None):
            nonlocal elapsed
            if not killed:
                raise subprocess.TimeoutExpired("nested", timeout)
            assert timeout is not None, "Killed nested child must never use an unbounded wait"
            messages.append(capsys.readouterr().err)
            if elapsed == 10:
                raise ProbeStopError()
            elapsed += 5
            raise subprocess.TimeoutExpired("nested", timeout)

    monkeypatch.setattr(control, "scope_fd", lambda: 123)
    monkeypatch.setenv(control.SCOPE, "nested-test")
    monkeypatch.setattr(control.subprocess, "Popen", lambda *_args, **_kwargs: Child())
    monkeypatch.setattr(control.time, "monotonic", lambda: elapsed)
    with pytest.raises(ProbeStopError):
        control.run_process(["nested"], 30)
    report = "".join(messages)
    assert report.count("Waiting to reap killed nested local process 456") == 3
    assert all(f"elapsed={tick:.1f}s" in report for tick in (0, 5, 10))
    assert report.count("session retained") == 3


@pytest.mark.parametrize("saved", [False, True], ids=["queued-only", "already-saved"])
@pytest.mark.parametrize("reverse", [False, True], ids=["int-term", "term-int"])
def test_masked_restore_coalesces_both_signals_before_caller_cleanup(control, saved, reverse):
    # Isolate the old implementation's second raising signal from pytest itself.
    queued = (
        (int(signal.SIGTERM), int(signal.SIGINT))
        if reverse
        else (int(signal.SIGINT), int(signal.SIGTERM))
    )
    script = f"""
import signal, os, sys
sys.path.insert(0, {str(Path(control.__file__).parent)!r})
import process_control as control
deliveries = []
def handler(signum, frame):
    deliveries.append(signum)
    raise InterruptedError('operator')
for sig in control.INTERRUPTS:
    signal.signal(sig, handler)
restore = control.InterruptDeferral.restore
sent = []
def queue(self):
    restore(self)
    for sig in {queued!r}:
        os.kill(os.getpid(), sig)
        sent.append(sig)
control.InterruptDeferral.restore = queue
drain = control.drain_group
def save_then_drain(child):
    if {saved!r}:
        os.kill(os.getpid(), signal.SIGINT)
    drain(child)
control.drain_group = save_then_drain
cleaned = False
try:
    try:
        control.run_process(['/bin/true'], 30)
    except InterruptedError:
        with control.cleanup_signals():
            cleaned = True
except InterruptedError:
    pass
assert len(sent) == 2, sent
assert len(deliveries) == 1, deliveries
assert cleaned, 'Leftover interrupt prevented caller cleanup entry'
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
        cleanup_child(child, group=True)


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("count", [1, 3])
def test_first_interrupt_during_drain_is_delivered_after_death(control, monkeypatch, signum, count):
    drain = control.drain_group
    previous = {sig: signal.signal(sig, control.interrupted) for sig in control.INTERRUPTS}
    killed = []
    sent = []

    def interrupt_then_drain(child):
        for _ in range(count):
            os.kill(os.getpid(), signum)
            sent.append(signum)
        drain(child)
        killed.append(child.poll() is not None)

    monkeypatch.setattr(control, "drain_group", interrupt_then_drain)
    try:
        with pytest.raises(InterruptedError):
            control.run_process([sys.executable, "-c", "pass"], 30)
        assert killed == [True]
        assert len(sent) == count
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


@pytest.mark.parametrize("nested", [False, True], ids=["owner", "nested"])
@pytest.mark.parametrize("failure", ["time limit", "interrupt"])
def test_cancellation_reports_before_signal_and_nested_reap(
    control, monkeypatch, capsys, nested, failure
):
    popen, killpg = subprocess.Popen, os.killpg
    messages, children = [], []
    lease = os.pipe()
    os.set_blocking(lease[0], False)
    info = os.fstat(lease[0])
    if nested:
        monkeypatch.setenv(control.SCOPE, f"{os.getpgrp()}:{lease[0]}:{info.st_dev}:{info.st_ino}")
    else:
        monkeypatch.delenv(control.SCOPE, raising=False)
    previous = {sig: signal.signal(sig, control.interrupted) for sig in control.INTERRUPTS}

    def fail_wait(*_args, **_kwargs):
        if failure == "interrupt":
            os.kill(os.getpid(), signal.SIGINT)
            pytest.fail("Interrupt must leave the guarded wait")
        raise subprocess.TimeoutExpired("command", 30)

    def check_cancellation(group):
        message = capsys.readouterr().err
        assert failure in message.lower()
        assert f"local process group {group}" in message
        if failure == "interrupt":
            assert str(signal.SIGINT.value) in message
        messages.append(message)

    def signal_group(group, sig):
        if sig == signal.SIGTERM:
            check_cancellation(group)
        return killpg(group, sig)

    def spawn(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        if nested:
            terminate, wait = child.terminate, child.wait
            grace_expired = False
            reap_reported = False

            def checked_terminate():
                check_cancellation(os.getpgrp())
                # The nested path names the direct child it ends, not the shared group.
                assert f"terminating nested local process {child.pid}" in messages[-1]
                terminate()

            def checked_wait(timeout=None):
                nonlocal grace_expired, reap_reported
                if not grace_expired:
                    grace_expired = True
                    raise subprocess.TimeoutExpired(child.args, timeout)
                if not reap_reported:
                    message = capsys.readouterr().err
                    assert f"Waiting to reap killed nested local process {child.pid}" in message
                    messages.append(message)
                    reap_reported = True
                return wait(timeout=timeout)

            child.communicate = fail_wait
            child.terminate = checked_terminate
            child.wait = checked_wait
        return child

    monkeypatch.setattr(control.subprocess, "Popen", spawn)
    monkeypatch.setattr(control.os, "killpg", signal_group)
    monkeypatch.setattr(control.PinnedOutput, "wait", fail_wait)
    try:
        expected = InterruptedError if failure == "interrupt" else subprocess.TimeoutExpired
        with pytest.raises(expected):
            control.run_process(["/bin/sleep", "60"], 30, stderr=subprocess.PIPE)
        assert children[0].returncode is not None
        assert "".join(messages).lower().count(failure) == 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        for child in children:
            cleanup_child(child)
        for fd in lease:
            os.close(fd)


def test_successful_shell_exit_still_drains_background_descendant(control):
    script = f"""
import subprocess, sys
sys.path.insert(0, {str(Path(__file__).parent)!r})
from process_identity import process_identity
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(child.pid, process_identity(child.pid))
"""
    result = control.run_process(
        [sys.executable, "-c", script], 30, stdout=subprocess.PIPE, text=True
    )
    raw_pid, identity = result.stdout.split()
    pid = int(raw_pid)
    try:
        assert result.returncode == 0
        from test_ops import process_alive

        alive = process_alive(pid)
        assert not alive, f"Background descendant {pid} outlived its operation"
    finally:
        kill_owned(pid, None if identity == "None" else identity)


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
def test_group_signals_and_scans_keep_original_leader_unreaped(
    control, monkeypatch, mode, readiness
):
    popen, killpg = subprocess.Popen, os.killpg
    children, observations = [], []

    def spawn(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        zombie_scan(monkeypatch, control, child)
        wait_for_ready(readiness[0])
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
    # Install SIGTERM's disposition before acknowledging readiness. The real
    # 0.1s deadline starts only after this handshake, excluding interpreter boot.
    ready = f"import os; os.write({readiness[1]}, b'R'); "
    script = (
        script.replace("time.sleep(60)", ready + "time.sleep(60)")
        if mode != "success"
        else ready + script
    )
    try:
        if mode != "success":
            with pytest.raises(subprocess.TimeoutExpired):
                control.run_process([sys.executable, "-c", script], 0.1, pass_fds=(readiness[1],))
        else:
            assert (
                control.run_process(
                    [sys.executable, "-c", script], 5, pass_fds=(readiness[1],)
                ).returncode
                == 0
            )
        assert signal.SIGKILL in observations
        assert children[0].returncode is not None
        with pytest.raises(ChildProcessError):
            observe_exit(children[0].pid)
    finally:
        for child in children:
            cleanup_child(child)


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
            cleanup_child(child)


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


@pytest.mark.parametrize("case", ["live", "unknown", "refused", "observation-error"])
def test_unconfirmed_drain_reports_periodically_and_does_not_reap(
    control, monkeypatch, capsys, case
):
    class Child:
        pid = 456
        returncode = None

        def wait(self):
            pytest.fail("Unconfirmed death must retain the session")

    elapsed = 0
    ticks = iter([1, 4.9, 5, 6, 9.9, 10, 15])
    messages = []
    members = None if case == "unknown" else ["456(S)"]

    def sleep(_seconds):
        nonlocal elapsed
        message = capsys.readouterr().err
        if elapsed in (0, 5, 10, 15):
            assert message.count("Waiting for local process group 456") == 1
            assert f"elapsed={elapsed:.1f}s" in message
            messages.append(message)
        else:
            assert message == ""
        try:
            elapsed = next(ticks)
        except StopIteration:
            raise ProbeStopError() from None

    def signal_group(*_args):
        if case == "refused":
            raise PermissionError("signal refused")

    def observe(_child):
        if case == "observation-error":
            raise OSError("observation unavailable")
        return False

    monkeypatch.setattr(control, "exited", observe)
    monkeypatch.setattr(control.os, "killpg", signal_group)
    monkeypatch.setattr(control, "group_members", lambda _group: members, raising=False)
    monkeypatch.setattr(control.time, "monotonic", lambda: elapsed)
    monkeypatch.setattr(control.time, "sleep", sleep)
    with pytest.raises(ProbeStopError):
        control.drain_group(Child())
    err = "".join(messages)
    assert err.count("Waiting for local process group 456") == 4
    assert "elapsed=0.0s" in err and "elapsed=15.0s" in err
    assert "session retained" in err
    assert (
        "unknown (inspection failed)" if case in ("unknown", "observation-error") else "456(S)"
    ) in err
    if case == "refused":
        assert "signal refused" in err
    if case == "observation-error":
        assert "observation unavailable" in err


@pytest.mark.parametrize("text", [False, True])
def test_pinned_capture_drains_both_full_pipes(control, monkeypatch, text, readiness):
    popen = subprocess.Popen
    children = []

    def spawn(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        zombie_scan(monkeypatch, control, child)
        wait_for_ready(readiness[0])
        return child

    monkeypatch.setattr(control.subprocess, "Popen", spawn)
    script = (
        f"import os,sys; os.write({readiness[1]}, b'R'); "
        "sys.stdout.buffer.write(b'x'*262144+b'\\r\\n'); "
        "sys.stderr.buffer.write(b'y'*262144+b'\\r\\n')"
    )
    try:
        result = control.run_process(
            [sys.executable, "-c", script],
            5,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=text,
            pass_fds=(readiness[1],),
        )
    finally:
        for child in children:
            cleanup_child(child)
    assert result.returncode == 0
    assert result.stdout == ("x" * 262144 + "\n" if text else b"x" * 262144 + b"\r\n")
    assert result.stderr == ("y" * 262144 + "\n" if text else b"y" * 262144 + b"\r\n")


@pytest.mark.parametrize("shell", [False, True], ids=["exec", "shell-substitution"])
def test_live_lease_keeps_nested_transport_in_same_group(
    control, monkeypatch, capsys, shell, readiness
):
    popen = subprocess.Popen
    children = []

    def spawn(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        zombie_scan(monkeypatch, control, child)
        wait_for_ready(readiness[0])
        return child

    monkeypatch.setattr(control.subprocess, "Popen", spawn)
    script = f"""
import os, select, subprocess, sys
sys.path.insert(0, {str(Path(control.__file__).parent)!r})
import process_control as control
assert control.scope_fd() is not None, 'Owner lease did not survive exec'
read_fd, write_fd = os.pipe()
popen = subprocess.Popen
def spawn(*args, **kwargs):
    child = popen(*args, **kwargs)
    assert select.select([read_fd], [], [], 120)[0], 'Nested child never became ready'
    assert os.read(read_fd, 1) == b'R'
    return child
control.subprocess.Popen = spawn
os.write({readiness[1]}, b'R')
command = [sys.executable, '-c', f'import os; os.write({{write_fd}}, b"R"); print(os.getpgrp())']
try:
    result = control.run_process(command, 5, stdout=subprocess.PIPE, text=True, pass_fds=(write_fd,))
finally:
    os.close(read_fd)
    os.close(write_fd)
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
    try:
        result = control.run_process(
            command,
            10,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            pass_fds=(readiness[1],),
        )
    finally:
        for child in children:
            cleanup_child(child)
    assert result.returncode == 0
    assert result.stdout.split() == [str(children[0].pid)] * 2
    assert result.stderr == ""
    assert capsys.readouterr().err == ""


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
        assert argv == [
            "ps",
            "-A",
            *(["-L"] if sys.platform == "linux" else []),
            "-o",
            "pid=",
            "-o",
            "pgid=",
            "-o",
            "stat=",
        ]
        assert kwargs["timeout"] == 1
        return subprocess.CompletedProcess(argv, 0, snapshot + f"\n{os.getpid()} 999999 S\n", "")

    monkeypatch.setattr(control.subprocess, "run", inspect)
    monkeypatch.setattr(control, "native_group_members", lambda _group: None, raising=False)
    assert control.group_members(123) == (
        None if sys.platform == "linux" and expected == [] else expected
    )


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
                # Model the selector's documented EINTR catch at the exact
                # boundary; no timer or child-startup race is involved.
                try:
                    os.kill(os.getpid(), signal.SIGINT)
                except InterruptedError:
                    return []
            return select(seconds)

        result.select = wait
        return result

    popen = subprocess.Popen

    def spawn(*args, **kwargs):
        child = popen(*args, **kwargs)
        zombie_scan(monkeypatch, control, child)
        return child

    previous = {sig: signal.signal(sig, control.interrupted) for sig in control.INTERRUPTS}
    monkeypatch.setattr(selectors, "DefaultSelector", selector)
    monkeypatch.setattr(control.subprocess, "Popen", spawn)
    try:
        with pytest.raises(InterruptedError):
            control.run_process(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                30,
                stdout=subprocess.PIPE if capture else None,
            )
        assert armed
    finally:
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
        cleanup_child(child)


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
        (entry / f"task/{pid}").mkdir(parents=True)
        (entry / f"task/{pid}/stat").write_text(f"{pid} (task) {fields}\n")
    own = tmp_path / str(os.getpid())
    own.mkdir()
    (own / "stat").write_text(f"{os.getpid()} (self) R 1 999\n")
    monkeypatch.setattr(control.sys, "platform", "linux")
    monkeypatch.setattr(control, "Path", lambda _path: tmp_path)
    assert control.native_group_members(123) == expected


def test_drain_diagnostics_bypass_captured_child_stderr(control, monkeypatch, capsys):
    scan = control.group_members
    scans = 0
    messages = []

    def temporarily_unknown(group):
        nonlocal scans
        scans += 1
        return None if scans == 1 else scan(group)

    def check_before_retry(_seconds):
        message = capsys.readouterr().err
        assert "Waiting for local process group" in message
        assert "unknown (inspection failed)" in message
        assert "session retained" in message
        messages.append(message)

    monkeypatch.setattr(control, "group_members", temporarily_unknown)
    monkeypatch.setattr(control.time, "sleep", check_before_retry)
    result = control.run_process(
        [sys.executable, "-c", "pass"], 30, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert result.returncode == 0
    assert result.stderr == b""
    assert len(messages) == 1
    assert capsys.readouterr().err == ""


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


@pytest.mark.parametrize("reaped_elsewhere", [False, True])
def test_lost_leader_identity_forbids_signals_and_inspection(
    control, monkeypatch, capsys, reaped_elsewhere
):
    class Child:
        pid = 456
        returncode = None if reaped_elsewhere else 0

        def wait(self):
            pytest.fail("Lost ownership cannot authorize reaping")

    def forbidden(*_args):
        pytest.fail("Cannot signal or inspect an unpinned group")

    def lost(*_args):
        raise ChildProcessError("another waiter reaped the leader")

    def no_retry(_seconds):
        pytest.fail("Lost leader identity is permanent; retry cannot recover it")

    monkeypatch.setattr(control.os, "killpg", forbidden)
    monkeypatch.setattr(control, "group_members", forbidden)
    monkeypatch.setattr(control, "observe_exit", lost)
    monkeypatch.setattr(control.time, "sleep", no_retry)
    with pytest.raises(subprocess.SubprocessError, match="leader identity unavailable"):
        control.drain_group(Child())
    err = capsys.readouterr().err
    assert "leader identity unavailable" in err
    assert err.count("leader identity unavailable") == 1


@pytest.mark.parametrize("own_visible", [False, True])
@pytest.mark.parametrize("tasks_visible", [False, True])
def test_linux_scan_checks_own_visibility_and_zombie_tasks(
    control, monkeypatch, tmp_path, own_visible, tasks_visible
):
    leader = tmp_path / "123"
    leader.mkdir()
    (leader / "stat").write_text("123 (leader) Z 1 123\n")
    if tasks_visible:
        (leader / "task/123").mkdir(parents=True)
        (leader / "task/124").mkdir()
        (leader / "task/123/stat").write_text("123 (leader) Z 1 123\n")
        (leader / "task/124/stat").write_text("124 (worker) S 1 123\n")
    if own_visible:
        own = tmp_path / str(os.getpid())
        own.mkdir()
        (own / "stat").write_text(f"{os.getpid()} (self) R 1 999\n")
    monkeypatch.setattr(control.sys, "platform", "linux")
    monkeypatch.setattr(control, "Path", lambda _path: tmp_path)
    assert control.native_group_members(123) == (
        ["124(S)"] if own_visible and tasks_visible else None
    )


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


@pytest.mark.parametrize(
    "boundary", ["spawn", "raising-enter", "wait", "raising-exit", "drain", "reap", "restore"]
)
def test_repeated_transition_signals_use_one_dispatcher_and_deliver_after_reap(
    control, monkeypatch, boundary
):
    popen, change_handler, drain = subprocess.Popen, signal.signal, control.drain_group
    children, deliveries, swaps, sent = [], [], [], []
    fired = False
    previous = {sig: signal.getsignal(sig) for sig in control.INTERRUPTS}

    def handler(signum, _frame):
        deliveries.append(signum)
        assert children and children[0].returncode is not None, "Interrupt delivered before reap"
        assert all(signal.getsignal(sig) is handler for sig in control.INTERRUPTS)
        raise InterruptedError("operator wins")

    def fire():
        nonlocal fired
        if not fired:
            fired = True
            signals = (signal.SIGINT, signal.SIGTERM, signal.SIGINT)
            for sig in signals if boundary in ("reap", "restore") else signals[:1]:
                sent.append(sig)
                os.kill(os.getpid(), sig)

    def spawn(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        wait = child.wait

        def reap(*args, **kwargs):
            if boundary == "reap":
                fire()
            return wait(*args, **kwargs)

        child.wait = reap
        if boundary == "spawn":
            fire()
        return child

    def record_handler(sig, replacement):
        swaps.append(sig)
        return change_handler(sig, replacement)

    def record_drain(child):
        if boundary == "drain":
            fire()
        if fired:
            # The first signal can raise out of fire(). Send the remaining
            # operator actions independently, after the drain defers delivery.
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGINT)[len(sent) :]:
                sent.append(sig)
                os.kill(os.getpid(), sig)
        return drain(child)

    lines, start = inspect.getsourcelines(control.InterruptDeferral.raising)
    after_yield = start + next(i for i, line in enumerate(lines) if "yield" in line) + 2
    exit_lines, exit_start = inspect.getsourcelines(control.deferred_interrupts)
    restore_edge = exit_start + max(
        i for i, line in enumerate(exit_lines) if "with blocked_interrupts():" in line
    )

    def trace(frame, event, _arg):
        if frame.f_code.co_filename == control.__file__:
            name = frame.f_code.co_name
            if (
                (boundary == "raising-enter" and name == "raising" and event == "call")
                or (boundary == "wait" and name == "wait" and event == "call")
                or (
                    boundary == "raising-exit"
                    and name == "raising"
                    and event == "line"
                    and frame.f_lineno >= after_yield
                )
                or (
                    boundary == "restore"
                    and name == "deferred_interrupts"
                    and event == "line"
                    and frame.f_lineno == restore_edge
                )
            ):
                fire()
        return trace

    for sig in previous:
        change_handler(sig, handler)
    monkeypatch.setattr(control.subprocess, "Popen", spawn)
    monkeypatch.setattr(control.signal, "signal", record_handler)
    monkeypatch.setattr(control, "drain_group", record_drain)
    try:
        sys.settrace(trace)
        with pytest.raises(InterruptedError, match="operator wins"):
            control.run_process([sys.executable, "-c", "pass"], 30)
        assert fired
        assert sent == [signal.SIGINT, signal.SIGTERM, signal.SIGINT]
        assert deliveries == [signal.SIGINT]
        assert all(swaps.count(sig) == 2 for sig in previous), swaps
        assert children[0].returncode is not None
    finally:
        sys.settrace(None)
        for sig, original in previous.items():
            change_handler(sig, original)
        for child in children:
            cleanup_child(child)


@pytest.mark.parametrize(
    "failure", [subprocess.TimeoutExpired("deadline", 30), ValueError("failure")]
)
def test_interrupt_during_timeout_drain_wins_and_chains_timeout(control, monkeypatch, failure):
    drain = control.drain_group
    previous = {sig: signal.signal(sig, control.interrupted) for sig in control.INTERRUPTS}
    sent = []

    def timeout(_output, _seconds):
        raise failure

    def interrupt_and_drain(child):
        for sig in (signal.SIGINT, signal.SIGTERM):
            os.kill(os.getpid(), sig)
            sent.append(sig)
        drain(child)

    monkeypatch.setattr(control.PinnedOutput, "wait", timeout)
    monkeypatch.setattr(control, "drain_group", interrupt_and_drain)
    try:
        with pytest.raises(InterruptedError) as caught:
            control.run_process([sys.executable, "-c", "pass"], 30)
        assert caught.value.__cause__ is failure
        assert sent == [signal.SIGINT, signal.SIGTERM]
    finally:
        for sig, original in previous.items():
            signal.signal(sig, original)


def test_spawn_interrupt_inside_callers_except_is_not_lost(control, monkeypatch):
    popen = subprocess.Popen
    previous = {sig: signal.signal(sig, control.interrupted) for sig in control.INTERRUPTS}

    def spawn(*args, **kwargs):
        child = popen(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGINT)
        return child

    monkeypatch.setattr(control.subprocess, "Popen", spawn)
    try:
        with pytest.raises(InterruptedError):
            try:
                raise ValueError("caller's existing exception")
            except ValueError:
                control.run_process([sys.executable, "-c", "pass"], 30)
    finally:
        for sig, original in previous.items():
            signal.signal(sig, original)


@pytest.mark.parametrize(
    "options",
    [
        {"preexec_fn": lambda: None},
        {"start_new_session": False},
        {"process_group": 0},
        {"close_fds": False},
        {"close_fds": None},
        {"shell": True},
        {"shell": "yes"},
        {"executable": sys.executable},
        {"user": os.getuid()},
        {"group": os.getgid()},
        {"extra_groups": []},
    ],
)
def test_unsupported_ownership_options_are_rejected_before_spawn(control, monkeypatch, options):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Ownership-changing options must be rejected before Popen")

    monkeypatch.setattr(control.subprocess, "Popen", forbidden)
    with pytest.raises(ValueError, match=next(iter(options))):
        control.run_process([sys.executable, "-c", "pass"], 30, **options)


@pytest.mark.parametrize("text", [False, True])
def test_timeout_keeps_callers_argv_and_captured_bytes(control, monkeypatch, text):
    def timeout(output, seconds):
        # This read is the readiness handshake: the child writes both pipes
        # before the deadline is injected, independent of startup speed.
        while not all(output.buffers.values()):
            output.read_ready(1)
        raise subprocess.TimeoutExpired(output.child.args, seconds)

    monkeypatch.setattr(control.PinnedOutput, "wait", timeout)
    argv = [
        sys.executable,
        "-c",
        "import os,time; os.write(1,b'\\xffout\\n'); os.write(2,b'\\xfferr\\n'); time.sleep(60)",
    ]
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        control.run_process(argv, 30, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text)
    assert caught.value.cmd == argv
    assert caught.value.stdout == b"\xffout\n"
    assert caught.value.stderr == b"\xfferr\n"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux zombie thread-group leader")
def test_linux_zombie_leader_with_live_thread_is_live(control):
    # pthread_exit kills only the main thread. The worker announces its TID;
    # /proc then provides an explicit handshake for the leader's zombie state.
    script = """
import ctypes, os, threading
def worker():
    print(threading.get_native_id(), flush=True)
    threading.Event().wait()
threading.Thread(target=worker).start()
ctypes.CDLL(None).pthread_exit(None)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script], start_new_session=True, stdout=subprocess.PIPE, text=True
    )
    try:
        tid = int(child.stdout.readline())
        end = time.monotonic() + 30
        while Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z":
            assert time.monotonic() < end, "Leader never reached pthread_exit"
            os.sched_yield()
        members = control.native_group_members(child.pid)
        assert members is not None and any(member.startswith(f"{tid}(") for member in members)
    finally:
        cleanup_child(child, group=True)
        child.stdout.close()
