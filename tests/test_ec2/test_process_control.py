"""Local cancellation barriers; no cloud commands or transport mocks."""

import importlib.util
import os
import signal
import subprocess
import sys
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
        alive = True
        try:
            os.kill(pid, 0)
            if sys.platform == "linux":
                alive = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
        except (ProcessLookupError, FileNotFoundError):
            alive = False
        assert not alive, f"Background descendant {pid} outlived its operation"
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
