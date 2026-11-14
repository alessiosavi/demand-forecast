"""The acceptance verifier must fail closed on unresolved process identities."""

import ctypes
import errno
import importlib.util
import json
import os
import sys
from pathlib import Path

import process_identity as identities
import pytest


@pytest.fixture
def verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_cancellation", Path(__file__).with_name("verify_cancellation.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failure", ["esrch", "eperm", "short", "not-enumerated"])
@pytest.mark.parametrize(
    "state", [None, 5, 2, "unavailable"], ids=["gone", "zombie", "live", "unknown"]
)
def test_tracked_darwin_lookup_requires_second_source(
    verifier, monkeypatch, tmp_path, failure, state
):
    own, tracked = os.getpid(), 987654
    own_identity, tracked_identity = (1, 2, 3, 4), (5, 6, 7, 8)
    group_log = tmp_path / "groups"
    group_log.write_text("")
    scanning = False

    class Function:
        def __init__(self, function):
            self.function = function

        def __call__(self, *args):
            return self.function(*args)

    def enumerate_pids(_kind, _filter, pids, _size):
        rows = [own] if not scanning or failure == "not-enumerated" else [own, tracked]
        for index, pid in enumerate(rows):
            pids[index] = pid
        return len(rows) * ctypes.sizeof(ctypes.c_int)

    def inspect_pid(pid, _flavor, _arg, info, size):
        if pid == own:
            info[1], info[3], info[4], info[25] = 2, own, 1, 999
            info[30:34] = own_identity
            return size
        ctypes.set_errno(errno.ESRCH if failure == "esrch" else errno.EPERM)
        return 4 if failure == "short" else 0

    class Libproc:
        proc_listpids = Function(enumerate_pids)
        proc_pidinfo = Function(inspect_pid)

    checked = []

    def second_source(pid):
        checked.append(pid)
        if state == "unavailable":
            raise PermissionError("sysctl unavailable")
        return None if state is None else (state, tracked_identity)

    monkeypatch.setattr(verifier.sys, "platform", "darwin")
    monkeypatch.setattr(verifier.ctypes, "CDLL", lambda *_args, **_kwargs: Libproc())
    monkeypatch.setattr(verifier, "darwin_process_info", second_source, raising=False)
    tracker = verifier.ProcessTracker(group_log)
    tracker.known[tracked] = tracked_identity
    scanning = True
    if state in (None, 5):
        assert tracker.scan() == []
        assert tracked not in tracker.known
    else:
        assert tracker.scan() == []
        assert tracker.known[tracked] == tracked_identity
        assert tracker.unresolved
        assert tracker.notes
        assert tracker.errors == []
    assert checked == [tracked], "A failed lookup or enumeration omission must be confirmed"


@pytest.mark.parametrize("gone", [False, True], ids=["still-visible", "confirmed-gone"])
def test_tracked_linux_pid_omitted_from_proc_requires_confirmation(
    verifier, monkeypatch, tmp_path, gone
):
    own, tracked = os.getpid(), 987654
    own_dir = tmp_path / str(own)
    own_dir.mkdir()
    (own_dir / "stat").write_text(f"{own} (self) S 1 999 " + "0 " * 16 + "123\n")
    group_log = tmp_path / "groups"
    group_log.write_text("")
    checks = []

    def probe(pid, sig):
        assert (pid, sig) == (tracked, 0)
        checks.append(pid)
        if gone:
            raise ProcessLookupError()

    monkeypatch.setattr(verifier.sys, "platform", "linux")
    monkeypatch.setattr(verifier, "Path", lambda _path: tmp_path)
    monkeypatch.setattr(verifier.os, "kill", probe)
    tracker = verifier.ProcessTracker(group_log)
    tracker.known[tracked] = "456"
    if gone:
        assert tracker.scan() == []
        assert tracked not in tracker.known
    else:
        assert tracker.scan() == []
        assert tracked in tracker.unresolved
        assert tracker.known[tracked] == "456"
    assert checks == [tracked]


@pytest.mark.parametrize("current", ["owned", "reused", None])
def test_cleanup_never_signals_reused_or_uninspectable_pid(monkeypatch, current):
    signaled = []
    monkeypatch.setattr(identities, "process_identity", lambda _pid: current)
    monkeypatch.setattr(identities.os, "kill", lambda pid, sig: signaled.append((pid, sig)))
    identities.kill_owned(987654, "owned")
    assert signaled == ([(987654, 9)] if current == "owned" else [])


@pytest.fixture
def darwin_tracker(verifier, monkeypatch, tmp_path):
    """Control OS observations while exercising the real snapshot and tracker."""
    own, tracked = os.getpid(), 987654
    group_log = tmp_path / "groups"
    group_log.write_text("")
    state = {"primary": False, "secondary": (2, (5, 0, 6, 0)), "tracked": False}

    class Function:
        def __init__(self, function):
            self.function = function

        def __call__(self, *args):
            return self.function(*args)

    def enumerate_pids(_kind, _filter, pids, _size):
        pids[0] = own
        if state["tracked"]:
            pids[1] = tracked
        return (2 if state["tracked"] else 1) * ctypes.sizeof(ctypes.c_int)

    def inspect_pid(pid, _flavor, _arg, info, size):
        if pid == tracked and not state["primary"]:
            return 0
        info[1], info[3], info[4], info[25] = 2, pid, 1, 999
        info[30:34] = (1, 0, 2, 0) if pid == own else (5, 0, 6, 0)
        return size

    class Libproc:
        proc_listpids = Function(enumerate_pids)
        proc_pidinfo = Function(inspect_pid)

    monkeypatch.setattr(verifier.sys, "platform", "darwin")
    monkeypatch.setattr(verifier.ctypes, "CDLL", lambda *_args, **_kwargs: Libproc())
    monkeypatch.setattr(
        verifier, "darwin_process_info", lambda _pid: state["secondary"], raising=False
    )
    tracker = verifier.ProcessTracker(group_log)
    tracker.known[tracked] = (5, 0, 6, 0)
    state["tracked"] = True
    return tracker, state


def test_reused_uninspectable_pid_is_dropped_without_error(darwin_tracker):
    tracker, state = darwin_tracker
    state["secondary"] = (2, (7, 0, 8, 0))
    assert tracker.scan() == []
    assert 987654 not in tracker.known
    assert tracker.unresolved == {}
    assert tracker.errors == []
    assert tracker.notes == []


@pytest.mark.parametrize("resolution", ["gone", "inspectable"])
def test_transient_inspection_failure_resolves_with_note_only(darwin_tracker, resolution):
    tracker, state = darwin_tracker
    assert tracker.scan() == []
    assert 987654 in tracker.known
    assert tracker.notes
    state["secondary"] = None
    state["primary"] = resolution == "inspectable"
    assert tracker.finish(timeout=0) == ([987654] if resolution == "inspectable" else [])
    assert tracker.unresolved == {}
    assert tracker.errors == []
    assert tracker.notes


@pytest.mark.parametrize("identity", [(5, 0, 6, 0), None], ids=["same", "unknown"])
def test_persistent_unresolved_identity_fails_final_verification(darwin_tracker, identity):
    tracker, state = darwin_tracker
    state["secondary"] = (2, identity)
    assert tracker.scan() == []
    assert tracker.errors == []
    assert tracker.finish(timeout=0) == []
    assert tracker.errors and "987654" in tracker.errors[0]
    assert 987654 in tracker.known


def test_final_scan_retries_transient_failure(darwin_tracker, verifier, monkeypatch):
    tracker, state = darwin_tracker
    state["secondary"] = iter([(2, (5, 0, 6, 0)), None])
    monkeypatch.setattr(verifier, "darwin_process_info", lambda _pid: next(state["secondary"]))
    assert tracker.finish(timeout=0.5) == []
    assert tracker.errors == []
    assert tracker.notes
    assert 987654 not in tracker.known


def test_watcher_continues_after_exception(darwin_tracker, monkeypatch):
    tracker, state = darwin_tracker
    scan = tracker.scan
    calls = []

    def interrupted_scan():
        calls.append(1)
        if len(calls) == 1:
            raise OSError("temporary enumeration failure")
        state["secondary"] = None
        result = scan()
        tracker.stop.set()
        return result

    monkeypatch.setattr(tracker, "scan", interrupted_scan)
    tracker.watch()
    assert len(calls) == 2
    assert 987654 not in tracker.known
    assert tracker.errors == []
    assert tracker.notes == ["temporary enumeration failure"]


@pytest.mark.parametrize("persistent", [False, True], ids=["transient", "persistent"])
def test_verification_result_only_fails_for_final_inspection_errors(
    darwin_tracker, verifier, monkeypatch, tmp_path, persistent
):
    tracker, state = darwin_tracker
    tracker.scan()
    if not persistent:
        state["secondary"] = None
    monkeypatch.setattr(tracker.thread, "start", lambda: None)
    monkeypatch.setattr(tracker.thread, "join", lambda: None)
    finish = tracker.finish
    monkeypatch.setattr(tracker, "finish", lambda: finish(timeout=0))
    monkeypatch.setattr(verifier, "ProcessTracker", lambda _path: tracker)
    monkeypatch.setattr(verifier, "run_command", lambda *_args: {"exit": 0})
    monkeypatch.setattr(verifier, "stress", lambda *_args, **_kwargs: {"cases": 1, "expected": 1})
    output = tmp_path / "verification"
    monkeypatch.setattr(verifier.sys, "argv", ["verify_cancellation.py", "--output", str(output)])
    assert verifier.main() == int(persistent)
    result = json.loads((output / "result.json").read_text())
    assert result["passed"] is (not persistent)
    assert bool(result["inspection_errors"]) is persistent
    assert result["inspection_notes"]


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin sysctl ABI")
def test_sysctl_start_time_matches_libproc():
    state, identity = identities.darwin_process_info(os.getpid())
    assert state in (1, 2, 3, 4)
    assert identity == tuple(map(int, identities.process_identity(os.getpid()).split(":")))
