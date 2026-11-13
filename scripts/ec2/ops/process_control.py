"""Finite subprocess lifetimes, including descendants and interrupted callers."""

import ctypes
import errno
import os
import selectors
import signal
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

SCOPE = "_EC2_OPS_PROCESS_GROUP"
INTERRUPTS = {signal.SIGINT, signal.SIGTERM}
DRAIN_DIAGNOSTIC_SECONDS = 5


def observe_exit(pid, block=False):
    """WNOWAIT, including macOS Python builds that do not expose os.waitid."""
    if hasattr(os, "waitid"):
        flags = os.WEXITED | os.WNOWAIT | (0 if block else os.WNOHANG)
        return os.waitid(os.P_PID, pid, flags) is not None
    if sys.platform != "darwin":
        raise RuntimeError("Non-reaping waitid is required")
    # Darwin sys/{wait,signal}.h: P_PID=1, WEXITED=4, WNOWAIT=32,
    # WNOHANG=1; siginfo_t starts with signo, errno, code, pid (four ints).
    # The oversized aligned buffer leaves room for the remaining ABI fields.
    info = (ctypes.c_int * 64)()
    waitid = ctypes.CDLL(None, use_errno=True).waitid
    waitid.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.c_int]
    waitid.restype = ctypes.c_int
    if waitid(1, pid, ctypes.byref(info), 4 | 32 | (0 if block else 1)):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return info[3] == pid


def exited(child):
    """Observe without wait/poll/communicate: the unreaped PID pins the PGID."""
    if child.returncode is not None:
        raise RuntimeError("Process-group leader was reaped before its death barrier")
    return observe_exit(child.pid)


def darwin_process_state(pid):
    """sysctl includes zombies and processes already leaving libproc's live list."""

    # sys/proc.h extern_proc prefix: the p_un union occupies two pointers,
    # followed by vmspace/sigacts pointers, flags, state and the aligned PID.
    class Prefix(ctypes.Structure):
        _fields_ = [
            ("links", ctypes.c_void_p * 4),
            ("flags", ctypes.c_int),
            ("state", ctypes.c_byte),
            ("pid", ctypes.c_int),
        ]

    query = ctypes.CDLL(None, use_errno=True).sysctl
    query.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    mib = (ctypes.c_int * 4)(1, 14, 1, pid)  # CTL_KERN, KERN_PROC, KERN_PROC_PID.
    data = ctypes.create_string_buffer(4096)
    size = ctypes.c_size_t(ctypes.sizeof(data))
    if query(mib, 4, data, ctypes.byref(size), None, 0):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if size.value == 0:
        return None  # Removed since enumeration.
    info = Prefix.from_buffer(data)
    if size.value < ctypes.sizeof(Prefix) or info.pid != pid:
        raise ValueError("Invalid Darwin process snapshot")
    return info.state


def native_group_members(group):
    """A complete scan includes the pinned leader, even when it is a zombie."""
    live, leader_seen = [], False
    try:
        if sys.platform == "linux":
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    # comm may contain spaces and parentheses. Fields after
                    # its final ')' are state, ppid, pgrp, ... (proc_pid_stat).
                    fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                except FileNotFoundError:
                    continue  # Exited/reaped since the directory enumeration.
                if int(fields[2]) == group:
                    leader_seen |= int(entry.name) == group
                    if fields[0] != "Z":
                        live.append(f"{entry.name}({fields[0]})")
        elif sys.platform == "darwin":
            lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
            listpids = lib.proc_listpids
            listpids.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.c_int]
            listpids.restype = ctypes.c_int
            pidinfo = lib.proc_pidinfo
            pidinfo.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint64,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            pidinfo.restype = ctypes.c_int
            capacity = 256
            while True:
                pids = (ctypes.c_int * capacity)()
                size = ctypes.sizeof(pids)
                used = listpids(2, group, pids, size)  # PROC_PGRP_ONLY, bytes returned.
                if used <= 0 or used > size or used % ctypes.sizeof(ctypes.c_int):
                    return None
                if used < size:
                    break
                capacity *= 2  # Never accept a potentially truncated snapshot.
            for pid in pids[: used // ctypes.sizeof(ctypes.c_int)]:
                leader_seen |= pid == group
                # sys/proc_info.h: proc_bsdshortinfo is 64 bytes; its first
                # four uint32s are pid, ppid, pgid, status. Flavor 13.
                info = (ctypes.c_uint32 * 16)()
                copied = pidinfo(pid, 13, 0, info, ctypes.sizeof(info))
                if copied == 0 and ctypes.get_errno() == errno.ESRCH:
                    # ESRCH also covers processes still exiting. Confirm their
                    # state instead of treating every lookup miss as death.
                    state = darwin_process_state(pid)
                    if state is not None and state != 5:
                        live.append(f"{pid}({state})")
                    continue
                if copied != ctypes.sizeof(info) or info[0] != pid:
                    return None
                if info[2] == group and info[3] != 5:  # SZOMB
                    state = {1: "I", 2: "R", 3: "S", 4: "T"}.get(info[3], str(info[3]))
                    live.append(f"{pid}({state})")
        else:
            return None
    except (OSError, ValueError, IndexError, AttributeError):
        return None
    return live if leader_seen else None


def group_members(group):
    """None means unknown; an empty list requires a valid, leader-inclusive scan."""
    native = native_group_members(group)
    if native is not None:
        return native
    try:
        result = subprocess.run(
            ["ps", "-A", "-o", "pid=", "-o", "pgid=", "-o", "stat="],
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode:
        return None
    live, leader_seen = [], False
    for line in result.stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 3 or not fields[0].isdigit() or not fields[1].isdigit():
            return None
        pid, pgid, state = fields
        if pgid == str(group):
            leader_seen |= pid == str(group)
            if not state.startswith("Z"):
                live.append(f"{pid}({state})")
    return live if leader_seen else None


def drain_group(child):
    """Keep the leader unreaped through the last signal and membership scan."""
    started = time.monotonic()
    diagnostic_at = started + DRAIN_DIAGNOSTIC_SECONDS
    while True:
        refused = None
        try:
            exited(child)
        except (OSError, RuntimeError) as exc:
            # Lost ownership cannot authorize a signal OR a scan of that ID.
            members, refused = None, f"leader identity unavailable: {exc}"
        else:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                break
            except OSError as exc:
                # macOS returns EPERM even for our own zombie-only group.
                # Inspect on refusal too, before deciding it is safe to reap.
                refused = f"SIGKILL refused: {exc}"
            members = group_members(child.pid)
        if members == []:
            break
        now = time.monotonic()
        if now >= diagnostic_at:
            detail = ", ".join(members) if members is not None else "unknown (inspection failed)"
            try:
                print(
                    f"Waiting for local process group {child.pid}: live members={detail}; "
                    f"elapsed={now - started:.1f}s; session retained"
                    + (f"; {refused}" if refused else ""),
                    file=sys.stderr,
                    flush=True,
                )
            except (OSError, ValueError):
                # A closed diagnostic pipe must never release the barrier.
                pass
            diagnostic_at = now + DRAIN_DIAGNOSTIC_SECONDS
        time.sleep(0.05)
    # No group signal or inspection is permitted after this point.
    child.wait()


def scope_fd():
    """An inherited number is insufficient: require an open lease from the owner."""
    try:
        group, fd, device, inode = map(int, os.environ.get(SCOPE, "").split(":"))
        info = os.fstat(fd)
        if (
            group != os.getpgrp()
            or not stat.S_ISFIFO(info.st_mode)
            or (info.st_dev, info.st_ino) != (device, inode)
            or os.get_blocking(fd)
        ):
            return None
        # Only the outer owner retains a writer. EOF proves that owner is gone;
        # EAGAIN proves a live writer still owns this exact inherited pipe.
        try:
            os.read(fd, 1)
        except BlockingIOError:
            return fd
    except (ValueError, OSError):
        pass
    return None


class PinnedOutput:
    """Drain pipes while observing exit, without Popen's implicit leader reap."""

    def __init__(self, child):
        self.child = child
        self.selector = selectors.DefaultSelector()
        self.buffers = {}

    def __enter__(self):
        try:
            if self.child.stdin:
                self.child.stdin.close()  # Like communicate(input=None).
            for stream in (self.child.stdout, self.child.stderr):
                if stream:
                    self.buffers[stream] = bytearray()
                    os.set_blocking(stream.fileno(), False)
                    self.selector.register(stream, selectors.EVENT_READ)
        except BaseException:
            self.selector.close()
            raise
        return self

    def read_ready(self, timeout):
        for key, _events in self.selector.select(timeout):
            chunk = os.read(key.fd, 65536)
            if chunk:
                self.buffers[key.fileobj].extend(chunk)
            else:
                self.selector.unregister(key.fileobj)

    def wait(self, seconds):
        end = time.monotonic() + seconds
        while not exited(self.child):
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(self.child.args, seconds)
            self.read_ready(min(0.01, remaining))

    def output(self, stream):
        if stream is None:
            return None
        data = bytes(self.buffers[stream])
        if self.child.text_mode:
            return (
                data.decode(stream.encoding, stream.errors)
                .replace("\r\n", "\n")
                .replace("\r", "\n")
            )
        return data

    def __exit__(self, *_exc):
        self.selector.close()
        for stream in (self.child.stdin, self.child.stdout, self.child.stderr):
            if stream:
                stream.close()


def run_process(argv, seconds, **kwargs):
    seconds = float(seconds)
    if not 0 < seconds < float("inf"):
        raise ValueError("Command timeout must be positive and finite")
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
        raise ValueError("Process-group ownership requires the default SIGCHLD disposition")
    # Deferral precedes Popen. Raising is enabled only inside the already
    # established finally below, including delivery of a spawn-time interrupt.
    with deferred_interrupts() as interrupts:
        child = None
        lease = None
        inherited = scope_fd()
        nested = inherited is not None
        try:
            env = dict(kwargs.pop("env", os.environ))
            if nested:
                command = argv
                env[SCOPE] = os.environ[SCOPE]
                fd = inherited
            else:
                lease = os.pipe()
                fd = lease[0]
                os.set_blocking(fd, False)
                command = [sys.executable, __file__, str(fd), *argv]
                env.pop(SCOPE, None)
            kwargs["pass_fds"] = tuple(set(kwargs.get("pass_fds", ())) | {fd})
            child = subprocess.Popen(command, start_new_session=not nested, env=env, **kwargs)
            try:
                if nested:
                    with interrupts.raising():
                        stdout, stderr = child.communicate(timeout=seconds)
                else:
                    with PinnedOutput(child) as output:
                        try:
                            with interrupts.raising():
                                output.wait(seconds)
                        except BaseException:
                            try:
                                exited(child)
                                os.killpg(child.pid, signal.SIGTERM)
                            except (OSError, RuntimeError):
                                pass
                            grace = time.monotonic() + 1
                            while not exited(child) and time.monotonic() < grace:
                                output.read_ready(0.01)
                            raise
                        finally:
                            drain_group(child)
                        # Once no live member remains, all owned writers are
                        # gone. Collect buffered bytes without waiting on an
                        # escaped/daemonized writer outside this operation.
                        end = time.monotonic() + seconds
                        while output.selector.select(0):
                            if time.monotonic() >= end:
                                raise subprocess.TimeoutExpired(argv, seconds)
                            output.read_ready(0)
                        stdout = output.output(child.stdout)
                        stderr = output.output(child.stderr)
            finally:
                if nested:
                    # Only the owner may signal the shared group. Nested
                    # supervisors only reap their direct child while unwinding.
                    if child.returncode is None:
                        try:
                            child.terminate()
                        except OSError:
                            pass
                        try:
                            child.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
        finally:
            # Includes signals at pipe setup / context-entry transition edges.
            if child is not None and not nested and child.returncode is None:
                drain_group(child)
            if child is not None:
                for stream in (child.stdin, child.stdout, child.stderr):
                    if stream:
                        stream.close()
            if lease:
                for fd in lease:
                    os.close(fd)
        return subprocess.CompletedProcess(argv, child.returncode, stdout, stderr)


def interrupted(signum, _frame):
    raise InterruptedError(f"Operator interrupted by signal {signum}")


def install_interrupts():
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)


@contextmanager
def blocked_interrupts():
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, INTERRUPTS)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


class InterruptDeferral:
    def __init__(self):
        self.pending = []
        self.previous = {sig: signal.getsignal(sig) for sig in INTERRUPTS}

    def defer(self, signum, _frame):
        if not self.pending:
            self.pending.append(signum)

    def restore(self):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)

    def forward(self, signum, frame):
        try:
            self.previous[signum](signum, frame)
        except InterruptedError as exc:
            # selectors swallow InterruptedError as a syscall EINTR. Carry
            # the operator's exception through I/O, translating it only once
            # we have left that boundary and entered the drain's finally.
            raise ForwardedInterrupt(exc) from None

    def deliver(self):
        if self.pending and sys.exc_info()[0] is None:
            signum = self.pending.pop(0)
            self.pending.clear()
            handler = self.previous[signum]
            if callable(handler):
                handler(signum, None)
            elif handler == signal.SIG_DFL:
                signal.raise_signal(signum)

    @contextmanager
    def raising(self):
        try:
            try:
                with blocked_interrupts():
                    for sig, handler in self.previous.items():
                        signal.signal(sig, self.forward if callable(handler) else handler)
                self.deliver()
                yield
            finally:
                with blocked_interrupts():
                    for sig in INTERRUPTS:
                        signal.signal(sig, self.defer)
        except ForwardedInterrupt as exc:
            raise exc.interrupt from None


class ForwardedInterrupt(BaseException):
    """Escape I/O's EINTR handling without changing the caller's exception API."""

    def __init__(self, interrupt):
        self.interrupt = interrupt


@contextmanager
def deferred_interrupts():
    with blocked_interrupts():
        interrupts = InterruptDeferral()
        for sig in INTERRUPTS:
            signal.signal(sig, interrupts.defer)
    try:
        yield interrupts
    finally:
        with blocked_interrupts():
            interrupts.restore()
    interrupts.deliver()


@contextmanager
def cleanup_signals():
    # Install/restore both dispositions atomically with respect to delivery.
    with blocked_interrupts():
        previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in INTERRUPTS}
    try:
        yield
    finally:
        with blocked_interrupts():
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    fd = int(sys.argv[1])
    info = os.fstat(fd)
    os.environ[SCOPE] = f"{os.getpgrp()}:{fd}:{info.st_dev}:{info.st_ino}"
    os.execvpe(sys.argv[2], sys.argv[2:], os.environ)
