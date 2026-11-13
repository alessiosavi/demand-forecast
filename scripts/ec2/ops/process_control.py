"""Owned subprocess groups, including descendants and interrupted callers."""

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


class LeaderIdentityLost(subprocess.SubprocessError):
    """Exclusive ownership was lost; do not signal or inspect the recycled ID."""


def diagnostic(message):
    try:
        print(message, file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass  # A closed diagnostic pipe must never release the death barrier.


def diagnose_cancellation(group, error, pending):
    if isinstance(error, subprocess.TimeoutExpired):
        reason = f"Time limit ({error.timeout:g}s) reached"
    elif pending is not None:
        reason = f"Interrupt signal {pending}"
    elif isinstance(error, (KeyboardInterrupt, InterruptedError)):
        reason = "Command interrupted"
    else:
        reason = f"Command failed ({type(error).__name__})"
    diagnostic(f"{reason}; terminating local process group {group}")


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
        raise LeaderIdentityLost("leader identity unavailable: leader was already reaped")
    try:
        return observe_exit(child.pid)
    except ChildProcessError as exc:
        raise LeaderIdentityLost(f"leader identity unavailable: {exc}") from exc


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
            own_seen = False
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    # comm may contain spaces and parentheses. Fields after
                    # its final ')' are state, ppid, pgrp, ... (proc_pid_stat).
                    fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                except FileNotFoundError:
                    continue  # Exited/reaped since the directory enumeration.
                own_seen |= int(entry.name) == os.getpid()
                if int(fields[2]) == group:
                    leader_seen |= int(entry.name) == group
                    if fields[0] != "Z":
                        live.append(f"{entry.name}({fields[0]})")
                    else:
                        # pthread_exit can leave a zombie leader with live
                        # threads. The kernel cannot reap it until they exit.
                        try:
                            tasks = list((entry / "task").iterdir())
                        except FileNotFoundError:
                            # A missing task directory is not proof that every
                            # thread exited while the leader is still visible.
                            try:
                                (entry / "stat").read_text()
                            except FileNotFoundError:
                                continue
                            return None
                        if not tasks:
                            return None
                        for task in tasks:
                            try:
                                state = (task / "stat").read_text().rsplit(")", 1)[1].split()[0]
                            except FileNotFoundError:
                                continue
                            if state != "Z":
                                live.append(f"{task.name}({state})")
            if not own_seen:
                return None
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
            [
                "ps",
                "-A",
                *(["-L"] if sys.platform == "linux" else []),
                "-o",
                "pid=",
                "-o",
                "pgid=",
                "-o",
                "stat=",
            ],
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode:
        return None
    live, leader_seen, own_seen = [], False, False
    for line in result.stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 3 or not fields[0].isdigit() or not fields[1].isdigit():
            return None
        pid, pgid, state = fields
        own_seen |= pid == str(os.getpid())
        if pgid == str(group):
            leader_seen |= pid == str(group)
            if not state.startswith("Z"):
                live.append(f"{pid}({state})")
    # On Linux a failed native task scan cannot be repaired by ps omitting
    # those same threads. Its fallback may identify live members, but only a
    # complete /proc task scan can certify a zombie-only group.
    if sys.platform == "linux" and not live:
        return None
    return live if leader_seen and (own_seen or sys.platform != "linux") else None


def drain_group(child):
    """Keep the leader unreaped through the last signal and membership scan."""
    started = time.monotonic()
    diagnostic_at = started
    while True:
        refused = None
        try:
            exited(child)
        except LeaderIdentityLost as exc:
            diagnostic(f"Local process group {child.pid}: {exc}; continuing caller cleanup")
            raise
        except (OSError, RuntimeError) as exc:
            # Lost ownership cannot authorize a signal OR a scan of that ID.
            members, refused = None, f"leader observation failed: {exc}"
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
            diagnostic(
                f"Waiting for local process group {child.pid}: live members={detail}; "
                f"elapsed={now - started:.1f}s; session retained"
                + (f"; {refused}" if refused else "")
            )
            diagnostic_at = now + DRAIN_DIAGNOSTIC_SECONDS
        time.sleep(0.05)
    # No group signal or inspection is permitted after this point.
    child.wait()


def scope_fd():
    """Reject accidental stale scope using group, pipe identity and a live writer.

    This is a nesting convention within the operator's own process tree, not
    authentication against that tree: a process controlling its environment
    and descriptors already controls itself. A matching pipe in the recorded
    group with a live writer is nested by definition.
    """
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

    def collect(self, seconds):
        end = time.monotonic() + seconds
        while self.selector.select(0):
            if time.monotonic() >= end:
                raise subprocess.TimeoutExpired(self.child.args, seconds)
            self.read_ready(0)

    def captured(self, stream):
        return bytes(self.buffers[stream]) if stream is not None else None

    def __exit__(self, *_exc):
        self.selector.close()
        for stream in (self.child.stdin, self.child.stdout, self.child.stderr):
            if stream:
                stream.close()


def run_process(argv, seconds, **kwargs):
    seconds = float(seconds)
    if not 0 < seconds < float("inf"):
        raise ValueError("Command timeout must be positive and finite")
    for option in (
        "preexec_fn",
        "start_new_session",
        "process_group",
        "executable",
        "user",
        "group",
        "extra_groups",
    ):
        if option in kwargs:
            raise ValueError(f"{option} is incompatible with process-group ownership")
    for option, unsupported in (("close_fds", False), ("shell", True)):
        if option in kwargs and bool(kwargs[option]) == unsupported:
            raise ValueError(f"{option}={unsupported} is incompatible with process-group ownership")
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
        raise ValueError("Process-group ownership requires the default SIGCHLD disposition")
    child = lease = None
    stdout = stderr = None
    # One dispatcher owns the entire lifetime. Only the guarded I/O wait can
    # raise its private carrier; every exit/drain edge remains deferred.
    with deferred_interrupts() as interrupts:
        inherited = scope_fd()
        nested = inherited is not None
        drain_attempted = False
        try:
            supplied_env = kwargs.pop("env", None)
            env = dict(os.environ if supplied_env is None else supplied_env)
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
            if nested:
                try:
                    with interrupts.raising():
                        stdout, stderr = child.communicate(timeout=seconds)
                except BaseException as exc:
                    if child.returncode is None:
                        diagnose_cancellation(os.getpgrp(), exc, interrupts.pending)
                    raise
                finally:
                    # Only the owner may signal the shared group.
                    if child.returncode is None:
                        try:
                            child.terminate()
                        except OSError:
                            pass
                        try:
                            child.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            diagnostic(f"Waiting to reap killed nested local process {child.pid}")
                            child.wait()
            else:
                with PinnedOutput(child) as output:
                    completed = False
                    try:
                        try:
                            with interrupts.raising():
                                output.wait(seconds)
                            completed = True
                        except BaseException as exc:
                            diagnose_cancellation(child.pid, exc, interrupts.pending)
                            try:
                                exited(child)
                                os.killpg(child.pid, signal.SIGTERM)
                                grace = time.monotonic() + 1
                                while not exited(child) and time.monotonic() < grace:
                                    output.read_ready(0.01)
                            except (OSError, LeaderIdentityLost):
                                pass  # The final drain diagnoses lost ownership.
                            raise
                        finally:
                            drain_attempted = True
                            drain_group(child)
                            output.collect(seconds)
                            if completed:
                                stdout = output.output(child.stdout)
                                stderr = output.output(child.stderr)
                    except subprocess.TimeoutExpired as exc:
                        exc.cmd = argv
                        exc.output = output.captured(child.stdout)
                        exc.stderr = output.captured(child.stderr)
                        raise

        finally:
            # Covers PinnedOutput setup and every other pre-wait transition.
            # Never retry a failed drain: lost identity is permanent.
            try:
                if child is not None and not nested and not drain_attempted:
                    drain_group(child)
            finally:
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
        self.pending = None
        self.raising_enabled = False
        self.previous = {sig: signal.getsignal(sig) for sig in INTERRUPTS}

    def dispatch(self, signum, _frame):
        if self.previous[signum] == signal.SIG_IGN:
            return
        if self.pending is None:
            self.pending = signum
        if self.raising_enabled:
            # Switch BEFORE raising: even a signal in the context-manager's
            # exit transition cannot interrupt the death barrier.
            self.raising_enabled = False
            raise ForwardedInterrupt()

    def restore(self):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)

    def deliver(self, error):
        if self.pending is None:
            return
        signum, self.pending = self.pending, None
        handler = self.previous[signum]
        try:
            if callable(handler):
                handler(signum, None)
            elif handler == signal.SIG_DFL:
                signal.raise_signal(signum)
        except BaseException as interrupt:
            # Explicit precedence, independent of a caller's active except.
            # The carrier is private; only the caller's handler defines its API.
            cause = None if isinstance(error, ForwardedInterrupt) else error
            raise interrupt from cause

    @contextmanager
    def raising(self):
        try:
            self.raising_enabled = True
            if self.pending is not None:
                self.raising_enabled = False
                raise ForwardedInterrupt()
            yield
        finally:
            self.raising_enabled = False


class ForwardedInterrupt(BaseException):
    """Escape selectors' EINTR handling; never delivered to the caller."""


@contextmanager
def deferred_interrupts():
    with blocked_interrupts():
        interrupts = InterruptDeferral()
        for sig in INTERRUPTS:
            signal.signal(sig, interrupts.dispatch)
    error = None
    try:
        yield interrupts
    except BaseException as exc:
        error = exc
    finally:
        interrupts.raising_enabled = False
        try:
            with blocked_interrupts():
                interrupts.restore()
        finally:
            # Also runs on exceptional exits, only after the barrier and restore.
            interrupts.deliver(error)
    if error is not None and not isinstance(error, ForwardedInterrupt):
        raise error


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
