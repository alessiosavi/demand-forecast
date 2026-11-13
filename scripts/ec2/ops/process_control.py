"""Finite subprocess lifetimes, including descendants and interrupted callers."""

import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager

SCOPE = "_EC2_OPS_PROCESS_GROUP"
INTERRUPTS = {signal.SIGINT, signal.SIGTERM}


def drain_group(child):
    """Shell exit is not a barrier: kill and wait for its entire local scope."""
    while True:
        try:
            os.killpg(child.pid, signal.SIGKILL)
            break
        except ProcessLookupError:
            break
        except OSError:
            # A refused signal does not prove death. Retain the session until
            # killing/inspection succeeds rather than unwinding into cleanup.
            time.sleep(0.1)
    child.wait()
    while True:
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            return
        except OSError:
            time.sleep(0.1)
            continue
        # Orphan zombies can remain until init reaps them (especially in Linux
        # containers). They cannot run or retain pipes, so do not wait on init.
        # If inspection fails, keep waiting: unconfirmed death cannot permit
        # the operator to close its paid-machine session.
        try:
            members = subprocess.run(
                ["ps", "-A", "-o", "pgid=,stat="], capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.SubprocessError):
            time.sleep(0.1)
            continue
        if members.returncode == 0 and not any(
            fields[0] == str(child.pid) and not fields[1].startswith("Z")
            for line in members.stdout.splitlines()
            if len(fields := line.split()) == 2
        ):
            return
        time.sleep(0.01)


def run_process(argv, seconds, **kwargs):
    seconds = float(seconds)
    if not 0 < seconds < float("inf"):
        raise ValueError("Command timeout must be positive and finite")
    # The exec trampoline records the new group's actual ID. Nested supervisors
    # inherit it and never detach their transports from the outer operation.
    nested = os.environ.get(SCOPE) == str(os.getpgrp())
    command = argv if nested else [sys.executable, __file__, *argv]
    with subprocess.Popen(command, start_new_session=not nested, **kwargs) as child:
        try:
            stdout, stderr = child.communicate(timeout=seconds)
        except BaseException:
            # Nested callers stop their direct child and unwind to the owner.
            # Only the owner can kill the shared group without killing itself.
            with cleanup_signals():
                try:
                    if nested:
                        child.terminate()
                    else:
                        os.killpg(child.pid, signal.SIGTERM)
                except OSError:
                    # Graceful TERM is best effort. Preserve the original
                    # timeout/interrupt; the owner still enforces group death.
                    pass
                try:
                    child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    if nested:
                        child.kill()
                        child.wait()
            raise
        finally:
            if not nested:
                with deferred_interrupts():
                    drain_group(child)
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


@contextmanager
def deferred_interrupts():
    pending = []
    with blocked_interrupts():
        previous = {sig: signal.getsignal(sig) for sig in INTERRUPTS}
        for sig in INTERRUPTS:
            signal.signal(sig, lambda signum, _frame: pending.append(signum))
    try:
        yield
    finally:
        with blocked_interrupts():
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    # Preserve an exception already unwinding through run_process's finally.
    # A first interrupt during a successful drain must still reach the caller.
    if pending and sys.exc_info()[0] is None:
        signum = pending[0]
        handler = previous[signum]
        if callable(handler):
            handler(signum, None)
        elif handler == signal.SIG_DFL:
            signal.raise_signal(signum)


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
    os.environ[SCOPE] = str(os.getpgrp())
    os.execvpe(sys.argv[1], sys.argv[1:], os.environ)
