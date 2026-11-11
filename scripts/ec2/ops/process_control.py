"""Finite subprocess lifetimes, including descendants and interrupted callers."""

import os
import signal
import subprocess
from contextlib import contextmanager


def run_process(argv, seconds, **kwargs):
    seconds = float(seconds)
    if not 0 < seconds < float("inf"):
        raise ValueError("Command timeout must be positive and finite")
    with subprocess.Popen(argv, start_new_session=True, **kwargs) as child:
        try:
            stdout, stderr = child.communicate(timeout=seconds)
        except BaseException:
            # Killing only the shell leaves ssh/rsync (and their pipes) alive.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            raise
        return subprocess.CompletedProcess(argv, child.returncode, stdout, stderr)


def interrupted(signum, _frame):
    raise InterruptedError(f"Operator interrupted by signal {signum}")


def install_interrupts():
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)


@contextmanager
def cleanup_signals():
    previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
