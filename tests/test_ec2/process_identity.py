"""Identity checks for test cleanup; an uninspectable PID is never ours to kill."""

import ctypes
import os
import signal
import sys
from pathlib import Path


def darwin_process_info(pid):
    """Return sysctl's (state, start identity), or None for a confirmed exit.

    sys/proc.h exports extern_proc.p_un.p_starttime as a timeval. Normalize
    its seconds/useconds to the four uint32 words used by proc_bsdinfo.
    """

    class Timeval(ctypes.Structure):
        _fields_ = [("seconds", ctypes.c_long), ("microseconds", ctypes.c_int)]

    class Prefix(ctypes.Structure):
        _fields_ = [
            ("start", Timeval),
            ("pointers", ctypes.c_void_p * 2),
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
    mib = (ctypes.c_int * 4)(1, 14, 1, pid)
    data = ctypes.create_string_buffer(4096)
    size = ctypes.c_size_t(ctypes.sizeof(data))
    if query(mib, 4, data, ctypes.byref(size), None, 0):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if size.value == 0:
        return None
    info = Prefix.from_buffer(data)
    if size.value < ctypes.sizeof(Prefix) or info.pid != pid:
        raise ValueError("Invalid Darwin process snapshot")
    seconds, microseconds = info.start.seconds, info.start.microseconds
    identity = (
        (seconds & 0xFFFFFFFF, seconds >> 32, microseconds, 0)
        if seconds > 0 and 0 <= microseconds < 1_000_000
        else None
    )
    return info.state, identity


def process_identity(pid):
    try:
        if sys.platform == "linux":
            return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        if sys.platform == "darwin":
            info = (ctypes.c_uint32 * 34)()
            inspect = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True).proc_pidinfo
            inspect.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint64,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            inspect.restype = ctypes.c_int
            if (
                inspect(pid, 3, 0, info, ctypes.sizeof(info)) == ctypes.sizeof(info)
                and info[3] == pid
            ):
                return ":".join(map(str, info[30:34]))
    except (OSError, IndexError, ValueError):
        pass
    return None


def kill_owned(pid, identity):
    if identity is not None and process_identity(pid) == identity:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
