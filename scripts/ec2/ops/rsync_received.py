#!/usr/bin/env python3
"""Read conservative received-byte counts from GNU rsync and macOS openrsync."""

import re
import sys
from pathlib import Path


def received_bytes(text: str) -> int:
    patterns = (
        r"Total bytes received:\s*([\d,]+)",
        r"Total received:\s*([\d,]+)\s+B\b",
        r"\breceived\s+([\d,]+)\s+bytes\b",
    )
    counts = [int(n.replace(",", "")) for pattern in patterns for n in re.findall(pattern, text)]
    if not counts:
        raise ValueError("No received-byte statistics; transfer hold must remain reserved")
    return max(counts)


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError("usage: rsync_received.py <rsync log>")
        print(received_bytes(Path(sys.argv[1]).read_text()))
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)
