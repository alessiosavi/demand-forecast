#!/usr/bin/env python3
"""Sample Linux task sessions once a second; these are sampled maxima, not peaks."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path


def parse_sample(processes: str, gpu_processes: str, session: int) -> list[tuple[int, int, int]]:
    """Return PID, RSS KiB, GPU MiB for the task's entire process session."""
    gpu = {}
    for line in gpu_processes.splitlines():
        if line.strip():
            pid, memory = line.split(",")
            gpu[int(pid)] = int(memory)
    rows = []
    for line in processes.splitlines():
        pid, sid, rss = map(int, line.split())
        if sid == session:
            rows.append((pid, rss, gpu.get(pid, 0)))
    return rows


def query(*command):
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=10).stdout


def monitor(session: int, directory: Path, exit_file: Path):
    with (
        (directory / "resources.csv").open("x", buffering=1) as per_pid,
        (directory / "device.csv").open("x", buffering=1) as device,
    ):
        writer, device_writer = csv.writer(per_pid), csv.writer(device)
        writer.writerow(["timestamp", "pid", "rss_KiB", "gpu_MiB"])
        device_writer.writerow(["timestamp", "device", "memory_MiB"])
        deadline = time.monotonic()
        while True:
            stamp = time.time()
            rows = parse_sample(
                query("ps", "-eo", "pid=,sid=,rss="),
                query(
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ),
                session,
            )
            writer.writerows((stamp, *row) for row in rows)
            devices = query(
                "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"
            )
            for line in devices.splitlines():
                index, memory = map(int, line.split(","))
                device_writer.writerow([stamp, index, memory])
            if exit_file.exists():
                return
            if not rows:
                raise RuntimeError(
                    "task session disappeared without an exit file; completion unknown"
                )
            deadline += 1
            time.sleep(max(0, deadline - time.monotonic()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--exit-file", type=Path, required=True)
    args = parser.parse_args()
    status = 1
    try:
        monitor(args.session, args.directory, args.exit_file)
        status = 0
    finally:
        (args.directory / "sampler.exit").write_text(f"{status}\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
