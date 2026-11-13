#!/usr/bin/env python3
"""Repeat offline acceptance and the two A7 loads; keep evidence in one directory.

python tests/test_ec2/verify_cancellation.py --output /private/tmp/ec2-cancel-check
The interpreter must have pytest; --ruff can select a separate Ruff installation.
"""

import argparse
import concurrent.futures
import ctypes
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "checkout": {"stall_contains": "git checkout"},
    "dispatch": {"stall_contains": "dispatch-job.sh"},
    "scp": {"stall_scp": True},
    "rsync": {"stall_rsync": True},
    "interrupt_false": False,
    "interrupt_true": True,
}


def load_ops():
    spec = importlib.util.spec_from_file_location("test_ops", ROOT / "tests/test_ec2/test_ops.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stress_worker(out, rounds, cases):
    ops = load_ops()
    records = []
    for iteration in range(rounds):
        for case in cases:
            work = out / f"{iteration:03d}-{case}"
            work.mkdir(parents=True)
            operator = ops.Operator(work)
            started = time.monotonic()
            record = {"case": case, "iteration": iteration, "outcome": "passed"}
            try:
                if case.startswith("interrupt"):
                    ops.test_real_guard_interrupt_during_copy_advances_to_cleanup(
                        operator, CASES[case]
                    )
                else:
                    ops.test_real_guard_stalled_remote_reaches_single_cleanup(operator, CASES[case])
            except BaseException:
                record.update(outcome="failed", error=traceback.format_exc())
            record["seconds"] = round(time.monotonic() - started, 3)
            state = json.loads(operator.state_path.read_text())
            ledger = json.loads(operator.ledger_path.read_text())
            record.update(
                live_leaves=[
                    state[key]
                    for key in ("stalled_pid", "stalled_ppid")
                    if state.get(key) and ops.process_alive(state[key])
                ],
                stops=sum(t["command"] == "stop" for t in ledger["transactions"]),
                reconciles=sum(t["command"] == "reconcile" for t in ledger["transactions"]),
                closed=all(s["status"] == "closed" for s in ledger["sessions"]),
            )
            if (
                record["live_leaves"]
                or (record["stops"], record["reconciles"]) != (1, 1)
                or not record["closed"]
            ):
                record["outcome"] = "failed"
            records.append(record)
            (out / "results.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(record), flush=True)
    return int(any(row["outcome"] != "passed" for row in records))


def process_snapshot():
    """Live PID -> (PPID, creation identity); never execute ps."""
    result = {}
    if sys.platform == "linux":
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            except FileNotFoundError:
                continue
            if fields[0] != "Z":
                result[int(entry.name)] = (int(fields[1]), fields[19])
    elif sys.platform == "darwin":
        lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        lib.proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        capacity = 4096
        while True:
            pids = (ctypes.c_int * capacity)()
            used = lib.proc_listpids(1, 0, pids, ctypes.sizeof(pids))
            if used <= 0:
                raise OSError("Unable to enumerate processes")
            if used < ctypes.sizeof(pids):
                break
            capacity *= 2
        for pid in pids[: used // ctypes.sizeof(ctypes.c_int)]:
            info = (ctypes.c_uint32 * 34)()  # 136-byte proc_bsdinfo, flavor 3.
            if lib.proc_pidinfo(pid, 3, 0, info, ctypes.sizeof(info)) == ctypes.sizeof(info):
                if info[1] != 5:
                    result[pid] = (info[4], tuple(info[30:34]))
    else:
        raise RuntimeError("Leftover inspection requires Linux /proc or Darwin libproc")
    return result


class ProcessTracker:
    def __init__(self):
        self.known = {os.getpid(): process_snapshot()[os.getpid()][1]}
        self.stop = threading.Event()
        self.errors = []
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def scan(self):
        snapshot = process_snapshot()
        parents = {
            pid
            for pid, identity in self.known.items()
            if snapshot.get(pid, (None, None))[1] == identity
        }
        while True:
            added = {
                pid for pid, (ppid, _identity) in snapshot.items() if ppid in parents
            } - parents
            if not added:
                break
            parents.update(added)
        for pid in parents:
            self.known[pid] = snapshot[pid][1]
        return sorted(
            pid
            for pid, identity in self.known.items()
            if pid != os.getpid() and snapshot.get(pid, (None, None))[1] == identity
        )

    def watch(self):
        while not self.stop.wait(0.2):
            try:
                self.scan()
            except Exception as exc:
                self.errors.append(str(exc))
                return


def run_command(argv, log, timeout=900, cwd=ROOT):
    started = time.monotonic()
    with log.open("w") as stream:
        try:
            rc = subprocess.run(
                argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout
            ).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    text = log.read_text()
    summary = next(
        (line for line in reversed(text.splitlines()) if "passed" in line or "failed" in line), ""
    )
    return {
        "command": list(map(str, argv)),
        "exit": rc,
        "seconds": round(time.monotonic() - started, 2),
        "summary": summary,
        "log": str(log),
    }


def stress(out, parallel, rounds, cases, cpu=False):
    out.mkdir()
    loops = []
    try:
        if cpu:
            for _ in range(28):
                loops.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-c",
                            "import time; end=time.monotonic()+150\nwhile time.monotonic()<end: sum(i*i for i in range(10000))",
                        ]
                    )
                )

        def worker(index):
            target = out / f"worker-{index}"
            target.mkdir()
            return run_command(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--output",
                    str(target),
                    "--rounds",
                    str(rounds),
                    "--cases",
                    ",".join(cases),
                ],
                out / f"worker-{index}.log",
                timeout=1800,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
            commands = list(pool.map(worker, range(parallel)))
    finally:
        for child in loops:
            if child.poll() is None:
                child.terminate()
            child.wait()
    rows = [
        row for path in out.glob("worker-*/results.json") for row in json.loads(path.read_text())
    ]
    return {
        "cases": len(rows),
        "expected": parallel * rounds * len(cases),
        "failures": sum(row["outcome"] != "passed" for row in rows),
        "per_case": {
            case: {
                "cases": sum(row["case"] == case for row in rows),
                "failures": sum(row["case"] == case and row["outcome"] != "passed" for row in rows),
            }
            for case in cases
        },
        "commands": commands,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ruff", default="/private/tmp/ruff-latest/bin/ruff")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--rounds", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument("--cases", default=",".join(CASES), help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.worker:
        return stress_worker(out, args.rounds, args.cases.split(","))
    out.mkdir(parents=True, exist_ok=False)
    tracker = ProcessTracker()
    tracker.thread.start()
    result = {"python": sys.executable, "root": str(ROOT), "stages": {}}

    def record(name, value):
        result["stages"][name] = value
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            name,
            json.dumps(
                {
                    key: value[key]
                    for key in ("exit", "summary", "cases", "failures")
                    if key in value
                }
            ),
            flush=True,
        )

    sources = [ROOT / "scripts/ec2/ops/process_control.py", *Path(__file__).parent.glob("*.py")]
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    try:
        for iteration in range(1, 4):
            record(
                f"acceptance-{iteration}",
                run_command(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "tests/test_ec2",
                        "tests/test_bench/test_ec2_bootstrap.py",
                        "-q",
                        "--basetemp",
                        str(out / f"pytest-{iteration}"),
                    ],
                    out / f"acceptance-{iteration}.log",
                ),
            )
        record("stress-16", stress(out / "stress-16", 16, 3, list(CASES)))
        record(
            "stress-cpu",
            stress(out / "stress-cpu", 8, 5, ["checkout", "dispatch", "interrupt_false"], cpu=True),
        )
        record("ruff-check", run_command([args.ruff, "check", "."], out / "ruff-check.log"))
        record(
            "ruff-format",
            run_command([args.ruff, "format", "--check", "."], out / "ruff-format.log"),
        )
    finally:
        tracker.stop.set()
        tracker.thread.join()
        result["leftover_pids"] = tracker.scan()
        result["inspection_errors"] = tracker.errors
        result["source_sha256"] = hashes
        result["sources_unchanged"] = all(
            hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
            for path, digest in hashes.items()
        )
        result["passed"] = (
            len(result["stages"]) == 7
            and not result["leftover_pids"]
            and not tracker.errors
            and result["sources_unchanged"]
            and all(
                value.get("exit", 0) == 0
                and value.get("failures", 0) == 0
                and value.get("cases") == value.get("expected")
                and all(cmd["exit"] == 0 for cmd in value.get("commands", []))
                for value in result["stages"].values()
            )
        )
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"result={out / 'result.json'} passed={result['passed']} leftovers={result['leftover_pids']}"
    )
    return int(not result["passed"])


if __name__ == "__main__":
    sys.exit(main())
