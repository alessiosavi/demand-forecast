#!/usr/bin/env python3
"""Run a resumable manifest in three light lanes, draining before each heavy job."""

import json
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def load_manifest(path: Path) -> list[dict]:
    items = json.loads(path.read_text())
    if not isinstance(items, list) or not items:
        raise ValueError("Manifest must be a nonempty list")
    names = set()
    for item in items:
        if not isinstance(item, dict) or not re.fullmatch(r"[a-zA-Z0-9_-]+", item.get("name", "")):
            raise ValueError("Each item needs a safe, unique name")
        if item["name"] in names or item.get("lane") not in ("light", "heavy"):
            raise ValueError("Duplicate name or invalid lane")
        names.add(item["name"])
        argv = shlex.split(item["cmd"]) if isinstance(item.get("cmd"), str) else item.get("cmd")
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(s, str) or "\0" in s for s in argv)
        ):
            raise ValueError("cmd must be a nonempty argv array or shell-quoted argv string")
        if argv[:3] in (["demand-forecast", "fev", "run"], ["demand-forecast", "fev", "tune"]):
            if "--resume" not in argv:
                argv.append("--resume")
        item["argv"] = argv
    return items


def run_item(item: dict, state: Path) -> bool:
    name, argv = item["name"], item["argv"]
    marker = state / f"{name}.json"
    # Let fev --resume validate its artifacts/inputs even after a prior success.
    # A wrapper-level success cache could hide changed configs or lost outputs.
    print(f"WINS_START {name} lane={item['lane']}", flush=True)
    with (state / f"{name}.log").open("a") as log:
        log.write(f"\nCOMMAND {shlex.join(argv)}\n")
        log.flush()
        try:
            rc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
        except OSError as exc:
            log.write(f"{exc}\n")
            rc = 127
    temporary = marker.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(dict(argv=argv, exit=rc)) + "\n")
    temporary.replace(marker)
    print(f"WINS_EXIT {name} exit={rc}", flush=True)
    return rc == 0


def run(stage: str, manifest: Path) -> int:
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", stage):
        raise ValueError("Invalid stage")
    items = load_manifest(manifest)
    root = Path(os.environ.get("WINS_STATE_DIR") or str(Path(os.environ["OPS"]) / "wins"))
    state = root / stage
    state.mkdir(parents=True, exist_ok=True)
    # A second queue must not overlap the first, including its tuning studies.
    import fcntl

    with (root / "queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another queue is already running under {root}") from exc
        results = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            pending = []
            for item in items:
                if item["lane"] == "heavy":
                    results.extend(job.result() for job in pending)
                    pending.clear()
                    results.append(run_item(item, state))
                else:
                    pending.append(pool.submit(run_item, item, state))
            results.extend(job.result() for job in pending)
        ok = sum(results)
        print(f"WINS_DONE ok={ok} failed={len(results) - ok}", flush=True)
        return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(run(sys.argv[1], Path(sys.argv[2])))
    except (OSError, ValueError, KeyError) as exc:
        print(f"WINS_ERROR {exc}", file=sys.stderr)
        sys.exit(2)
