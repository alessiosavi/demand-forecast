"""Reproducible tiny offline profiling harness (not an extrapolated cloud result).

Run: python -m tests.test_corpus_v4.test_audit_sketch_benchmark /tmp/sketch-bench
Baseline source is read from the local pinned commit; no network or Git mutations.
"""

import argparse
import cProfile
import importlib.util
import json
import pstats
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Collected in the environment without the bench extra too: skip there instead of failing collection.
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from scripts.corpus_v4 import audit_sketch as current  # noqa: E402
from scripts.corpus_v4.audit_fev_overlap import BenchmarkSeries  # noqa: E402


def _profile_worker(specification):
    module = current
    if specification["version"] == "before":
        spec = importlib.util.spec_from_file_location("sketch_baseline", specification["baseline"])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    request = specification["request"]
    profile = cProfile.Profile()
    start = time.perf_counter()
    result = profile.runcall(module.scan_folder, request)
    elapsed = time.perf_counter() - start
    stats = pstats.Stats(profile)

    def stage(file, name):
        return sum(
            v[3] for (f, _, n), v in stats.stats.items() if Path(f).name == file and n == name
        )

    stages = {
        name: stage(file, function)
        for name, file, function in [
            ("decode", "native.py", "iter_rows"),
            ("sketch", Path(module.__file__).name, "sketch"),
            ("lookup", Path(module.__file__).name, "lookup"),
            ("series", Path(module.__file__).name, "_series"),
            ("verify", Path(module.__file__).name, "verify_candidate"),
            ("emit", Path(module.__file__).name, "emit"),
            ("ranking", Path(module.__file__).name, "_rank_key"),
            ("assembly", Path(module.__file__).name, "_assemble"),
        ]
    }
    destination = Path(request["directory"])
    rows, values = specification["channels"], specification["values"]
    record = dict(
        shape=specification["shape"],
        version=specification["version"],
        repeat=specification["repeat"],
        channels=rows,
        values=values,
        seconds=elapsed,
        us_per_channel=elapsed / rows * 1e6,
        ns_per_value=elapsed / values * 1e9,
        stages=stages,
        counts=result["counts"],
        evidence_bytes=sum(p.stat().st_size for p in destination.glob("*.jsonl*")),
    )
    profile.dump_stats(str(destination / "profile.pstats"))
    print(json.dumps(record))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    baseline_file = args.out / "baseline.py"
    source = subprocess.run(
        ["git", "show", "0523bc6:scripts/corpus_v4/audit_sketch.py"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    baseline_file.write_text(source)
    rng = np.random.default_rng(44)
    benchmarks = [
        BenchmarkSeries(
            "fixture", str(i), "target", "h", np.ones(8192) if i < 48 else rng.normal(30, 8, 8192)
        )
        for i in range(64)
    ]
    spool, identity = current.prepare_benchmarks(args.out / "state", benchmarks)
    current.build_index(spool, args.out / "index", identity, memory_budget_bytes=4 * 1024**3)
    records = []
    for shape, rows, length in [("short", 120, 336), ("long", 24, 8192), ("constant", 36, 8192)]:
        folder = args.out / shape
        folder.mkdir(exist_ok=True)
        values = [
            np.ones(length) if shape == "constant" else rng.normal(30, 8, length)
            for _ in range(rows)
        ]
        pq.write_table(
            pa.Table.from_pylist(
                [
                    dict(item_id=str(i), source=shape, freq="h", target=[v.tolist()])
                    for i, v in enumerate(values)
                ]
            ),
            folder / "train-000.parquet",
            row_group_size=12,
        )
        for version in ("before", "after"):
            for repeat in range(args.repeats):
                destination = args.out / f"{shape}-{version}-{repeat}"
                destination.mkdir(exist_ok=True)
                request = dict(
                    entry=dict(folder="univariate/" + shape, status="allowed"),
                    directory=str(destination),
                    folder_path=str(folder),
                    spool=str(spool),
                    index=str(args.out / "index"),
                    opportunities=1 << 80,
                    budget=4 * 1024**3,
                )
                specification = dict(
                    version=version,
                    baseline=str(baseline_file),
                    request=request,
                    shape=shape,
                    repeat=repeat,
                    channels=rows,
                    values=rows * length,
                )
                # A fresh process matches production and avoids accumulating watchdog threads.
                process = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "tests.test_corpus_v4.test_audit_sketch_benchmark",
                        "--worker",
                    ],
                    input=json.dumps(specification),
                    text=True,
                    capture_output=True,
                    check=True,
                )
                record = json.loads(process.stdout)
                records.append(record)
                print(json.dumps(record), flush=True)
    (args.out / "measurements.json").write_text(json.dumps(records, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        _profile_worker(json.load(sys.stdin))
    else:
        raise SystemExit(main())
