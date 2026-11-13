"""Offline crowded-index profile; synthetic values only, no downloads.

Run with the corpus extra installed, e.g.:
python -m scripts.corpus_v4.profile_sketch /private/tmp/sketch-profile \
    --baseline /private/tmp/sketch-baseline.py

Save the pre-change audit_sketch.py separately for --baseline. No Git mutation
or source-identity bypass is involved. Timings exclude fixture/index preparation.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.corpus_v4 import audit_sketch as current
from scripts.corpus_v4.audit_fev_overlap import BenchmarkSeries

MIXES = (("worldtrace", 400, 96), ("factorynet", 800, 48), ("borg", 3000, 24), ("load", 8000, 12))
REFERENCE = dict(
    channels=3072,
    records=18672,
    posting_cap=3270,
    numerical_fit=8894,
    shortcut=6477,
)


def _module(path):
    if path is None:
        return current
    spec = importlib.util.spec_from_file_location("sketch_profile_baseline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _benchmarks(seed, groups, decoys):
    rng = np.random.default_rng(seed)
    lengths = (100, 400, 800, 3000, 8000, 100000)
    for group in range(groups):
        for side in range(2):
            for member in range(4):
                # Two step prices and two informative histories per key.
                n = lengths[(group + member) % len(lengths)]
                level = float(10 + group)
                if member < 2:
                    x = np.full(n, level)
                    x[n // 3 : 2 * n // 3] += 1 + member
                else:
                    x = rng.lognormal(0, 2.5, n)
                if side == 0:
                    x[:24] = level
                    x[-24:] = level + 0.375 + member
                else:
                    x[-24:] = level
                    x[:24] = level + 0.625 + member
                yield BenchmarkSeries("prices-flags", f"{group}-{side}-{member}", "x", "h", x)
    for i in range(decoys):
        # Strongly skewed quantiles put thousands of unrelated series in the
        # same summary bucket, without making endpoint hits artificially common.
        n = (400, 800, 3000, 8000)[i % 4]
        yield BenchmarkSeries("summary-crowd", str(i), "x", "h", rng.lognormal(0, 2.5, n))


def _corpus(path, seed, groups, length, channels):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed + length)
    rows = []
    for i in range(channels):
        x = rng.lognormal(0, 2.5, length)
        level = float(10 + i % groups)
        # Most channels hit one endpoint; a quarter hit both. Occasional longer
        # plateaus exercise 64/256-point families too. No retrieval is mocked.
        width = 64 if i % 11 == 0 else 256 if i % 31 == 0 else 24
        if i % 4 != 1:
            x[:width] = level
        if i % 4 in (0, 1):
            x[-width:] = level
        if i % 19 == 0:
            x[length // 2] = np.nan
        rows.append(dict(item_id=str(i), source="synthetic", freq="h", target=[x.tolist()]))
    path.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path / "train-000.parquet", row_group_size=12)


def _pad_index(directory, postings):
    """Model the real index's byte size without sketching a million histories.

    Extra sorted postings have an unqueried all-ones key and an existing owner;
    they never enter a bucket used by these fixtures. This models search cost,
    NOT a million independent numerical histories or their working set.
    """
    path = directory / "postings.bin"
    actual = path.stat().st_size // current.POSTING.itemsize
    if postings <= actual:
        return actual
    with path.open("ab") as stream:
        block = np.zeros(min(65536, postings - actual), dtype=current.POSTING)
        block["hi"] = np.iinfo(np.uint64).max
        block["lo"] = np.iinfo(np.uint64).max
        block["owner"] = 1
        for start in range(actual, postings, len(block)):
            stream.write(block[: min(len(block), postings - start)].tobytes())
    return postings


def _record_mix(directory):
    import gzip

    counts, families, postings = Counter(), Counter(), []
    for path in directory.glob("*.jsonl.gz"):
        with gzip.open(path, "rt") as stream:
            for line in stream:
                rec = json.loads(line)
                counts["records"] += 1
                if "unresolved" in rec:
                    counts["posting_cap"] += 1
                    postings.append(rec["unresolved"]["postings"])
                elif "fit" in rec:
                    counts["shortcut" if rec.get("shortcut") else "numerical_fit"] += 1
                    families[rec["proposal"]] += 1
    return dict(
        **counts,
        median_capped_postings=float(np.median(postings)) if postings else 0,
        families=dict(families),
    )


def _worker(spec):
    module = _module(spec.get("baseline"))
    request = spec["request"]
    # Explicit timers avoid cProfile's cross-thread event attribution on some
    # Python 3.13 builds (the native RSS watchdog sleeps on another thread).
    stages, calls = Counter(), Counter()
    originals = []

    def instrument(obj, name, stage):
        original = getattr(obj, name)
        originals.append((obj, name, original))

        @functools.wraps(original)
        def measured(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                stages[stage] += time.perf_counter() - start
                calls[stage] += 1

        setattr(obj, name, measured)

    for obj, name, stage in (
        (module, "sketch", "sketch"),
        (module.SketchIndex, "lookup", "lookup"),
        (module, "_series", "benchmark_series_load"),
        (module, "verify_candidate", "shortcut"),
        (module.old, "fit_segment", "numerical_fit"),
        (module.MemoryBudget, "check", "memory_check"),
        (module._UnitJournal, "write", "journal_write"),
    ):
        instrument(obj, name, stage)
    start = time.perf_counter()
    try:
        result = module.scan_folder(request)
        elapsed = time.perf_counter() - start
    finally:
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)
    stages["shortcut"] -= stages["numerical_fit"]
    stages["everything_else"] = elapsed - sum(stages.values())
    directory = Path(request["directory"])
    (directory / "result.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    if spec.get("controls"):
        controls = module._check_controls(result=result, **spec["controls"])
        (directory / "controls.json").write_text(
            json.dumps(controls, sort_keys=True, indent=2) + "\n"
        )
        if not controls["ok"]:
            raise AssertionError(controls)
    channels = result["counts"]["channels"]
    print(
        json.dumps(
            dict(
                seconds=elapsed,
                seconds_per_channel=elapsed / channels,
                stages_per_channel={k: v / channels for k, v in stages.items()},
                calls=dict(calls),
                counts=result["counts"],
                mix=_record_mix(directory),
            )
        )
    )


def _compare(before, after):
    names = sorted(
        p.name for p in before.iterdir() if p.name.endswith((".jsonl", ".jsonl.gz", ".json"))
    )
    other = sorted(
        p.name for p in after.iterdir() if p.name.endswith((".jsonl", ".jsonl.gz", ".json"))
    )
    differences = sorted(set(names) ^ set(other)) + [
        name
        for name in names
        if name in other and (before / name).read_bytes() != (after / name).read_bytes()
    ]
    return dict(identical=not differences, files=names, differences=differences)


def _controls(out, seed):
    import pandas as pd

    rng = np.random.default_rng(seed)
    values = rng.normal(30, 8, 512)
    masked = rng.normal(30, 8, 400)
    masked[::17] = np.nan
    benchmarks = [
        BenchmarkSeries("identifying", "whole", "x", "h", values),
        BenchmarkSeries(
            "masked-monthly",
            "monthly",
            "x",
            "ME",
            masked,
            pd.date_range("2000", periods=len(masked), freq="ME"),
        ),
        BenchmarkSeries("degenerate", "price", "x", "h", np.repeat([10.0, 11.0, 10.0], 100)),
        # The constant-one negative control collides with these 0/1 flag
        # endpoints; the unchanged verifier rejects the joint overlaps.
        BenchmarkSeries("flags", "snap", "x", "h", np.repeat([1.0, 0.0, 1.0], 100)),
    ]
    spool, identity = current.prepare_benchmarks(out / "control-state", benchmarks)
    index = out / "control-index"
    current.build_index(spool, index, identity, memory_budget_bytes=4 * 1024**3)
    folder = out / "controls"
    cases, unplantable, skipped = current._control_shards(spool, folder, 1 << 80)
    return folder, spool, index, dict(cases=cases, unplantable=unplantable, skipped=skipped)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=405)
    parser.add_argument("--groups", type=int, default=256)
    parser.add_argument("--decoys", type=int, default=8192)
    parser.add_argument("--index-postings", type=int, default=20_000_000)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--channel-scale", type=float, default=1)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("out must be a new directory (do not accidentally reuse a stale fixture)")
    if min(args.groups, args.decoys, args.repeats, args.channel_scale) <= 0:
        parser.error("groups, decoys, repeats and channel-scale must be positive")
    args.out.mkdir(parents=True)
    spool, identity = current.prepare_benchmarks(
        args.out / "state", _benchmarks(args.seed, args.groups, args.decoys)
    )
    index = args.out / "index"
    current.build_index(spool, index, identity, memory_budget_bytes=4 * 1024**3)
    postings = _pad_index(index, args.index_postings)
    measurements = []
    workloads = []
    for shape, length, channels in MIXES:
        channels = max(1, round(channels * args.channel_scale))
        folder = args.out / shape
        _corpus(folder, args.seed, args.groups, length, channels)
        workloads.append((shape, folder, spool, index, None))
    workloads.append(("controls", *_controls(args.out, args.seed)))
    for shape, folder, spool, index, controls in workloads:
        for repeat in range(args.repeats):
            for version in ("before", "after"):
                destination = args.out / f"{shape}-{version}-{repeat}"
                request = dict(
                    entry=dict(folder="univariate/" + shape, status="allowed"),
                    directory=str(destination),
                    folder_path=str(folder),
                    spool=str(spool),
                    index=str(index),
                    opportunities=1 << 80,
                    budget=4 * 1024**3,
                )
                if controls:
                    request.update(
                        control=True,
                        control_benchmarks={
                            c["item_id"]: c["benchmark"] for c in controls["cases"]
                        },
                    )
                spec = dict(
                    request=request,
                    baseline=str(args.baseline) if version == "before" else None,
                    controls=controls,
                )
                process = subprocess.run(
                    [sys.executable, "-m", "scripts.corpus_v4.profile_sketch", "--worker"],
                    input=json.dumps(spec),
                    text=True,
                    capture_output=True,
                    check=True,
                )
                record = dict(
                    shape=shape, version=version, repeat=repeat, **json.loads(process.stdout)
                )
                measurements.append(record)
                print(json.dumps(record), flush=True)
            comparison = _compare(
                args.out / f"{shape}-before-{repeat}", args.out / f"{shape}-after-{repeat}"
            )
            print(json.dumps(dict(shape=shape, repeat=repeat, comparison=comparison)), flush=True)
            if not comparison["identical"]:
                raise AssertionError(comparison)
    worldtrace = next(r for r in measurements if r["shape"] == "worldtrace")
    actual_mix = {
        k: worldtrace["mix"][k] / worldtrace["counts"]["channels"]
        for k in REFERENCE
        if k != "channels"
    }
    reference_mix = {k: v / REFERENCE["channels"] for k, v in REFERENCE.items() if k != "channels"}
    report = dict(
        seed=args.seed,
        benchmark_series=identity["count"],
        index_postings=postings,
        reference_worldtrace_per_channel=reference_mix,
        actual_worldtrace_per_channel=actual_mix,
        worldtrace_mix_percent_difference={
            k: 100 * (v / reference_mix[k] - 1) for k, v in actual_mix.items()
        },
        measurements=measurements,
        journals_byte_identical=True,
        limitations=[
            "Index padding models byte size, not a million distinct numerical histories.",
            "Explicit wall timers include instrumentation overhead; cold downloads and multi-worker scaling are unmeasured.",
            "shortcut includes precheck and post-fit gates; numerical_fit includes its own degeneracy/information gates.",
            "Endpoint proposals are mostly 24-point hits; the real parts have additional 64/256-point proposals. Other folder names denote lengths, not fitted record distributions.",
        ],
    )
    (args.out / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        _worker(json.load(sys.stdin))
    else:
        raise SystemExit(main())
