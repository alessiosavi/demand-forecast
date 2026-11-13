"""Independent raw-array refits and a selectively downloaded guest job, offline."""

import copy
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from scripts.corpus_v4 import refit_matches as refit  # noqa: E402
from scripts.corpus_v4.audit_fev_overlap import BenchmarkSeries, BenchmarkStore  # noqa: E402
from scripts.corpus_v4.make_corpus_job import ENVIRONMENT  # noqa: E402
from scripts.corpus_v4.make_refit_job import render  # noqa: E402


def fixture(tmp_path, x, y, *, offset=0, column="target", channel=0):
    root = tmp_path / "corpus"
    folder = "covariates/demo" if column == "covariates" else "univariate/demo"
    path = root / folder / "train-00002.parquet"
    path.parent.mkdir(parents=True)
    rows = []
    for item in ("decoy", "wanted"):
        row = dict(item_id=item, source="fixture", freq="h", target=[x.tolist()])
        if column == "covariates":
            row.update(
                target=np.zeros(len(x)).tolist(), covariates=[np.zeros(len(x)).tolist(), x.tolist()]
            )
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1)
    spool = tmp_path / "benchmark.sqlite"
    store = BenchmarkStore(
        spool,
        [
            BenchmarkSeries("fixture", "decoy", "flag", "h", np.ones(len(y))),
            BenchmarkSeries("fixture", "wanted", "flag", "h", y),
        ],
    )
    store.db.close()
    start, end = max(0, offset), min(len(x), len(y) + offset)
    count = int((np.isfinite(x[start:end]) & np.isfinite(y[start - offset : end - offset])).sum())
    record = dict(
        corpus=dict(
            shard=path.name,
            row=1,
            column=column,
            channel=channel,
            item_id="wanted",
            source="fixture",
            freq="h",
        ),
        benchmark=dict(dataset="fixture", item_id="wanted", column="flag", freq="h"),
        offset=offset,
        fit=dict(
            t_start=start,
            t_end=end,
            index_start=start - offset,
            index_end=end - offset,
            segment=count,
            r2=0.05,
            exact_frac=0.99,
            max_resid=1.0,
        ),
    )
    report = dict(folders=[dict(folder=folder, status="allowed", matches=[record])])
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    return report_path, root, spool, tmp_path / "out", report


@pytest.mark.parametrize("offset", [0, 3, -4])
def test_affine_copy_keeps_offsets_and_joint_mask_without_verifier(tmp_path, monkeypatch, offset):
    y = np.zeros(140)
    y[[8, 30, 61, 98, 126]] = [1, 2, 3, 4, 5]
    y[10] = np.nan
    x = np.full(145, np.nan)
    start, end = max(0, offset), min(len(x), len(y) + offset)
    x[start:end] = -3 * y[start - offset : end - offset] + 7
    x[20] = np.nan
    args = fixture(tmp_path, x, y, offset=offset)

    def forbidden(*args, **kwargs):
        pytest.fail("independent refit called the verifier")

    monkeypatch.setattr("scripts.corpus_v4.audit_sketch.verify_candidate", forbidden)
    # Cover every callable in the disputed module, including indirect helpers.
    from demand_forecast.pretrain import audit_match

    for name, value in vars(audit_match).copy().items():
        if callable(value):
            monkeypatch.setattr(audit_match, name, forbidden)
    assert refit.main([str(p) for p in args[:4]]) == 0
    result = json.loads((args[3] / "refit.json").read_text())
    rec = result["records"][0]
    assert rec["alignment_ok"] is True
    assert rec["alignment"]["finite_pairs"] == args[4]["folders"][0]["matches"][0]["fit"]["segment"]
    for direction in ("benchmark_on_corpus", "corpus_on_benchmark"):
        assert rec[direction]["r2"] == pytest.approx(1)
        assert rec[direction]["pearson"] == pytest.approx(-1)
    assert rec["benchmark_on_corpus"]["coef"] == pytest.approx([-1 / 3, 7 / 3])
    assert rec["off_modal"]["intersection_count"] == 5
    assert rec["off_modal"]["benchmark_reproduced_frac"] == 1
    assert rec["exact_equal_frac"] == 0
    assert rec["benchmark_on_corpus"]["within_range_frac"] == 1
    assert rec["reported_fit"] == dict(r2=0.05, exact_frac=0.99, max_resid=1)
    assert len(rec["first_pairs"]) == 120
    assert rec["benchmark_off_modal_pairs"][0]["benchmark_index"] == 8
    assert rec["benchmark_off_modal_pairs"][0]["corpus_index"] == 8 + offset
    assert result["summary"]["r2_at_least_half"] == 1
    assert result["summary"]["majority_benchmark_off_modal"] == 1
    assert result["summary"]["r2_differs_over_001"] == 1
    assert (args[3] / "refit.md").read_text().splitlines()[:3] == [
        "Pairs with r² ≥ 0.5 in either direction: 1/1",
        "Pairs reproducing > half the benchmark off-modal positions: 1/1",
        "Pairs differing from reported r² by > 0.01: 1/1",
    ]


def test_independent_sparse_flags_have_chance_overlap_despite_raw_agreement(tmp_path):
    x, y = np.zeros(10000), np.zeros(10000)
    x[:100] = 1
    y[np.r_[0, np.arange(100, 199)]] = 1
    args = fixture(tmp_path, x, y, column="covariates", channel=1)
    assert refit.main([str(p) for p in args[:4]]) == 0
    rec = json.loads((args[3] / "refit.json").read_text())["records"][0]
    assert abs(rec["benchmark_on_corpus"]["r2"]) < 1e-12
    assert abs(rec["benchmark_on_corpus"]["pearson"]) < 1e-12
    assert rec["off_modal"]["intersection_count"] == 1
    assert rec["off_modal"]["expected_intersection"] == 1
    assert rec["off_modal"]["benchmark_reproduced_frac"] == 0.01
    assert rec["corpus_stats"] == dict(distinct=2, mode=0, mode_count=9900, mode_share=0.99)
    assert rec["benchmark_stats"] == rec["corpus_stats"]
    assert rec["exact_equal_frac"] == pytest.approx(0.9802)
    assert rec["off_modal"]["benchmark_positions"] == [0, *range(100, 199)]


def test_samples_are_capped_but_all_off_modal_positions_are_retained(tmp_path):
    y = np.r_[np.zeros(600), np.arange(1, 301)]
    args = fixture(tmp_path, y.copy(), y)
    assert refit.main([str(p) for p in args[:4]]) == 0
    rec = json.loads((args[3] / "refit.json").read_text())["records"][0]
    assert len(rec["first_pairs"]) == 120
    assert len(rec["benchmark_off_modal_pairs"]) == 200
    assert rec["benchmark_off_modal_pairs"][-1]["benchmark_index"] == 799
    assert rec["off_modal"]["benchmark_positions"] == list(range(600, 900))


def test_constant_target_has_undefined_r2_and_no_off_modal_reproduction(tmp_path):
    args = fixture(tmp_path, np.arange(32.0), np.ones(32))
    assert refit.main([str(p) for p in args[:4]]) == 0
    result = json.loads((args[3] / "refit.json").read_text())
    rec = result["records"][0]
    assert rec["benchmark_on_corpus"]["r2"] is None
    assert rec["benchmark_on_corpus"]["pearson"] is None
    assert rec["corpus_on_benchmark"]["r2"] == 0
    assert rec["off_modal"]["benchmark_reproduced_frac"] is None
    assert result["summary"]["majority_benchmark_off_modal"] == 0


@pytest.mark.parametrize("defect", ["segment", "bounds", "offset", "owner", "item", "row"])
def test_bad_alignment_is_recorded_and_fails_cli_without_skipping_other_records(tmp_path, defect):
    args = fixture(tmp_path, np.arange(32.0), np.arange(32.0))
    bad = copy.deepcopy(args[4]["folders"][0]["matches"][0])
    if defect == "segment":
        bad["fit"]["segment"] += 1
    elif defect == "bounds":
        bad["fit"]["t_end"] += 1
        bad["fit"]["index_end"] += 1
    elif defect == "offset":
        bad["offset"] = 1
    elif defect == "owner":
        bad["benchmark"]["item_id"] = "missing"
    elif defect == "item":
        bad["corpus"]["item_id"] = "other"
    else:
        bad["corpus"]["row"] = 100
    args[4]["folders"][0]["matches"].insert(0, bad)
    args[0].write_text(json.dumps(args[4]))
    result = subprocess.run(
        [sys.executable, "-m", "scripts.corpus_v4.refit_matches", *map(str, args[:4])],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stderr
    payload = json.loads((args[3] / "refit.json").read_text())
    assert len(payload["records"]) == 2
    assert payload["records"][0]["alignment_ok"] is False
    assert payload["records"][0]["error"]
    assert payload["records"][1]["alignment_ok"] is True
    assert payload["summary"]["alignment_failures"] == 1
    assert "ERROR" in (args[3] / "refit.md").read_text()
    if defect == "segment":
        assert payload["records"][0]["alignment"]["finite_pairs"] == 32


def test_job_syntax_and_only_requested_shards(tmp_path):
    args = fixture(tmp_path, np.arange(32.0), np.arange(32.0))
    assert refit.needed_shards(args[4]) == ["univariate/demo/train-00002.parquet"]
    source = render(
        "a" * 40,
        "gs://bucket/report.json",
        "gs://bucket/corpus",
        "gs://bucket/spool",
        "gs://bucket/refit",
    )
    job = tmp_path / "job.sh"
    job.write_text(source)
    result = subprocess.run(["bash", "-n", str(job)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("output", ["gs://bucket/corpus", "gs://bucket/spool/sub", "gs://bucket"])
def test_job_rejects_outputs_overwriting_inputs(output):
    with pytest.raises(ValueError):
        render(
            "a" * 40, "gs://bucket/report.json", "gs://bucket/corpus", "gs://bucket/spool", output
        )


@pytest.mark.parametrize("failure", [None, "alignment", "upload"])
def test_guest_selective_transfers_publication_and_exit_status(tmp_path, failure):
    args = fixture(tmp_path, np.arange(32.0), np.arange(32.0))
    if failure == "alignment":
        args[4]["folders"][0]["matches"][0]["fit"]["segment"] += 1
        args[0].write_text(json.dumps(args[4]))
    remote = tmp_path / "remote"
    remote.mkdir()
    shutil.copyfile(args[0], remote / "report.json")
    shutil.copytree(args[1], remote / "corpus")
    spool = remote / "spool"
    spool.mkdir()
    for suffix in (".sqlite", ".values", ".stamps"):
        shutil.copyfile(args[2].with_suffix(suffix), spool / ("benchmark" + suffix))
    # This file exists remotely but must never be downloaded.
    (remote / "corpus/univariate/demo/train-unused.parquet").write_text("not a shard")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name in ("python", "python3"):
        path = binaries / name
        path.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
        path.chmod(0o755)
    git = binaries / "git"
    git.write_text("""#!/bin/sh
case "$1" in
clone) mkdir -p "$4/.git" ;;
rev-parse) printf '%s\\n' aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ;;
esac
""")
    git.chmod(0o755)
    cloud = binaries / "gcloud"
    cloud.write_text(
        f"#!{sys.executable}\n"
        + """import json, os, pathlib, shutil, sys
args = sys.argv[1:]
assert args[:2] == ["storage", "cp"], args
source, destination = args[2:]
with open(os.environ["TRANSFERS"], "a") as log:
    log.write(json.dumps([source, destination]) + "\\n")
if os.environ["FAILURE"] == "upload" and destination.startswith("gs://bucket/refit/"):
    sys.exit(21)
def local(path):
    return pathlib.Path(os.environ["REMOTE"]) / path.removeprefix("gs://bucket/") if path.startswith("gs://") else pathlib.Path(path)
target = local(destination)
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(local(source), target)
"""
    )
    cloud.chmod(0o755)
    job = tmp_path / "job.sh"
    # The unchanged shared installer is covered by the bootstrap suite; this
    # offline shell test uses the already prepared interpreter instead.
    job.write_text(
        render(
            "a" * 40,
            "gs://bucket/report.json",
            "gs://bucket/corpus",
            "gs://bucket/spool",
            "gs://bucket/refit",
        ).replace(ENVIRONMENT, "")
    )
    guest, transfers = tmp_path / "guest", tmp_path / "transfers.jsonl"
    env = dict(
        os.environ,
        RUN_DIR=str(guest),
        STATE_URI="gs://bucket/state",
        REMOTE=str(remote),
        TRANSFERS=str(transfers),
        FAILURE=failure or "",
        PATH=str(binaries) + os.pathsep + os.environ["PATH"],
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
    )
    result = subprocess.run(["bash", str(job)], env=env, capture_output=True, text=True)
    assert result.returncode == (
        1 if failure == "alignment" else 3 if failure == "upload" else 0
    ), result.stderr
    payload = json.loads((guest / "state/refit.json").read_text())
    assert payload["summary"]["alignment_failures"] == (1 if failure == "alignment" else 0)
    for name in ("refit.json", "refit.md"):
        assert (remote / "state" / name).read_bytes() == (guest / "state" / name).read_bytes()
        if failure != "upload":
            assert (remote / "refit" / name).read_bytes() == (guest / "state" / name).read_bytes()
    copied = [json.loads(line) for line in transfers.read_text().splitlines()]
    assert [a for a, b in copied if a.startswith("gs://bucket/corpus/")] == [
        "gs://bucket/corpus/univariate/demo/train-00002.parquet"
    ]
