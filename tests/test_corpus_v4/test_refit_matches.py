"""Independent raw-array refits and a selectively downloaded guest job, offline."""

import copy
import hashlib
import json
import math
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


def fixture(tmp_path, x, y, *, offset=0, column="target", channel=0, proposal="prefix/24"):
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
        proposal=proposal,
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
    report = dict(
        input_fingerprint="synthetic-fingerprint",
        benchmark_revision="synthetic-benchmark-revision",
        method={"name": "synthetic-retrieval", "window": 24},
        folders=[dict(folder=folder, status="allowed", matches=[record])],
    )
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
    assert result["summary"]["strong_outside_retrieval_window"] == 1
    assert result["summary"]["r2_differs_over_001"] == 1
    markdown = (args[3] / "refit.md").read_text()
    assert "Pairs with r² ≥ 0.5 in either direction: 1/1" in markdown
    assert "Pairs with strong off-modal association outside retrieval window: 1/1" in markdown
    assert "Pairs differing from reported r² by > 0.01: 1/1" in markdown
    assert "descriptive, not a significance test" in markdown


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
    assert rec["off_modal"]["phi"] is None
    assert rec["off_modal"]["hypergeom_upper_tail"] == 1
    assert result["summary"]["strong_outside_retrieval_window"] == 0


@pytest.mark.parametrize("proposal", ["prefix/64", "suffix/64"])
def test_copy_association_is_perfect_inside_and_outside_retrieval_window(tmp_path, proposal):
    y = np.tile([0.0, 0, 0, 1], 32)
    args = fixture(tmp_path, 3 * y + 7, y, proposal=proposal)
    assert refit.main([str(p) for p in args[:4]]) == 0
    payload = json.loads((args[3] / "refit.json").read_text())
    rec = payload["records"][0]
    assert rec["off_modal"]["phi"] == 1
    assert rec["off_modal"]["expected_intersection"] == 8
    assert rec["off_modal"]["hypergeom_upper_tail"] == pytest.approx(
        1 / math.comb(128, 32), rel=1e-12, abs=0
    )
    assert rec["retrieval_window"]["proposal"] == proposal
    for region in ("inside", "outside"):
        stats = rec["retrieval_window"][region]
        assert stats["finite_pairs"] == 64
        assert (
            stats["corpus_count"] == stats["benchmark_count"] == stats["intersection_count"] == 16
        )
        assert stats["expected_intersection"] == 4
        assert stats["phi"] == 1
        assert stats["hypergeom_upper_tail"] == pytest.approx(
            1 / math.comb(64, 16), rel=1e-12, abs=0
        )
    assert payload["summary"]["strong_outside_retrieval_window"] == 1
    assert rec["retrieval_end_run"]["affine"]["finite_pairs"] == 128
    assert rec["retrieval_end_run"]["affine"]["coef"] == pytest.approx([1 / 3, -7 / 3])
    assert rec["retrieval_end_run"]["exact"]["finite_pairs"] == 0


def test_unrelated_dense_series_have_high_raw_overlap_but_zero_phi(tmp_path):
    x = np.tile([0.0, 1, 2, 3], 64)
    y = np.tile(np.repeat([0.0, 1, 2, 3], 4), 16)
    args = fixture(tmp_path, x, y, proposal="prefix/64")
    assert refit.main([str(p) for p in args[:4]]) == 0
    payload = json.loads((args[3] / "refit.json").read_text())
    rec = payload["records"][0]
    assert rec["off_modal"]["benchmark_reproduced_frac"] == 0.75
    assert rec["off_modal"]["intersection_count"] == 144
    assert rec["off_modal"]["expected_intersection"] == 144
    assert rec["off_modal"]["phi"] == 0
    assert rec["retrieval_window"]["outside"]["phi"] == 0
    assert payload["summary"]["strong_outside_retrieval_window"] == 0


def test_hypergeometric_tail_includes_the_observed_intersection(tmp_path):
    x, y = np.zeros(8), np.zeros(8)
    x[[0, 1, 2]], y[[0, 1, 3]] = 1, 1
    args = fixture(tmp_path, x, y)
    assert refit.main([str(p) for p in args[:4]]) == 0
    off = json.loads((args[3] / "refit.json").read_text())["records"][0]["off_modal"]
    assert off["expected_intersection"] == 9 / 8
    assert off["phi"] == pytest.approx(7 / 15)
    # Of C(8,3)=56 placements, C(3,2)*C(5,1)+C(3,3)*C(5,0)=16 share >=2.
    assert off["hypergeom_upper_tail"] == pytest.approx(2 / 7)


@pytest.mark.parametrize(
    ("outside_length", "events", "shared", "strong"),
    [(64, 1, 1, 0), (16, 4, 4, 0), (32, 16, 12, 1), (32, 16, 11, 0)],
)
def test_strong_headline_requires_effect_size_events_and_remainder(
    tmp_path, outside_length, events, shared, strong
):
    x, y = np.zeros(64 + outside_length), np.zeros(64 + outside_length)
    x[64 : 64 + events] = 1
    y[64 : 64 + shared] = 1
    y[64 + events : 64 + 2 * events - shared] = 1
    args = fixture(tmp_path, x, y, proposal="prefix/64")
    assert refit.main([str(p) for p in args[:4]]) == 0
    payload = json.loads((args[3] / "refit.json").read_text())
    assert payload["summary"]["strong_outside_retrieval_window"] == strong


def test_outside_events_keep_full_overlap_modes_when_local_mode_changes(tmp_path):
    x = np.r_[np.zeros(64), np.ones(24), np.zeros(8)]
    y = np.r_[np.zeros(64), np.ones(16), np.zeros(8), np.ones(8)]
    args = fixture(tmp_path, x, y, proposal="prefix/64")
    assert refit.main([str(p) for p in args[:4]]) == 0
    rec = json.loads((args[3] / "refit.json").read_text())["records"][0]
    outside = rec["retrieval_window"]["outside"]
    assert outside["corpus_count"] == outside["benchmark_count"] == 24
    assert outside["intersection_count"] == 16
    assert outside["expected_intersection"] == 18
    assert outside["phi"] == pytest.approx(-1 / 3)


@pytest.mark.parametrize("proposal", ["prefix/64", "suffix/64"])
def test_agreement_only_inside_retrieval_window_does_not_count_as_strong(tmp_path, proposal):
    x = np.tile([0.0, 0, 0, 1], 48)
    y = np.r_[x[:64], np.tile(np.repeat([0.0, 0, 0, 1], 4), 8)]
    if proposal.startswith("suffix"):
        x, y = x[::-1], y[::-1]
    args = fixture(tmp_path, x, y, proposal=proposal)
    assert refit.main([str(p) for p in args[:4]]) == 0
    payload = json.loads((args[3] / "refit.json").read_text())
    rec = payload["records"][0]
    assert rec["retrieval_window"]["inside"]["phi"] == 1
    assert rec["retrieval_window"]["outside"]["phi"] == 0
    assert payload["summary"]["strong_outside_retrieval_window"] == 0
    run = rec["retrieval_end_run"]
    assert run["affine"]["finite_pairs"] == run["exact"]["finite_pairs"] == 67
    assert run["affine"]["position_start"] == (0 if proposal.startswith("prefix") else 125)
    assert run["affine"]["position_end"] == (67 if proposal.startswith("prefix") else 192)


@pytest.mark.parametrize("proposal", ["identity", "calendar", None, "prefix/0", "prefix/24/extra"])
def test_other_proposals_report_only_whole_overlap(tmp_path, proposal):
    y = np.tile([0.0, 0, 0, 1], 32)
    args = fixture(tmp_path, y.copy(), y, proposal=proposal)
    assert refit.main([str(p) for p in args[:4]]) == 0
    payload = json.loads((args[3] / "refit.json").read_text())
    rec = payload["records"][0]
    assert rec["off_modal"]["phi"] == 1
    assert rec["retrieval_window"]["outside"] is None
    assert rec["retrieval_end_run"] is None
    assert payload["summary"]["strong_outside_retrieval_window"] == 0
    assert "whole overlap only" in (args[3] / "refit.md").read_text()


@pytest.mark.parametrize("proposal", ["prefix/8", "suffix/8", "prefix/999"])
def test_window_bounds_use_unmasked_positions_and_empty_outside_is_undefined(tmp_path, proposal):
    y = np.tile([0.0, 0, 0, 1], 8)
    y[[1, 7, 24]] = np.nan
    args = fixture(tmp_path, y.copy(), y, proposal=proposal)
    assert refit.main([str(p) for p in args[:4]]) == 0
    rec = json.loads((args[3] / "refit.json").read_text())["records"][0]
    window = rec["retrieval_window"]
    assert (
        window["inside"]["finite_pairs"]
        == {"prefix/8": 6, "suffix/8": 7, "prefix/999": 29}[proposal]
    )
    assert (
        window["outside"]["finite_pairs"]
        == {"prefix/8": 23, "suffix/8": 22, "prefix/999": 0}[proposal]
    )
    if proposal == "prefix/999":
        assert window["outside"]["phi"] is None
        assert window["outside"]["expected_intersection"] is None
        assert window["outside"]["hypergeom_upper_tail"] is None
    assert rec["retrieval_end_run"]["exact"]["position_start"] == 0
    assert rec["retrieval_end_run"]["exact"]["position_end"] == 32


def test_artifacts_bind_report_spool_and_deduplicated_shards_to_provisional_refit(tmp_path):
    y = np.tile([0.0, 0, 0, 1], 32)
    args = fixture(tmp_path, y.copy(), y)
    args[4]["folders"][0]["matches"] *= 2
    args[0].write_text(json.dumps(args[4], indent=3) + "\n")
    assert refit.main([*map(str, args[:4]), "--repo-commit", "b" * 40]) == 0
    payload = json.loads((args[3] / "refit.json").read_text())
    assert payload["artifact_type"] == "corpus-match-refit"
    assert payload["provisional"] is True
    assert payload["release_eligible"] is False
    assert "descriptive, not a significance test" in json.dumps(payload)
    for key in ("input_fingerprint", "benchmark_revision", "method"):
        assert payload[key] == args[4][key]
    assert payload["repository_commit"] == "b" * 40
    provenance = payload["provenance"]
    files = [
        (provenance["report"], args[0]),
        (provenance["spool_sqlite"], args[2]),
        (provenance["spool_values"], args[2].with_suffix(".values")),
    ]
    assert len(provenance["shards"]) == 1
    shard = provenance["shards"][0]
    assert shard["path"] == "univariate/demo/train-00002.parquet"
    files.append((shard, args[1] / shard["path"]))
    markdown = (args[3] / "refit.md").read_text()
    for entry, path in files:
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry["sha256"] == expected
        assert expected in markdown
    for value in (
        "corpus-match-refit",
        "synthetic-fingerprint",
        "synthetic-benchmark-revision",
        "synthetic-retrieval",
        "b" * 40,
    ):
        assert value in markdown
    assert "provisional: true" in markdown
    assert "release_eligible: false" in markdown


@pytest.mark.parametrize(
    "folder",
    [
        "univariate/../x",
        "univariate/x/y",
        "other/x",
        "univariate/.x",
        "univariate/",
        "/univariate/x",
        "covariates/x\n",
    ],
)
def test_shard_path_rejects_noncanonical_folders(folder):
    with pytest.raises(ValueError, match="invalid native folder"):
        refit._shard_path({"folder": folder}, {"corpus": {"shard": "train-0.parquet"}})


@pytest.mark.parametrize(
    "shard",
    [
        "../train-0.parquet",
        "test-0.parquet",
        "train-.parquet",
        "/train-0.parquet",
        "train-0/1.parquet",
        "train-0.parquet\n",
    ],
)
def test_shard_path_rejects_noncanonical_shards(shard):
    with pytest.raises(ValueError, match="invalid native shard"):
        refit._shard_path({"folder": "univariate/x"}, {"corpus": {"shard": shard}})


@pytest.mark.parametrize("folder", ["univariate/x", "covariates/Demo_2.v1-a"])
@pytest.mark.parametrize("shard", ["train-a_0.parquet", "validation-a-1.parquet"])
def test_allowed_and_quarantined_shard_paths_are_distinct_and_deduplicated(folder, shard):
    record = {"corpus": {"shard": shard}}
    report = {
        "folders": [
            {"folder": folder, "status": status, "matches": [record, record]}
            for status in ("allowed", "quarantined")
        ]
    }
    assert refit.needed_shards(report) == sorted(
        [f"quarantine/{folder}/{shard}", f"{folder}/{shard}"]
    )


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
    assert payload["repository_commit"] == "a" * 40
    assert payload["summary"]["alignment_failures"] == (1 if failure == "alignment" else 0)
    for name in ("refit.json", "refit.md"):
        assert (remote / "state" / name).read_bytes() == (guest / "state" / name).read_bytes()
        if failure != "upload":
            assert (remote / "refit" / name).read_bytes() == (guest / "state" / name).read_bytes()
    copied = [json.loads(line) for line in transfers.read_text().splitlines()]
    assert [a for a, b in copied if a.startswith("gs://bucket/corpus/")] == [
        "gs://bucket/corpus/univariate/demo/train-00002.parquet"
    ]
