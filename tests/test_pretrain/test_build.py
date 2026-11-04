"""Offline conversion, provenance, filtering and byte-level reproducibility."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
datasets = pytest.importorskip("datasets")

from demand_forecast.cli import app  # noqa: E402
from demand_forecast.pretrain.build import build_corpus, source_files  # noqa: E402
from demand_forecast.pretrain.build_config import CorpusBuildConfig  # noqa: E402
from demand_forecast.pretrain.corpus import (  # noqa: E402
    corpus_manifest,
    read_shard,
    verify_manifest,
)


def _entry(key, **kwargs):
    return dict(
        licence="CC BY 4.0",
        bucket="yes",
        attribution="Fixture authors",
        url="https://example.org/data",
        **{key: kwargs.pop("name")},
        **kwargs,
    )


@pytest.fixture
def offline_recipe(tmp_path, monkeypatch):
    import huggingface_hub

    def no_download(*args, **kwargs):
        raise AssertionError("tests must not download")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_download)
    chronos, gift = tmp_path / "chronos", tmp_path / "gift"
    folder = chronos / "real"
    folder.mkdir(parents=True)
    kept = np.arange(80, dtype=np.float32)
    kept[5] = np.nan
    duplicate = kept.copy()
    duplicate.view(np.uint32)[5] = 0xFFC00001
    rows = [
        ("z_unique", np.arange(80) + 100),
        ("b_duplicate", duplicate),
        ("a_keep", kept),
        ("c_all_nan", np.full(80, np.nan)),
        ("d_short", np.arange(10)),
        ("e_constant", np.ones(80)),
        ("f_few_finite", np.r_[np.arange(20), np.full(60, np.nan)]),
    ]
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(
                    id=name,
                    timestamp=pd.date_range("2020-01-01", periods=len(target), freq="D").tolist(),
                    target=np.asarray(target, dtype=np.float32),
                    ignored="extra",
                )
                for name, target in rows
            ]
        ),
        folder / "train.parquet",
    )
    datasets.Dataset.from_list(
        [
            dict(
                item_id="one",
                start=pd.Timestamp("2021-01-03"),
                freq="W",
                target=[kept.tolist(), (np.arange(80) + 200).tolist(), np.ones(80).tolist()],
                past_feat_dynamic_real=[[1.0] * 80],
            )
        ]
    ).save_to_disk(str(gift / "multi"))
    datasets.Dataset.from_list(
        [
            dict(
                item_id="two",
                start=pd.Timestamp("2021-01-03"),
                freq="h",
                target=(np.arange(80) + 300).astype(np.float32),
            ),
            dict(
                item_id="nan",
                start=pd.Timestamp("2021-01-03"),
                freq="h",
                target=np.full(80, np.nan, dtype=np.float32),
            ),
        ]
    ).save_to_disk(str(gift / "single"))
    config = dict(
        version=2,
        commercial_use=False,
        sources=[
            dict(
                name="chronos_real",
                repo="fixture/chronos",
                revision="a" * 40,
                kind="chronos_real",
                local_path=str(chronos),
                include=[_entry("config", name="real")],
                excluded={},
            ),
            dict(
                name="gifteval_pretrain",
                repo="fixture/gift",
                revision="b" * 40,
                kind="gifteval_pretrain",
                bundle_terms="research-only (dataset card)",
                local_path=str(gift),
                include=[_entry("folder", name="single"), _entry("folder", name="multi")],
                excluded={},
            ),
        ],
        filters=dict(
            min_points=64,
            drop_all_nan=True,
            drop_constant=True,
            dedupe="value_hash",
            max_series_per_folder=None,
        ),
        shards=dict(max_series=1, max_bytes=10000),
    )
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _outputs(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_build_reproducible_filters_multivariate_and_verification(offline_recipe, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    report = build_corpus(offline_recipe, first)
    build_corpus(offline_recipe, second)
    assert _outputs(first) == _outputs(second)
    before = _outputs(first)
    build_corpus(offline_recipe, first)
    assert _outputs(first) == before
    assert {"manifest.json", "corpus.json", "NOTICE-corpus.md", "duplicates.jsonl"} <= before.keys()
    assert report["totals"] == dict(
        series_seen=12,
        series_kept=4,
        points_kept=320,
        finite_points_kept=319,
        freq_fallback=0,
        duplicate_pairs_count=2,
        dropped=dict(all_nan=2, min_points=2, constant=2, duplicate=2, subsample=0),
    )
    pairs = [json.loads(line) for line in (first / "duplicates.jsonl").read_text().splitlines()]
    assert pairs == [
        dict(kept_id="chronos_real/real/a_keep", dropped_id="chronos_real/real/b_duplicate"),
        dict(kept_id="chronos_real/real/a_keep", dropped_id="gifteval_pretrain/multi/one/0"),
    ]
    refs = corpus_manifest(first)
    assert len(refs) == 4
    rows = [row for ref in refs for row in pq.read_table(ref.path).to_pylist()]
    assert [row["id"] for row in rows] == [
        "chronos_real/real/a_keep",
        "chronos_real/real/z_unique",
        "gifteval_pretrain/multi/one/1",
        "gifteval_pretrain/single/two",
    ]
    for ref in refs:
        assert ref.size <= 10000
        table = pq.read_table(ref.path)
        assert table.schema.field("target").type == pa.list_(pa.float32())
        assert table.schema.field("start").type == pa.timestamp("ns")
        assert table.schema.metadata is None
        assert read_shard(ref.path)[0].start == pd.Timestamp("2000-01-02")
    assert rows[0]["start"] == pd.Timestamp("2020-01-01")
    assert [row["freq"] for row in rows] == ["D", "D", "W", "h"]
    assert np.isnan(rows[0]["target"][5])
    assert verify_manifest(first) == {"chronos_real": (2, 0), "gifteval_pretrain": (2, 0)}
    runner = CliRunner()
    assert runner.invoke(app, ["pretrain", "corpus", "verify", "--root", str(first)]).exit_code == 0
    broken = refs[0].path
    data = bytearray(broken.read_bytes())
    data[20] ^= 1
    broken.write_bytes(data)
    result = runner.invoke(app, ["pretrain", "corpus", "verify", "--root", str(first)])
    assert result.exit_code == 1
    assert "chronos_real: 1 verified, 1 failed" in result.output
    broken.unlink()
    assert verify_manifest(first)["chronos_real"] == (1, 1)
    assert "commercial_use: false" in (first / "NOTICE-corpus.md").read_text()
    assert len(report["config_sha256"]) == 64
    saved_report = json.loads((first / "corpus.json").read_text())
    assert saved_report["pandas_version"] == pd.__version__
    assert saved_report["pyarrow_version"] == pa.__version__
    assert all(folder["freq_fallback"] == 0 for s in report["sources"] for folder in s["folders"])


def _chronos_recipe(offline_recipe, rows):
    raw = yaml.safe_load(offline_recipe.read_text())
    raw["sources"] = raw["sources"][:1]
    raw["filters"]["min_points"] = 2
    path = Path(raw["sources"][0]["local_path"]) / "real/train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    offline_recipe.write_text(yaml.safe_dump(raw))


@pytest.mark.parametrize("drop_business", [False, True], ids=["both-kept", "count-before-filter"])
def test_chronos_business_day_frequency_fallback(offline_recipe, tmp_path, drop_business):
    # Ordinary Mon-Fri calendars infer as B. A missing holiday makes inference fail.
    business = pd.bdate_range("2020-01-06", periods=81).delete(8)
    regular = pd.date_range("2020-01-01", periods=80, freq="D")
    assert pd.infer_freq(business) is None
    assert all(business.day_of_week < 5)
    _chronos_recipe(
        offline_recipe,
        [
            dict(
                id="business",
                timestamp=business.tolist(),
                target=np.ones(80) if drop_business else np.arange(80),
            ),
            dict(id="regular", timestamp=regular.tolist(), target=np.arange(80) + 100),
        ],
    )
    root = tmp_path / "frequencies"
    report = build_corpus(offline_recipe, root)
    folder = report["sources"][0]["folders"][0]
    assert folder["series_seen"] == 2
    assert folder["series_kept"] == 2 - drop_business
    assert folder["dropped"]["constant"] == int(drop_business)
    assert folder["freq_fallback"] == 1
    assert report["sources"][0]["totals"]["freq_fallback"] == 1
    assert report["totals"]["freq_fallback"] == 1
    rows = {
        row["id"]: row
        for ref in corpus_manifest(root)
        for row in pq.read_table(ref.path).to_pylist()
    }
    assert rows["chronos_real/real/regular"]["freq"] == pd.infer_freq(regular)
    if not drop_business:
        row = rows["chronos_real/real/business"]
        assert row["freq"] == pd.tseries.frequencies.to_offset(pd.Timedelta(days=1)).freqstr
        assert row["start"] == business[0]
        assert row["target"] == list(range(80))


@pytest.mark.parametrize(
    ("days", "expected_days"),
    [([0, 2, 3, 5, 6], 1), ([0, 1, 3, 5, 7], 2), ([0, 2], 2), ([0, 0, 0, 1, 1, 2], 1)],
    ids=["tie-smallest", "mode-not-smallest", "two-timestamps", "ignore-zero-deltas"],
)
def test_chronos_frequency_fallback_deltas(offline_recipe, tmp_path, days, expected_days):
    timestamps = pd.Timestamp("2020-01-01") + pd.to_timedelta(days, unit="D")
    _chronos_recipe(
        offline_recipe,
        [dict(id="deltas", timestamp=timestamps.tolist(), target=list(range(len(days))))],
    )
    root = tmp_path / "deltas"
    report = build_corpus(offline_recipe, root)
    row = pq.read_table(corpus_manifest(root)[0].path).to_pylist()[0]
    assert row["freq"] == pd.tseries.frequencies.to_offset(pd.Timedelta(days=expected_days)).freqstr
    assert report["totals"]["freq_fallback"] == 1


@pytest.mark.parametrize(
    ("days", "error"),
    [([0, 2, 1], "non-monotonic"), ([2, 1, 0], "non-monotonic"), ([0, 0, 0], "no positive")],
    ids=["out-of-order", "descending", "no-positive-delta"],
)
def test_chronos_invalid_timestamps_name_series(offline_recipe, tmp_path, days, error):
    timestamps = pd.Timestamp("2020-01-01") + pd.to_timedelta(days, unit="D")
    _chronos_recipe(
        offline_recipe,
        [dict(id="bad_calendar", timestamp=timestamps.tolist(), target=list(range(len(days))))],
    )
    with pytest.raises(ValueError, match=rf"{error}.*chronos_real/real/bad_calendar"):
        build_corpus(offline_recipe, tmp_path / "invalid")


def test_cli_build_dry_run_and_subset(offline_recipe, tmp_path):
    runner = CliRunner()
    root = tmp_path / "cli"
    args = ["pretrain", "corpus", "build", "--config", str(offline_recipe), "--root", str(root)]
    result = runner.invoke(app, args + ["--dry-run", "--source", "chronos_real"])
    assert result.exit_code == 0, result.output
    assert "real/train.parquet" in result.output and "bytes" in result.output
    assert "gifteval_pretrain" not in result.output
    assert not root.exists()
    result = runner.invoke(app, args + ["--source", "chronos_real", "--keep-downloads"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert set(verify_manifest(root)) == {"chronos_real"}
    # A partial second build must not quietly defeat cross-kind deduplication.
    result = runner.invoke(app, args + ["--source", "gifteval_pretrain"])
    assert result.exit_code != 0 and "unselected kind" in result.output
    assert runner.invoke(app, args + ["--source", "unknown"]).exit_code != 0


def test_hub_calls_use_pins_and_allowlist(offline_recipe, tmp_path, monkeypatch):
    import huggingface_hub

    raw = yaml.safe_load(offline_recipe.read_text())
    source = raw["sources"][0]
    local = Path(source.pop("local_path"))
    raw["sources"] = [source]
    offline_recipe.write_text(yaml.safe_dump(raw))
    calls = []

    class API:
        def list_repo_tree(self, repo, **kwargs):
            calls.append((repo, kwargs))
            return [
                SimpleNamespace(path="real/train.parquet", size=42),
                SimpleNamespace(path="excluded/file.parquet", size=99),
            ]

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    cfg = CorpusBuildConfig.from_yaml(offline_recipe)
    assert source_files(cfg.sources[0]) == [dict(path="real/train.parquet", size=42)]
    assert calls[0][1]["revision"] == "a" * 40
    downloads = []

    def download(**kwargs):
        import shutil

        downloads.append(kwargs)
        shutil.copytree(local, kwargs["local_dir"])
        # A stale parquet from another revision must not enter this build.
        (kwargs["local_dir"] / "real/stale.parquet").write_bytes(b"not a parquet")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    root = tmp_path / "download"
    build_corpus(offline_recipe, root)
    assert downloads == [
        dict(
            repo_id="fixture/chronos",
            repo_type="dataset",
            revision="a" * 40,
            allow_patterns=["real/*.parquet"],
            local_dir=root / "downloads/chronos_real",
        )
    ]
    assert not (root / "downloads/chronos_real").exists()
    build_corpus(offline_recipe, root, keep_downloads=True)
    assert (root / "downloads/chronos_real/real/train.parquet").exists()


def test_mapping_subsampling_and_atomic_conversion_failure(offline_recipe, tmp_path):
    raw = yaml.safe_load(offline_recipe.read_text())
    path = Path(raw["sources"][0]["local_path"]) / "real/train.parquet"
    table = pq.read_table(path)
    pq.write_table(table.rename_columns(["id", "timestamp", "consumption_kW", "ignored"]), path)
    raw["sources"][0]["include"][0]["target_column"] = "consumption_kW"
    raw["filters"]["max_series_per_folder"] = 1
    offline_recipe.write_text(yaml.safe_dump(raw))
    root = tmp_path / "mapped"
    report = build_corpus(offline_recipe, root)
    assert report["totals"]["dropped"]["subsample"] == 1
    before = _outputs(root)
    build_corpus(offline_recipe, root)
    assert _outputs(root) == before
    raw["shards"]["max_bytes"] = 1
    offline_recipe.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="exceeds shards.max_bytes"):
        build_corpus(offline_recipe, root)
    assert _outputs(root) == before


def test_legacy_shards_are_preserved(offline_recipe, tmp_path):
    from tests.helpers.corpus_fixture import write_corpus_fixture

    root = tmp_path / "mixed"
    legacy = write_corpus_fixture(root)
    before = legacy.read_bytes()
    build_corpus(offline_recipe, root)
    assert legacy.read_bytes() == before
    assert "tsmixup" in verify_manifest(root)
