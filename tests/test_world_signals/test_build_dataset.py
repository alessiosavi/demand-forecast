import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml

from demand_forecast.bench.specs import load_task_specs
from scripts.world_signals import build_dataset as builder
from scripts.world_signals import fetch_news, fetch_signals, sources
from scripts.world_signals.fetch_prices import PANEL_SYMBOLS
from scripts.world_signals.fev_io import read_fev_parquet


@pytest.fixture
def inputs():
    grid = pd.date_range("2021-12-01", "2025-12-31")
    frames = []
    for i, asset in enumerate(PANEL_SYMBOLS):
        dates = grid[grid >= "2021-12-10"] if asset == "SOL" else grid
        if asset in {"GOLD", "SILVER"}:
            dates = dates[(dates.dayofweek < 5) & (dates != "2022-01-17")]
        frames.append(
            pd.DataFrame(
                {"id": asset, "date": dates, "close": 100 + i + np.arange(len(dates), dtype=float)}
            )
        )
    prices = pd.concat(frames, ignore_index=True)
    prices.attrs["missing_quote_dates"] = {"GOLD": ["2022-01-17"]}
    articles = fetch_news.normalize_articles(
        [
            pd.DataFrame(
                {
                    "id": ["a", "b", "c"],
                    "newsDatetime": ["2021-12-01", "2022-01-04", "2025-12-01"],
                    "currencies": ["BTC,ETH", "BTC", "BTC,ETH"],
                }
            )
        ]
    )
    cov = fetch_news.coverage(articles, PANEL_SYMBOLS)
    news = fetch_news.aggregate_daily(articles, PANEL_SYMBOLS, cov)
    signals = pd.DataFrame(
        {
            "date": grid,
            **{name: np.arange(len(grid), dtype=float) for name in builder.GLOBAL_COLUMNS},
        }
    )
    signals.loc[signals.date == "2022-01-02", "fng"] = np.nan
    calendar = pd.DataFrame(
        {
            "date": grid,
            "holiday_us": (grid.day == 1).astype(float),
            "holiday_gf": 0.0,
            "fomc_day": 0.0,
        }
    )
    return builder.BuildInputs(
        prices=prices,
        news=news,
        signals=signals,
        calendar=calendar,
        gdelt={"keep": False, "errors": []},
    )


@pytest.fixture
def bundle(tmp_path, inputs):
    out = tmp_path / "built"
    builder.write_bundle(
        inputs, out, start="2021-12-01", end="2025-12-31", raw_dir=tmp_path / "raw"
    )
    return out


def test_on_off_same_schema_off_all_nan(bundle):
    on, off = [pq.read_table(bundle / f"panel_{arm}.parquet") for arm in ("on", "off")]
    assert on.schema == off.schema
    assert on["id"].to_pylist() == sorted(PANEL_SYMBOLS)
    manifest = json.loads((bundle / "manifest.json").read_text())
    roles = manifest["features"]
    for name in ("id", "timestamp", "asset_class"):
        assert on[name].equals(off[name])
    for left, right in zip(on["target"].to_pylist(), off["target"].to_pylist(), strict=True):
        np.testing.assert_equal(left, right)
    for name in roles["known_dynamic"] + roles["past_dynamic"]:
        assert all(np.isnan(row).all() for row in off[name].to_pylist())
    assert not any(name.startswith("gdelt_") for name in on.column_names)
    for arm in ("on", "off"):
        tasks = yaml.safe_load((bundle / f"probe_{arm}.yaml").read_text())["tasks"]
        assert len(tasks) == 4
        for task in tasks:
            assert task["known_dynamic_columns"] == roles["known_dynamic"]
            assert task["past_dynamic_columns"] == roles["past_dynamic"]
            assert isinstance(task["initial_cutoff"], str)
            assert task["extra_metrics"] == ["MASE", "WAPE", "WQL"]
            assert task["seasonality"] == 7
    profiles = yaml.safe_load((bundle / "profiles.yaml").read_text())
    assert len(profiles) == 9
    assert all(
        p == {"target_support": "real", "categorical_statics": ["asset_class"]}
        for p in profiles.values()
    )


def test_relative_paths_resolve_after_copy(bundle, tmp_path):
    copied = tmp_path / "copy"
    shutil.copytree(bundle, copied)
    shutil.rmtree(bundle)
    for arm in ("on", "off", "smoke"):
        specs = load_task_specs(copied / f"probe_{arm}.yaml", copied / "profiles.yaml")
        for spec in specs:
            assert Path(spec.dataset_path).parent == copied
            assert Path(spec.dataset_path).is_file()


@pytest.mark.bench
def test_future_dates_equal_grid(bundle, tmp_path, monkeypatch):
    import datasets

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")
    result = builder.validate_bundle_windows(bundle)
    assert result == {
        "tasks": 9,
        "windows": 162,
        "series_windows": 1620,
        "ids": sorted(PANEL_SYMBOLS),
    }


def test_no_lookahead(inputs):
    cutoff = pd.Timestamp("2022-01-03")
    before, _, _ = builder.build_long(inputs, "2021-12-01", "2025-12-31")
    signals = inputs.signals.copy()
    signals.loc[signals.date > cutoff, builder.GLOBAL_COLUMNS] = 99999.0
    articles = fetch_news.normalize_articles(
        [
            pd.DataFrame(
                {
                    "id": ["a", "b", "c", "sentinel"],
                    "newsDatetime": [
                        "2021-12-01",
                        "2022-01-04",
                        "2025-12-01",
                        "2022-01-04T00:00:00Z",
                    ],
                    "currencies": ["BTC,ETH", "BTC", "BTC,ETH", "BTC,ETH"],
                }
            )
        ]
    )
    news = fetch_news.aggregate_daily(
        articles, PANEL_SYMBOLS, fetch_news.coverage(articles, PANEL_SYMBOLS)
    )
    prices = inputs.prices.copy()
    prices.loc[prices.date > cutoff, "close"] *= 2
    after, _, _ = builder.build_long(
        replace(inputs, signals=signals, news=news, prices=prices), "2021-12-01", "2025-12-31"
    )
    pd.testing.assert_frame_equal(
        before.loc[before.timestamp <= cutoff], after.loc[after.timestamp <= cutoff]
    )
    assert not before.equals(after)  # Sentinel is effective in the future.


def test_metals_target_nan_off_session(inputs):
    panel, statics, prices = builder.build_long(inputs, "2021-12-01", "2025-12-31")
    gold = panel.loc[panel.id == "GOLD"].set_index("timestamp")
    close = prices.loc[prices.id == "GOLD"].set_index("date")
    assert gold.loc[gold.index.dayofweek >= 5, "target"].isna().all()
    assert np.isnan(gold.loc["2022-01-17", "target"])
    assert gold.loc["2022-01-10", "target"] == pytest.approx(
        np.log(close.loc["2022-01-10", "close"] / close.loc["2022-01-07", "close"])
    )
    assert close.loc["2022-01-17", "quote_missing"]
    assert not close.loc["2022-01-17", "session"]
    assert not close.loc["2022-01-08", "quote_missing"]
    assert np.isnan(close.loc["2022-01-08", "close"])
    assert prices.session.dtype == bool and prices.quote_missing.dtype == bool
    assert prices.close.dtype == np.float64
    assert gold[["news_count_1d", "news_count_3d"]].isna().all().all()
    assert panel.groupby("id").head(1).target.isna().all()
    assert statics.set_index("id").loc["GOLD", "asset_class"] == "metals"
    for _, item in panel.groupby("id"):
        np.testing.assert_array_equal(
            item.timestamp.to_numpy(),
            pd.date_range(item.timestamp.min(), "2025-12-31").to_numpy(),
        )
    btc = panel.loc[panel.id == "BTC"].set_index("timestamp")
    assert np.isnan(btc.loc["2022-01-02", "fng"])
    np.testing.assert_array_equal(
        gold.news_all_count_1d.to_numpy(), btc.news_all_count_1d.to_numpy()
    )


def test_manifest_coverage_and_checksums(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["build_args"]["end"] == "2025-12-31"
    assert len(manifest["repo_sha"]) == 40
    assert len(manifest["sources"]) == 7
    fng = manifest["coverage"]["BTC"]["fng"]
    assert fng["folds"]["2022"]["train"]["finite_count"] == 33
    assert fng["folds"]["2022"]["train"]["days"] == 34
    assert fng["folds"]["2022"]["test"]["start"] == "2022-01-04"
    assert fng["folds"]["2022"]["test"]["end"] == "2022-11-18"
    assert manifest["missing_quote_dates"]["GC=F"] == ["2022-01-17"]
    files = set()
    for line in (bundle / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == digest
        files.add(name)
    assert files == {p.name for p in bundle.iterdir()} - {"SHA256SUMS"}
    for name, digest in manifest["output_sha256"].items():
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == digest


def test_gdelt_kept_only_after_preflight(inputs, tmp_path):
    signals = inputs.signals.copy()
    for name in fetch_signals.GDELT_COLUMNS:
        signals[name] = 1.0
    decision = fetch_signals.gdelt_preflight(signals, "2021-12-01", "2025-12-31")
    builder.write_bundle(
        replace(inputs, signals=signals, gdelt=decision),
        tmp_path,
        start="2021-12-01",
        end="2025-12-31",
    )
    panel = read_fev_parquet(tmp_path / "panel_on.parquet")
    assert all(panel[name].eq(1).all() for name in fetch_signals.GDELT_COLUMNS)
    task = yaml.safe_load((tmp_path / "probe_off.yaml").read_text())["tasks"][0]
    assert set(fetch_signals.GDELT_COLUMNS) <= set(task["past_dynamic_columns"])


def test_crosscheck_vs_reference():
    reference = Path("data/world-signals/raw/reference/closes.csv")
    prices = Path("data/world-signals/built/prices.parquet")
    if not reference.is_file() or not prices.is_file():
        pytest.skip("Optional normalized reference closes/built prices are absent")
    result = builder.crosscheck_reference(pd.read_parquet(prices), pd.read_csv(reference))
    assert all(row["overlap_count"] > 0 for row in result.values())
    assert all(np.isfinite(row["max_relative_difference"]) for row in result.values())


def test_gdelt_single_slower_retry_pass_retains_cache(monkeypatch, tmp_path):
    calls, sleeps = [], []

    def fetch(raw_dir, **kwargs):
        calls.append(kwargs)
        frame = pd.DataFrame({"date": pd.date_range("2024-01-01", "2024-01-31")})
        frame.attrs["errors"] = [{"http_status": 429, "error": "HTTP Error 429"}]
        return frame

    monkeypatch.setattr(fetch_signals, "fetch_gdelt_daily", fetch)
    monkeypatch.setattr(builder.time, "sleep", sleeps.append)
    _, decision = builder.fetch_gdelt_with_retry(tmp_path, "2024-01-01", "2024-01-31", refresh=True)
    assert len(calls) == 2
    assert calls[0]["refresh"] is True and calls[1]["refresh"] is False
    assert calls[1]["interval"] >= 10 and calls[1]["retry_wait"] >= 60
    assert sleeps == [60]
    assert not decision["keep"] and len(decision["errors"]) == 2


def test_gdelt_transport_backoff_and_stops_rate_limited_pass(monkeypatch, tmp_path):
    requests, waits, intervals = [], [], []
    monkeypatch.setattr(sources, "_pace", lambda host, interval: intervals.append(interval))
    monkeypatch.setattr(sources.time, "sleep", waits.append)

    def blocked(request, **kwargs):
        requests.append(request.full_url)
        raise HTTPError(request.full_url, 429, "limited", {}, None)

    frame = fetch_signals.fetch_gdelt_daily(
        tmp_path,
        start="2018-01-01",
        end="2025-12-31",
        opener=blocked,
        interval=10,
        retry_wait=60,
        stop_on_rate_limit=True,
    )
    assert len(requests) == 3 and len(set(requests)) == 1
    assert waits == [60, 60] and intervals == [10, 10, 10]
    assert frame.attrs["stopped_on_rate_limit"]
    assert frame.attrs["errors"][0]["http_status"] == 429
    assert set(fetch_signals.GDELT_COLUMNS) <= set(frame)


def test_provenance_includes_verified_cache_identity(tmp_path):
    import io

    sources.cached_download(
        "https://api.alternative.me/synthetic",
        tmp_path / "fng",
        opener=lambda *a, **kw: io.BytesIO(b"synthetic"),
    )
    records = {r["name"]: r for r in builder.source_provenance(tmp_path)}
    cache = records["fng"]["cache"][0]
    assert cache["sha256"] == hashlib.sha256(b"synthetic").hexdigest()
    assert cache["fetched_at"] and cache["url"].endswith("synthetic")
    assert records["cryptopanic"]["pinned_revision"] == fetch_news.CRYPTOPANIC_REVISION
    assert records["reference_crypto"]["role"] == "reference_only_never_input"
    (tmp_path / cache["cache_path"]).write_bytes(b"modified")
    with pytest.raises(ValueError, match="integrity"):
        builder.source_provenance(tmp_path)


def test_offline_build_needs_no_network(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("network called")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    with pytest.raises(Exception, match="[Oo]ffline.*cache"):
        builder.fetch_inputs(tmp_path, "2018-01-01", "2025-12-31", offline=True)
    frame, decision = builder.fetch_gdelt_with_retry(
        tmp_path / "gdelt", "2018-01-01", "2025-12-31", offline=True
    )
    assert not decision["keep"]
    assert frame[list(fetch_signals.GDELT_COLUMNS)].isna().all().all()
    with pytest.raises(ValueError, match="refresh"):
        builder.fetch_inputs(tmp_path, "2018-01-01", "2025-12-31", offline=True, refresh=True)


def test_offline_gdelt_uses_cached_chunks_and_counts_missing(monkeypatch, tmp_path):
    import io
    from urllib.parse import urlencode

    monkeypatch.setattr(sources, "_pace", lambda *args: None)
    grid = pd.date_range("2025-01-01", "2025-01-10")
    urls = []
    for query in fetch_signals.GDELT_QUERIES:
        for mode, series in (("TimelineVolRaw", "Article Count"), ("TimelineTone", "Average Tone")):
            url = (
                fetch_signals.GDELT_URL
                + "?"
                + urlencode(
                    {
                        "query": query,
                        "mode": mode,
                        "format": "json",
                        "STARTDATETIME": "20250101000000",
                        "ENDDATETIME": "20250110235959",
                    }
                )
            )
            body = json.dumps(
                {
                    "timeline": [
                        {
                            "series": series,
                            "data": [
                                {"date": day.strftime("%Y%m%dT%H%M%SZ"), "value": 2.0}
                                for day in grid
                            ],
                        }
                    ]
                }
            ).encode()
            sources.cached_download(
                url, tmp_path, opener=lambda *args, body=body, **kwargs: io.BytesIO(body)
            )
            urls.append(url)
    frame, decision = builder.fetch_gdelt_with_retry(tmp_path, grid[0], grid[-1], offline=True)
    assert decision["keep"]
    assert frame[list(fetch_signals.GDELT_COLUMNS)].eq(2).all().all()
    key = hashlib.sha256(urls[-1].encode()).hexdigest()
    (tmp_path / f"{key}.json").unlink()
    frame, decision = builder.fetch_gdelt_with_retry(tmp_path, grid[0], grid[-1], offline=True)
    assert not decision["keep"]
    assert frame["gdelt_gold_tone"].isna().all()
    assert "Offline cache miss" in decision["errors"][0]["error"]
