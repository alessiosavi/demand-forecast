import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.world_signals import fetch_news as news

FIXTURE = Path(__file__).parent / "fixtures/cryptopanic_min.csv"


def articles():
    return news.normalize_articles([pd.read_csv(FIXTURE)])


def test_day_boundary_and_multi_coin():
    data = articles()
    assert len(data) == 5
    assert str(data.timestamp_utc.dt.tz) == "UTC"
    cov = news.coverage(data, ["BTC", "ETH"])
    counts = news.aggregate_daily(data, ["BTC", "ETH"], cov).set_index(["id", "date"])
    assert counts.loc[("BTC", "2024-01-01"), "news_count_1d"] == 1
    assert counts.loc[("ETH", "2024-01-01"), "news_count_1d"] == 1
    assert counts.loc[("BTC", "2024-01-02"), "news_count_1d"] == 1
    assert counts.loc[("ALL", "2024-01-01"), "news_count_1d"] == 1
    assert counts.loc[("BTC", "2024-01-03"), "news_count_3d"] == 2
    assert counts.loc[("BTC", "2024-01-04"), "news_count_3d"] == 2
    assert pd.isna(counts.loc[("BTC", "2024-01-02"), "news_count_3d"])


def test_zero_inside_nan_outside_coverage():
    data = articles()
    cov = news.coverage(data, ["BTC", "ETH", "GOLD"])
    assert set(cov.id) == {"BTC", "ETH", "GOLD", "ALL"}
    assert cov.set_index("id").loc["GOLD"].isna().all()
    frame = news.aggregate_daily(data, ["BTC", "ETH", "GOLD"], cov).set_index(["id", "date"])
    assert frame.loc[("BTC", "2024-01-03"), "news_count_1d"] == 0
    assert pd.isna(frame.loc[("BTC", "2023-12-31"), "news_count_1d"])
    assert pd.isna(frame.loc[("BTC", "2024-01-05"), "news_count_1d"])
    assert frame.loc["GOLD"].isna().all().all()
    assert "coverage" in data.attrs["assumptions"][1].lower()


def test_votes_not_exposed():
    frame = pd.read_csv(FIXTURE)
    clean = news.normalize_articles([frame])
    assert list(clean) == ["article_id", "timestamp_utc", "coins"]
    frame[["positive", "negative", "important"]] = 100000
    pd.testing.assert_frame_equal(clean, news.normalize_articles([frame]))
    assert "UTC" in clean.attrs["assumptions"][0]


def test_duplicate_ids_union_tags_and_reject_conflicting_dates():
    frame = pd.read_csv(FIXTURE).iloc[[0]].copy()
    extra = frame.assign(currencies="SOL")
    assert news.normalize_articles([frame, extra]).coins.iloc[0] == ["BTC", "ETH", "SOL"]
    with pytest.raises(ValueError, match="timestamp"):
        news.normalize_articles([frame, extra.assign(newsDatetime="2024-01-02 12:00:00")])


def test_download_pinned_rar_bsdtar_and_recorded_assumption(tmp_path, monkeypatch):
    requests, commands = [], []
    body = b"Rar!synthetic-archive"

    def opener(request, **kwargs):
        requests.append(request.full_url)
        return io.BytesIO(body)

    def run(command, **kwargs):
        commands.append(command)
        if "-tf" in command:
            return SimpleNamespace(stdout="news_currencies_source_joinedResult.csv\n")
        out = Path(command[command.index("-C") + 1])
        (out / "news_currencies_source_joinedResult.csv").write_bytes(FIXTURE.read_bytes())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(news.subprocess, "run", run)
    paths = news.download_archive(tmp_path, revision=news.CRYPTOPANIC_REVISION, opener=opener)
    assert len(paths) == 1 and paths[0].exists()
    assert f"/{news.CRYPTOPANIC_REVISION}/" in requests[0]
    assert any(command[:2] == ["bsdtar", "-xf"] for command in commands)
    meta = json.loads((tmp_path / news.CRYPTOPANIC_REVISION / "archive_manifest.json").read_text())
    assert meta["revision"] == news.CRYPTOPANIC_REVISION
    assert meta["sha256"] == hashlib.sha256(body).hexdigest()
    assert "UTC" in meta["assumptions"][0]
    news.download_archive(tmp_path, revision=news.CRYPTOPANIC_REVISION, opener=opener)
    assert len(requests) == 1
    with pytest.raises(ValueError, match="commit SHA"):
        news.download_archive(tmp_path, revision="main", opener=opener)


def test_empty_news_keeps_schema():
    empty = news.normalize_articles([])
    assert str(empty.timestamp_utc.dtype) == "datetime64[ns, UTC]"
    cov = news.coverage(empty, ["BTC"])
    assert news.aggregate_daily(empty, ["BTC"], cov).empty
