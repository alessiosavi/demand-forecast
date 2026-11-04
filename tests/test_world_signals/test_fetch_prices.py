import io
import json
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest

from scripts.world_signals import fetch_prices as prices
from scripts.world_signals import sources


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    monkeypatch.setattr(prices, "_utc_now", lambda: pd.Timestamp("2024-03-12T12:00Z"))
    monkeypatch.setattr(sources, "_pace", lambda *args: None)


def test_crypto_dates_utc_nan_and_unfinished_dropped(tmp_path, fixture_opener):
    frame = prices.fetch_yahoo_daily(
        "BTC-USD",
        "2024-03-08",
        "2024-03-12",
        raw_dir=tmp_path,
        opener=fixture_opener("yahoo_chart_crypto_min.json"),
    )
    assert frame.date.tolist() == list(pd.to_datetime(["2024-03-08", "2024-03-09", "2024-03-11"]))
    assert frame.close.tolist() == [10, 12, 18]
    assert frame.date.dt.tz is None
    assert frame.attrs["missing_quote_dates"] == ["2024-03-10"]


def test_exchange_dates_use_new_york_across_dst(tmp_path, fixture_opener):
    frame = prices.fetch_yahoo_daily(
        "GC=F",
        "2024-03-08",
        "2024-03-12",
        raw_dir=tmp_path,
        opener=fixture_opener("yahoo_chart_gc_min.json"),
    )
    assert frame.date.tolist() == list(pd.to_datetime(["2024-03-08", "2024-03-11"]))
    # 03:00Z March 12 is March 11 in New York, not March 12.
    assert frame.attrs["missing_quote_dates"] == ["2024-03-11"]
    assert frame.close.tolist() == [20, 30]


def test_crypto_current_day_dropped(tmp_path, json_opener):
    payload = {
        "chart": {
            "error": None,
            "result": [
                {
                    "timestamp": [1710201600],
                    "indicators": {"quote": [{"close": [100], "volume": [1]}]},
                }
            ],
        }
    }
    frame = prices.fetch_yahoo_daily(
        "BTC-USD", "2024-03-12", "2024-03-12", raw_dir=tmp_path, opener=json_opener(payload)
    )
    assert frame.empty


def test_session_returns_weekend_nonpositive_and_ids():
    frame = pd.DataFrame(
        {
            "id": ["GOLD", "WTI", "GOLD", "WTI", "WTI", "WTI"],
            "date": pd.to_datetime(
                ["2024-03-11", "2020-04-17", "2024-03-08", "2020-04-20", "2020-04-21", "2020-04-22"]
            ),
            "close": [30, 10, 20, -2, 8, 16],
        }
    )
    result = prices.session_returns(frame).set_index(["id", "date"])
    assert result.loc[("GOLD", "2024-03-11"), "ret"] == pytest.approx(np.log(1.5))
    assert result.loc[("WTI", "2020-04-22"), "ret"] == pytest.approx(np.log(2))
    assert result.loc["WTI", "ret"].iloc[:3].isna().all()
    assert np.isnan(result.loc[("GOLD", "2024-03-08"), "ret"])
    zero = frame.copy()
    zero.loc[3, "close"] = 0
    assert prices.session_returns(zero).query("id == 'WTI'").ret.iloc[:3].isna().all()


def test_request_cache_metadata_and_bounded_retry(tmp_path, fixture_opener, monkeypatch):
    calls = []
    monkeypatch.setattr(sources.time, "sleep", lambda delay: None)
    good = fixture_opener("yahoo_chart_crypto_min.json")

    def opener(req, **kwargs):
        calls.append(req)
        if len(calls) < 3:
            raise HTTPError(req.full_url, 429, "rate limited", {}, None)
        return good(req)

    first = prices.fetch_yahoo_daily(
        "BTC-USD", "2024-03-08", "2024-03-11", raw_dir=tmp_path, opener=opener
    )
    second = prices.fetch_yahoo_daily("BTC-USD", "2024-03-08", "2024-03-11", raw_dir=tmp_path)
    pd.testing.assert_frame_equal(first, second)
    assert len(calls) == 3
    assert "Mozilla" in calls[0].get_header("User-agent")
    assert parse_qs(urlparse(calls[0].full_url).query)["interval"] == ["1d"]
    metadata = json.loads(next(tmp_path.glob("*.meta.json")).read_text())
    assert metadata["url"] == calls[0].full_url
    assert len(metadata["sha256"]) == 64
    assert metadata["fetched_at"]
    calls.clear()

    def failing(req, **kwargs):
        calls.append(req)
        raise HTTPError(req.full_url, 429, "rate limited", {}, None)

    with pytest.raises(HTTPError):
        prices.fetch_yahoo_daily(
            "ETH-USD", "2024-03-08", "2024-03-11", raw_dir=tmp_path, opener=failing
        )
    assert len(calls) == 3


def test_invalid_response_not_cached(tmp_path):
    with pytest.raises(ValueError):
        prices.fetch_yahoo_daily(
            "BTC-USD",
            "2024-03-08",
            "2024-03-11",
            raw_dir=tmp_path,
            opener=lambda *a, **k: io.BytesIO(b'{"chart":{"error":{"code":"bad"}}}'),
        )
    assert not list(tmp_path.glob("*.json"))


def test_panel_passes_injected_opener(tmp_path, fixture_opener):
    frame = prices.fetch_panel(
        {"BTC": "BTC-USD", "ETH": "ETH-USD"},
        "2024-03-08",
        "2024-03-11",
        raw_dir=tmp_path,
        opener=fixture_opener("yahoo_chart_crypto_min.json"),
    )
    assert frame.groupby("id").size().to_dict() == {"BTC": 3, "ETH": 3}
    assert len(prices.PANEL_SYMBOLS) == 10
    assert prices.CROSS_SYMBOLS == {"VIX": "^VIX", "DXY": "DX-Y.NYB", "SPX": "^GSPC", "WTI": "CL=F"}
