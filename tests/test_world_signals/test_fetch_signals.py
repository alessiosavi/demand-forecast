import io
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest

from scripts.world_signals import fetch_signals as signals
from scripts.world_signals import sources

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(sources, "_pace", lambda *a: None)
    monkeypatch.setattr(sources.time, "sleep", lambda *a: None)


def test_fng_parse_and_no_fill(tmp_path, fixture_opener):
    frame = signals.fetch_fng(tmp_path, opener=fixture_opener("fng_min.json"))
    assert list(frame) == ["date", "fng"]
    expanded = frame.set_index("date").reindex(pd.date_range("2024-01-01", "2024-01-03"))
    assert expanded.fng.iloc[[0, 2]].tolist() == [20, 42]
    assert pd.isna(expanded.fng.iloc[1])
    assert frame.date.dt.tz is None


def test_gpr_columns(tmp_path):
    source = pd.read_csv(FIXTURES / "gpr_min.csv", parse_dates=["date"])
    buffer = io.BytesIO()
    source.to_stata(buffer, write_index=False)
    data = buffer.getvalue()
    frame = signals.fetch_gpr(tmp_path, opener=lambda *a, **kw: io.BytesIO(data))
    assert list(frame) == ["date", "gpr", "gpr_act", "gpr_threat"]
    assert frame.gpr.tolist() == [12, 15]
    assert pd.isna(frame.gpr_act.iloc[1])
    assert "revised" in frame.attrs["assumptions"][0]
    assert len(list(tmp_path.glob("*.dta.meta.json"))) == 1


def timeline_opener(request, **kwargs):
    params = parse_qs(urlparse(request.full_url).query)
    mode = params["mode"][0]
    data = json.loads((FIXTURES / "gdelt_timeline_min.json").read_text())
    if mode == "TimelineTone":
        data["timeline"][0]["series"] = "Average Tone"
        data["timeline"][0]["data"][0]["value"] = -2.5
        data["timeline"][0]["data"][1]["value"] = None
    return io.BytesIO(json.dumps(data).encode())


def test_gdelt_fixture_shape_and_preflight(tmp_path):
    requests = []

    def opener(request, **kwargs):
        requests.append(parse_qs(urlparse(request.full_url).query))
        return timeline_opener(request, **kwargs)

    frame = signals.fetch_gdelt_daily(tmp_path, start="2024-01-01", end="2024-01-31", opener=opener)
    assert list(frame) == [
        "date",
        "gdelt_btc_vol",
        "gdelt_btc_tone",
        "gdelt_eth_vol",
        "gdelt_eth_tone",
        "gdelt_gold_vol",
        "gdelt_gold_tone",
    ]
    assert len(frame) == 31
    assert frame.gdelt_btc_vol.iloc[0] == 2  # Raw count, not value/norm.
    assert frame.gdelt_btc_tone.iloc[0] == -2.5
    assert pd.isna(frame.gdelt_btc_vol.iloc[1])
    assert frame.gdelt_btc_vol.iloc[2] == 0  # Observed zero stays zero.
    assert all("TIMELINESMOOTH" not in {k.upper() for k in p} for p in requests)
    assert all(p["STARTDATETIME"] == ["20240101000000"] for p in requests)
    assert all(p["ENDDATETIME"] == ["20240131235959"] for p in requests)
    result = signals.gdelt_preflight(frame, "2024-01-01", "2024-01-31")
    assert result["keep"] is False
    assert result["complete_ratio"]["gdelt_btc_vol"] == pytest.approx(2 / 31)
    assert not signals.gdelt_preflight(frame)["keep"]  # Default fixed historical interval.


def test_preflight_requires_all_columns_99_percent_finite_days():
    grid = pd.date_range("2018-01-01", "2025-12-31")
    frame = pd.DataFrame(
        {"date": grid, **{column: np.ones(len(grid)) for column in signals.GDELT_COLUMNS}}
    )
    frame.loc[:28, "gdelt_btc_vol"] = np.nan
    assert signals.gdelt_preflight(frame)["keep"] is True
    frame.loc[29, "gdelt_btc_vol"] = np.inf
    assert signals.gdelt_preflight(frame)["keep"] is False
    assert signals.gdelt_preflight(frame.drop(columns="gdelt_eth_vol"))["keep"] is False
    assert signals.gdelt_preflight(pd.DataFrame(columns=["date"]))["keep"] is False
    with pytest.raises(ValueError, match="Duplicate"):
        signals.gdelt_preflight(pd.concat([frame, frame.iloc[[0]]]))


def test_gdelt_chunks_daily_resolution_cache_and_failures(tmp_path):
    requests = []

    def empty(request, **kwargs):
        p = parse_qs(urlparse(request.full_url).query)
        requests.append(p)
        first = pd.to_datetime(p["STARTDATETIME"][0], format="%Y%m%d%H%M%S")
        last = pd.to_datetime(p["ENDDATETIME"][0], format="%Y%m%d%H%M%S")
        assert pd.Timedelta(days=7) < last - first <= pd.Timedelta(days=366)
        return io.BytesIO(b'{"timeline": []}')

    frame = signals.fetch_gdelt_daily(
        tmp_path, queries=("bitcoin",), start="2020-01-01", end="2022-01-03", opener=empty
    )
    assert len(requests) == 6  # 3 chunks x 2 modes, short tail padded to daily resolution.
    assert frame.drop(columns="date").isna().all().all()
    signals.fetch_gdelt_daily(
        tmp_path, queries=("bitcoin",), start="2020-01-01", end="2022-01-03", opener=empty
    )
    assert len(requests) == 6

    def blocked(request, **kwargs):
        raise HTTPError(request.full_url, 429, "limited", {}, None)

    failed = signals.fetch_gdelt_daily(
        tmp_path / "blocked",
        queries=("bitcoin",),
        start="2018-01-01",
        end="2018-01-31",
        opener=blocked,
    )
    assert failed.drop(columns="date").isna().all().all()
    assert len(failed.attrs["errors"]) == 2
    assert all("429" in error["error"] for error in failed.attrs["errors"])


def test_gdelt_intraday_not_silently_counted_as_daily(tmp_path, json_opener):
    payload = {
        "timeline": [
            {"series": "Article Count", "data": [{"date": "20240101T120000Z", "value": 2}]}
        ]
    }
    frame = signals.fetch_gdelt_daily(
        tmp_path,
        queries=("bitcoin",),
        start="2024-01-01",
        end="2024-01-31",
        opener=json_opener(payload),
    )
    assert frame.gdelt_btc_vol.isna().all()
    assert "daily" in frame.attrs["errors"][0]["error"]


def test_cross_series_wti_nonpositive_is_nan(tmp_path, monkeypatch):
    requests = []
    marker = object()

    def panel(symbols, start, end, **kwargs):
        requests.append((symbols, kwargs))
        return pd.concat(
            [
                pd.DataFrame(
                    {
                        "id": key,
                        "date": pd.to_datetime(
                            ["2020-04-17", "2020-04-20", "2020-04-21", "2020-04-22"]
                        ),
                        "close": [10, -2, 8, 16] if key == "WTI" else [10, 20, 40, 80],
                        "volume": 1,
                    }
                )
                for key in symbols
            ],
            ignore_index=True,
        )

    monkeypatch.setattr(signals, "fetch_panel", panel)
    frame = signals.cross_series("2020-04-17", "2020-04-22", tmp_path, opener=marker).set_index(
        "date"
    )
    assert list(frame) == ["vix", "dxy", "spx_ret", "wti_ret", "btc_ret", "eth_ret", "gold_ret"]
    assert frame.wti_ret.iloc[:3].isna().all()
    assert frame.wti_ret.iloc[3] == pytest.approx(np.log(2))
    assert frame.vix.tolist() == [10, 20, 40, 80]
    assert frame.gold_ret.iloc[1] == pytest.approx(np.log(2))
    assert requests[0][1]["opener"] is marker
    assert pd.Timestamp("2020-04-18") not in frame.index


@pytest.mark.parametrize("payload", [[], {"timeline": [None]}])
def test_gdelt_malformed_json_shape_is_recorded_as_missing(tmp_path, json_opener, payload):
    frame = signals.fetch_gdelt_daily(
        tmp_path,
        queries=("bitcoin",),
        start="2024-01-01",
        end="2024-01-31",
        opener=json_opener(payload),
    )
    assert frame.drop(columns="date").isna().all().all()
    assert len(frame.attrs["errors"]) == 2
