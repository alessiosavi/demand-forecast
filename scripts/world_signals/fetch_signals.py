"""Past-only world-state observations; missing data is never filled."""

import io
import json
from urllib.error import URLError
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from .fetch_prices import CROSS_SYMBOLS, PANEL_SYMBOLS, _day, fetch_panel, session_returns
from .sources import cached_download

FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.dta"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_QUERIES = {"bitcoin": "btc", "ethereum": "eth", "gold": "gold"}
GDELT_COLUMNS = tuple(
    f"gdelt_{asset}_{kind}" for asset in GDELT_QUERIES.values() for kind in ("vol", "tone")
)
GPR_ASSUMPTIONS = [
    "GPR is weekly-published, revised history of day-t articles, not vintage-exact availability.",
]


def _daily(frame):
    if frame.date.isna().any() or not frame.date.eq(frame.date.dt.normalize()).all():
        raise ValueError("Expected midnight daily labels")
    if frame.date.duplicated().any():
        raise ValueError("Duplicate daily labels")
    return frame.sort_values("date").reset_index(drop=True)


def _fng(body):
    payload = json.loads(body)
    if payload.get("metadata", {}).get("error") or "data" not in payload:
        raise ValueError("Invalid F&G response")
    data = pd.DataFrame(payload["data"])
    if not {"timestamp", "value"}.issubset(data.columns):
        raise ValueError("F&G response lacks observations")
    dates = pd.to_datetime(pd.to_numeric(data.timestamp), unit="s", utc=True).dt.tz_localize(None)
    return _daily(pd.DataFrame({"date": dates, "fng": pd.to_numeric(data.value).astype(float)}))


def fetch_fng(raw_dir=None, *, refresh=False, opener=None):
    """F&G values on the provider's 00:00 UTC labels (not a vintage audit)."""
    body = cached_download(FNG_URL, raw_dir, refresh=refresh, opener=opener, validate=_fng)
    result = _fng(body)
    result.attrs["assumptions"] = [
        "F&G uses the provider's 00:00 UTC day labels; publication vintages are not verified."
    ]
    return result


def _gpr(body):
    source = pd.read_stata(io.BytesIO(body))
    required = {"date", "GPRD", "GPRD_ACT", "GPRD_THREAT"}
    if not required.issubset(source):
        raise ValueError(f"GPR missing columns: {sorted(required - set(source))}")
    frame = source[["date", "GPRD", "GPRD_ACT", "GPRD_THREAT"]].rename(
        columns={"GPRD": "gpr", "GPRD_ACT": "gpr_act", "GPRD_THREAT": "gpr_threat"}
    )
    frame["date"] = pd.to_datetime(frame.date, utc=True).dt.tz_localize(None)
    frame[["gpr", "gpr_act", "gpr_threat"]] = frame[["gpr", "gpr_act", "gpr_threat"]].astype(float)
    return _daily(frame)


def fetch_gpr(raw_dir=None, *, refresh=False, opener=None):
    """Official daily .dta; moving averages and event annotations are excluded."""
    body = cached_download(
        GPR_URL, raw_dir, suffix=".dta", refresh=refresh, opener=opener, validate=_gpr
    )
    result = _gpr(body)
    result.attrs["assumptions"] = list(GPR_ASSUMPTIONS)
    return result


def _timeline(body, mode, start, end):
    payload = json.loads(body)
    if not isinstance(payload, dict) or not isinstance(payload.get("timeline"), list):
        raise ValueError("GDELT response lacks timeline")
    if not payload["timeline"]:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], name="date"))
    if not all(isinstance(series, dict) for series in payload["timeline"]):
        raise ValueError("GDELT timeline series must be objects")
    series_name = "Article Count" if mode == "TimelineVolRaw" else "Average Tone"
    matches = [series for series in payload["timeline"] if series.get("series") == series_name]
    if len(matches) != 1:
        raise ValueError(f"GDELT missing unique {series_name} series")
    points = matches[0].get("data")
    if not isinstance(points, list):
        raise ValueError("GDELT missing data array")
    if not points:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], name="date"))
    data = pd.DataFrame(points)
    if not {"date", "value"}.issubset(data):
        raise ValueError("GDELT missing date/value")
    dates = pd.to_datetime(data.date, format="%Y%m%dT%H%M%SZ", utc=True).dt.tz_localize(None)
    frame = _daily(pd.DataFrame({"date": dates, "value": pd.to_numeric(data.value).astype(float)}))
    if not frame.date.between(start, end).all():
        raise ValueError("GDELT dates outside requested UTC interval")
    return frame.set_index("date")["value"]


def fetch_gdelt_daily(
    raw_dir=None,
    queries=("bitcoin", "ethereum", "gold"),
    start="2018-01-01",
    end="2025-12-31",
    *,
    refresh=False,
    opener=None,
    interval=5,
    retry_wait=0,
    stop_on_rate_limit=False,
):
    """Fetch raw counts and average tone; return inclusive daily grid plus errors.

    Requests span at most one year and more than a week for daily resolution.
    A short final chunk is padded backwards, then trimmed to the requested days.
    No TIMELINESMOOTH is requested. HTTP/format failures leave that chunk NaN and
    are recorded in attrs['errors']; preflight decides eligibility for the builder.
    """
    start, end = _day(start), _day(end)
    if end < start:
        raise ValueError("end precedes start")
    if len(set(queries)) != len(queries) or any(q not in GDELT_QUERIES for q in queries):
        raise ValueError("queries must be unique members of bitcoin, ethereum, gold")
    grid = pd.date_range(start, end, freq="D", name="date")
    frame = pd.DataFrame(
        np.nan,
        index=grid,
        columns=[f"gdelt_{GDELT_QUERIES[q]}_{kind}" for q in queries for kind in ("vol", "tone")],
    )
    errors = []
    stopped = False
    for query in queries:
        if stopped:
            break
        for kind, mode in (("vol", "TimelineVolRaw"), ("tone", "TimelineTone")):
            if stopped:
                break
            column = f"gdelt_{GDELT_QUERIES[query]}_{kind}"
            frame[column] = np.nan
            first = start
            while first <= end:
                last = min(first + pd.DateOffset(years=1) - pd.Timedelta(days=1), end)
                request_start = min(first, last - pd.Timedelta(days=8))
                params = {
                    "query": query,
                    "mode": mode,
                    "format": "json",
                    "STARTDATETIME": request_start.strftime("%Y%m%d000000"),
                    "ENDDATETIME": last.strftime("%Y%m%d235959"),
                }
                url = f"{GDELT_URL}?{urlencode(params)}"
                try:
                    body = cached_download(
                        url,
                        raw_dir,
                        refresh=refresh,
                        opener=opener,
                        interval=max(5, interval),
                        retry_wait=retry_wait,
                        validate=lambda b: _timeline(b, mode, request_start, last),
                    )
                    series = _timeline(body, mode, request_start, last)
                    # Pandas aligns by midnight labels; absent bins remain NaN.
                    frame.loc[first:last, column] = series.reindex(frame.loc[first:last].index)
                except (URLError, TimeoutError, ValueError) as error:
                    errors.append(
                        {
                            "url": url,
                            "column": column,
                            "start": str(first.date()),
                            "end": str(last.date()),
                            "error": str(error),
                            "http_status": getattr(error, "code", None),
                        }
                    )
                    if stop_on_rate_limit and getattr(error, "code", None) == 429:
                        stopped = True
                        break
                first = last + pd.Timedelta(days=1)
    result = frame.reset_index()
    result.attrs["errors"] = errors
    result.attrs["stopped_on_rate_limit"] = stopped
    result.attrs["assumptions"] = [
        "GDELT reconstructed raw counts/tone; ingestion-time vintages are not verified."
    ]
    return result


def gdelt_preflight(frame, start="2018-01-01", end="2025-12-31"):
    """Require >=99% finite days in *all six* planned columns, or drop all six."""
    start, end = _day(start), _day(end)
    if end < start:
        raise ValueError("end precedes start")
    data = frame.copy()
    data["date"] = pd.to_datetime(data.date)
    data = _daily(data).set_index("date").reindex(pd.date_range(start, end, freq="D"))
    ratios = {
        column: float(np.isfinite(data[column].to_numpy(dtype=float)).mean())
        if column in data
        else 0.0
        for column in GDELT_COLUMNS
    }
    return {
        "keep": bool(all(ratio >= 0.99 for ratio in ratios.values())),
        "complete_ratio": ratios,
        "threshold": 0.99,
        "days": len(data),
        "start": str(start.date()),
        "end": str(end.date()),
    }


def cross_series(start, end, raw_dir=None, *, refresh=False, opener=None):
    """Session-date levels and log returns (including self cross-returns)."""
    symbols = {**CROSS_SYMBOLS, **{key: PANEL_SYMBOLS[key] for key in ("BTC", "ETH", "GOLD")}}
    prices = fetch_panel(symbols, start, end, raw_dir=raw_dir, refresh=refresh, opener=opener)
    returns = session_returns(prices).pivot(index="date", columns="id", values="ret")
    levels = prices.pivot(index="date", columns="id", values="close")
    result = pd.DataFrame(index=levels.index)
    for key in ("VIX", "DXY"):
        result[key.lower()] = levels[key] if key in levels else np.nan
    for key in ("SPX", "WTI", "BTC", "ETH", "GOLD"):
        result[f"{key.lower()}_ret"] = returns[key] if key in returns else np.nan
    return result.reset_index()
