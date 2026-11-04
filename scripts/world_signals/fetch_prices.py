"""Yahoo daily candles, dated by their sessions; no filling or adjustment."""

import json
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd

from .sources import cached_download

PANEL_SYMBOLS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "LTC": "LTC-USD",
    "XRP": "XRP-USD",
    "BNB": "BNB-USD",
    "DOGE": "DOGE-USD",
    "ADA": "ADA-USD",
    "SOL": "SOL-USD",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
}
CROSS_SYMBOLS = {"VIX": "^VIX", "DXY": "DX-Y.NYB", "SPX": "^GSPC", "WTI": "CL=F"}
EXCHANGE_SYMBOLS = {"GC=F", "SI=F", *CROSS_SYMBOLS.values()}


def _utc_now():
    return pd.Timestamp.now(tz="UTC")


def _day(value):
    """Interpret a date argument as a UTC calendar label, returned tz-naive."""
    return pd.Timestamp(value).tz_localize(None).normalize()


def _chart(body):
    payload = json.loads(body)
    chart = payload.get("chart", {})
    if chart.get("error") or not chart.get("result"):
        raise ValueError(f"Invalid Yahoo chart response: {chart.get('error')}")
    result = chart["result"][0]
    try:
        timestamps = result["timestamp"]
        quotes = result["indicators"]["quote"][0]
        if len(timestamps) != len(quotes["close"]):
            raise ValueError("Yahoo timestamp/close lengths differ")
        if "volume" in quotes and len(timestamps) != len(quotes["volume"]):
            raise ValueError("Yahoo timestamp/volume lengths differ")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("Yahoo response lacks daily candles") from error
    return result


def fetch_yahoo_daily(symbol, start, end, *, raw_dir=None, refresh=False, opener=None):
    """Return completed candles on the inclusive [start, end] session-date range.

    Exchange dates use America/New_York, crypto uses UTC. Today's calendar label
    in that zone is conservatively excluded even if trading has already closed.
    Missing quotes are dropped but their dates remain in ``attrs`` for the future
    price sidecar. Raw timestamps/quotes remain in the private JSON cache.
    """
    start, end = _day(start), _day(end)
    if end < start:
        raise ValueError("end precedes start")
    timezone = "America/New_York" if symbol in EXCHANGE_SYMBOLS else "UTC"
    params = {
        "period1": int(start.tz_localize("UTC").timestamp()),
        "period2": int((end + pd.Timedelta(days=1)).tz_localize("UTC").timestamp()),
        "interval": "1d",
    }
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}?{urlencode(params)}"
    body = cached_download(
        url, raw_dir, refresh=refresh, opener=opener, interval=1, validate=_chart
    )
    result = _chart(body)
    stamps = pd.to_datetime(result["timestamp"], unit="s", utc=True)
    dates = stamps.tz_convert(timezone).tz_localize(None).normalize()
    quotes = result["indicators"]["quote"][0]
    frame = pd.DataFrame(
        {
            "date": dates,
            "close": quotes["close"],
            "volume": quotes.get("volume", [None] * len(dates)),
        }
    )
    frame[["close", "volume"]] = frame[["close", "volume"]].astype(float)
    today = _utc_now().tz_convert(timezone).tz_localize(None).normalize()
    frame = frame.loc[frame.date.between(start, end) & (frame.date < today)]
    missing = frame.loc[frame.close.isna(), "date"].dt.strftime("%Y-%m-%d").tolist()
    frame = frame.dropna(subset=["close"]).sort_values("date")
    if frame.date.duplicated().any():
        raise ValueError("Multiple Yahoo closes for one session date")
    frame = frame.reset_index(drop=True)
    frame.attrs["missing_quote_dates"] = missing
    return frame


def fetch_panel(symbols, start, end, *, raw_dir=None, refresh=False, opener=None):
    """Fetch an id→Yahoo-symbol mapping, sequentially with shared pacing."""
    frames, missing = [], {}
    for asset, symbol in symbols.items():
        frame = fetch_yahoo_daily(
            symbol, start, end, raw_dir=raw_dir, refresh=refresh, opener=opener
        )
        missing[asset] = frame.attrs["missing_quote_dates"]
        frame.attrs = {}
        frames.append(frame.assign(id=asset))
    result = (
        pd.concat(frames, ignore_index=True)[["id", "date", "close", "volume"]]
        if frames
        else pd.DataFrame(columns=["id", "date", "close", "volume"])
    )
    result.attrs["missing_quote_dates"] = missing
    return result


def session_returns(frame):
    """Log changes within each id's ordered sessions, before daily reindexing.

    A non-positive or missing price invalidates both adjacent returns. No price
    is filled, and no calendar days are inserted here.
    """
    result = frame[["id", "date", "close"]].sort_values(["id", "date"]).copy()
    if result.duplicated(["id", "date"]).any():
        raise ValueError("Duplicate id/session")
    positive = result["close"].where(result["close"] > 0)
    result["ret"] = np.log(positive).groupby(result["id"]).diff()
    return result[["id", "date", "ret"]].reset_index(drop=True)
