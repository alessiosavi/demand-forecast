"""Deterministic holidays and scheduled-only FOMC decision dates."""

import hashlib
import io
import json
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from .sources import DEFAULT_RAW_DIR, _atomic_write, cached_download

FOMC_URL = "https://www.kaggle.com/api/v1/datasets/download/pushbuildteam/fomc-meeting-calendar"
FOMC_ASSUMPTIONS = [
    "Rows marked unscheduled=False are treated as the advance schedule; historical schedule vintages are not independently verified.",
    "Use endDate, ignoring cancelled, status, statementDate and statementTimeET (outcomes/hindsight).",
]


def us_holidays(years):
    """Observed federal holidays whose observed date is in the requested years."""
    years = sorted(set(years))
    if not years:
        return pd.DatetimeIndex([])
    holidays = USFederalHolidayCalendar().holidays(
        start=f"{years[0]}-01-01", end=f"{years[-1]}-12-31"
    )
    return holidays[holidays.year.isin(years)]


def good_friday(years):
    """Gregorian computus (Meeus/Jones/Butcher), Easter Sunday minus two days."""
    days = []
    for year in sorted(set(years)):
        a = year % 19
        b, c = divmod(year, 100)
        d, e = divmod(b, 4)
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i, k = divmod(c, 4)
        l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741 -- conventional computus name
        m = (a + 11 * h + 22 * l) // 451
        month, remainder = divmod(h + l - 7 * m + 114, 31)
        days.append(date(year, month, remainder + 1) - timedelta(days=2))
    return pd.DatetimeIndex(days)


def _dates(dates):
    return pd.DatetimeIndex(pd.to_datetime(dates, utc=True)).tz_localize(None).normalize()


def holiday_flags(dates):
    dates = _dates(dates)
    years = dates.year.tolist()
    return pd.DataFrame(
        {
            "date": dates,
            "holiday_us": dates.isin(us_holidays(years)).astype(int),
            "holiday_gf": dates.isin(good_friday(years)).astype(int),
        }
    )


def _fomc(body):
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        names = [name for name in archive.namelist() if Path(name).name == "fomc-meetings.csv"]
        if len(names) != 1:
            raise ValueError("FOMC archive must contain exactly one fomc-meetings.csv")
        csv = archive.read(names[0])
    source = pd.read_csv(io.BytesIO(csv), dtype={"unscheduled": str})
    if not {"unscheduled", "endDate"}.issubset(source):
        raise ValueError("FOMC missing unscheduled/endDate")
    flags = source.unscheduled.str.strip().str.lower()
    if not flags.isin(["true", "false"]).all():
        raise ValueError("FOMC unscheduled must be true or false")
    scheduled = source.loc[flags == "false", "endDate"]
    dates = pd.to_datetime(scheduled, format="ISO8601", utc=True)
    if dates.isna().any():
        raise ValueError("FOMC scheduled endDate is missing")
    dates = pd.DatetimeIndex(dates).tz_localize(None).normalize().unique().sort_values()
    return dates, csv


def fomc_flags(dates, raw_dir=None, *, refresh=False, opener=None):
    """Flag scheduled endDate only, 0/1 over the entire supplied grid.

    The public Kaggle endpoint returns a ZIP, with no authentication needed.
    Cache metadata records the fetched ZIP's hash and date; this manifest also
    records the embedded CSV hash. Cancellation/status fields are never used.
    """
    body = cached_download(
        FOMC_URL, raw_dir, suffix=".zip", refresh=refresh, opener=opener, validate=_fomc
    )
    scheduled, csv = _fomc(body)
    metadata = {
        "url": FOMC_URL,
        "license": "CC0-1.0",
        "sha256": hashlib.sha256(body).hexdigest(),
        "csv_sha256": hashlib.sha256(csv).hexdigest(),
        "assumptions": FOMC_ASSUMPTIONS,
    }
    directory = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    _atomic_write(
        directory / "fomc_manifest.json", (json.dumps(metadata, indent=2) + "\n").encode()
    )
    dates = _dates(dates)
    return pd.DataFrame({"date": dates, "fomc_day": dates.isin(scheduled).astype(int)})
