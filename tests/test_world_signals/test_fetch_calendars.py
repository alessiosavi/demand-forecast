import hashlib
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from scripts.world_signals import fetch_calendars as calendars

FIXTURE = Path(__file__).parent / "fixtures/fomc_min.csv"


def zip_opener(csv):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("fomc-meetings.csv", csv)
    body = buffer.getvalue()

    def opener(request, **kwargs):
        assert request.full_url == calendars.FOMC_URL
        return io.BytesIO(body)

    return opener, body


def test_observed_holiday_shift_including_next_year():
    flags = calendars.holiday_flags(
        pd.to_datetime(["2021-07-04", "2021-07-05", "2021-12-31", "2022-01-01"])
    )
    assert flags.holiday_us.tolist() == [0, 1, 1, 0]
    holidays = calendars.us_holidays([2021])
    assert pd.Timestamp("2021-12-31") in holidays
    assert pd.Timestamp("2021-06-18") in holidays
    assert pd.Timestamp("2020-06-19") not in calendars.us_holidays([2020])


def test_good_friday_computus():
    assert list(calendars.good_friday([2023, 2024, 2025])) == list(
        pd.to_datetime(["2023-04-07", "2024-03-29", "2025-04-18"])
    )
    frame = calendars.holiday_flags(pd.date_range("2024-03-28", "2024-03-31"))
    assert frame.holiday_gf.tolist() == [0, 1, 0, 0]
    assert frame.holiday_us.tolist() == [0, 0, 0, 0]


def test_scheduled_fomc_end_date_no_unscheduled(tmp_path):
    opener, body = zip_opener(FIXTURE.read_bytes())
    dates = pd.date_range("2020-03-01", "2020-04-30")
    frame = calendars.fomc_flags(dates, tmp_path, opener=opener)
    assert frame.loc[frame.fomc_day == 1, "date"].tolist() == list(
        pd.to_datetime(["2020-03-18", "2020-04-29"])
    )
    assert set(frame.fomc_day) == {0, 1}
    metadata = json.loads((tmp_path / "fomc_manifest.json").read_text())
    assert metadata["url"] == calendars.FOMC_URL
    assert metadata["sha256"] == hashlib.sha256(body).hexdigest()
    assert metadata["csv_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    cached = calendars.fomc_flags(dates, tmp_path)
    pd.testing.assert_frame_equal(frame, cached)


def test_flags_are_pure_functions_of_schedule(tmp_path):
    source = pd.read_csv(FIXTURE)
    dates = pd.date_range("2020-03-01", "2020-04-30")
    original, _ = zip_opener(source.to_csv(index=False))
    before = calendars.fomc_flags(dates, tmp_path, opener=original)
    source["cancelled"] = ~source.cancelled
    source["status"] = "unknown future outcome"
    source["statementDate"] = "2099-12-31"
    source["statementTimeET"] = "23:59"
    changed, _ = zip_opener(source.to_csv(index=False))
    after = calendars.fomc_flags(dates, tmp_path, refresh=True, opener=changed)
    pd.testing.assert_frame_equal(before, after)


def test_fomc_unknown_unscheduled_flag_fails_closed(tmp_path):
    data = pd.read_csv(FIXTURE).astype({"unscheduled": str})
    data.loc[0, "unscheduled"] = "unknown"
    opener, _ = zip_opener(data.to_csv(index=False))
    with pytest.raises(ValueError, match="unscheduled"):
        calendars.fomc_flags(pd.date_range("2020-03-01", "2020-03-31"), tmp_path, opener=opener)


def test_empty_holiday_grid():
    assert calendars.holiday_flags([]).empty
    assert len(calendars.us_holidays([])) == 0
    assert len(calendars.good_friday([])) == 0
