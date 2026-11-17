"""Dataset-independent dimensions and numpy input records for v5 episodes."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from demand_forecast.data.frequency import to_offset
from demand_forecast.pretrain.periods import pretrain_long_lag, pretrain_period

PATCH = 16
MAX_CONTEXT = 2048
MAX_HORIZON = 288
BANK_WINDOW = 8192
MAX_FORECAST_ROWS = 16
MAX_FEATURE_ROWS = 16
MAX_CONTEXT_ROWS = 8
MAX_ATTRIBUTE_TOKENS = 16
MAX_STATIC_NUMBERS = 16
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
KIND_NUMBER, KIND_FLAG, KIND_CATEGORY = 0, 1, 2
ROLE_FORECAST, ROLE_SIBLING, ROLE_KNOWN, ROLE_PAST, ROLE_PEER, ROLE_PANEL = 0, 1, 2, 3, 4, 5
FREQ_CLASSES = (
    "undated",
    "subhourly",
    "hourly",
    "daily",
    "weekly",
    "monthly",
    "quarterly",
    "yearly",
)
STEP_FEATURES = (
    "roll_mean_s",
    "roll_std_s",
    "roll_min_s",
    "roll_max_s",
    "roll_mean_l",
    "roll_std_l",
    "roll_min_l",
    "roll_max_l",
    "diff_1",
    "diff_season",
    "lag_season",
    "lag_long",
    "zero_share",
    "zero_run",
    "since_nonzero",
)
STEP_FAMILIES = {"rolling": slice(0, 8), "lags": slice(8, 12), "zeros": slice(12, 15)}
SUMMARY_FEATURES = (
    "skewness",
    "kurtosis",
    "spread_over_level",
    "zero_share",
    "adi",
    "cv2_nonzero",
    "trend_slope",
    "trend_strength",
    "acf_1",
    "acf_season",
    "acf_long",
    "spec_period",
    "spec_strength",
    "mean_abs_change",
    "cid_ce",
    "beyond_2_sigma",
    "longest_above",
    "longest_below",
    *(f"profile_{i}" for i in range(8)),
    "log_length",
    "missing_share",
)
VALUE_STATS = 6
CH_VALUE, CH_OBSERVED, CH_SUPPORT, CH_STEP, CH_STEP_FLAG, CH_VALUE_STATS, N_CHANNELS = (
    0,
    1,
    2,
    3,
    18,
    33,
    39,
)
N_SUMMARY = 28
N_VALUE_TOKEN = 2 * N_SUMMARY + 1
CALENDAR_FIELDS = ("hour", "dow", "dom", "month", "woy", "quarter")
CALENDAR_SIZES = (24, 7, 31, 12, 53, 4)
N_PHASES = 16


@dataclass(frozen=True)
class Item:
    """One item's arrays, beginning at an absolute integer grid position."""

    item_id: str
    start: int
    targets: dict[str, np.ndarray]
    dynamic: dict[str, np.ndarray]
    static: dict[str, object]


@dataclass(frozen=True)
class Panel:
    """Items on a dated grid, or on an explicitly aligned or unaligned undated grid."""

    name: str
    items: tuple[Item, ...]
    roles: dict[str, str]
    freq: str | None
    epoch: pd.Timestamp | None
    aligned: bool

    def validate(self) -> None:
        """Reject inconsistent grids, array shapes, raw dtypes and dynamic roles."""
        if self.freq is None:
            if self.epoch is not None:
                raise ValueError(f"Panel {self.name!r}: an undated panel must have epoch=None")
        else:
            if self.epoch is None or pd.isna(self.epoch):
                raise ValueError(f"Panel {self.name!r}: freq requires an epoch")
            if not self.aligned:
                raise ValueError(f"Panel {self.name!r}: a dated panel must be aligned")
            try:
                offset = to_offset(self.freq)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Panel {self.name!r}: invalid freq {self.freq!r}") from error
            if offset.n <= 0 or not offset.is_on_offset(self.epoch):
                raise ValueError(f"Panel {self.name!r}: epoch must lie on a positive freq grid")

        for item in self.items:
            length = None
            for is_target, columns in ((True, item.targets), (False, item.dynamic)):
                for name, values in columns.items():
                    context = f"Item {item.item_id!r}, column {name!r}"
                    if not isinstance(values, np.ndarray) or values.ndim != 1:
                        raise ValueError(f"{context}: expected a one-dimensional numpy array")
                    if values.dtype != np.float64 and (is_target or values.dtype != object):
                        expected = "float64" if is_target else "float64 or object"
                        raise ValueError(f"{context}: expected {expected}")
                    if length is None:
                        length = len(values)
                    elif len(values) != length:
                        raise ValueError(f"{context}: unequal array lengths within the item")
                    if not is_target and self.roles.get(name) not in ("known", "past"):
                        raise ValueError(f"{context}: role must be 'known' or 'past'")
        for name, role in self.roles.items():
            if role not in ("known", "past"):
                raise ValueError(f"Panel {self.name!r}, column {name!r}: invalid role {role!r}")

    def timestamps(self, start: int, length: int) -> pd.DatetimeIndex | None:
        """Return absolute grid timestamps, preserving calendar offsets and DST."""
        if self.freq is None:
            return None
        offset = to_offset(self.freq)
        first = self.epoch + start * offset
        return pd.date_range(start=first, periods=length, freq=offset)

    @property
    def season(self) -> int:
        """Use the existing pretraining convention, with one for undated rows."""
        return pretrain_period(self.freq)

    @property
    def long_lag(self) -> int | None:
        """Expose a long-cycle lag only when it exceeds the seasonal lag."""
        lag = pretrain_long_lag(self.freq)
        return lag if lag > self.season else None

    @property
    def freq_class(self) -> int:
        """Return the fixed vocabulary index for the grid's frequency family."""
        if self.freq is None:
            return FREQ_CLASSES.index("undated")
        offset = to_offset(self.freq)
        for classes, name in (
            ((pd.offsets.YearBegin, pd.offsets.YearEnd), "yearly"),
            ((pd.offsets.QuarterBegin, pd.offsets.QuarterEnd), "quarterly"),
            ((pd.offsets.MonthBegin, pd.offsets.MonthEnd), "monthly"),
            ((pd.offsets.Week,), "weekly"),
        ):
            if isinstance(offset, classes):
                return FREQ_CLASSES.index(name)
        if isinstance(offset, pd.offsets.Tick):
            if offset.nanos < pd.Timedelta(hours=1).value:
                return FREQ_CLASSES.index("subhourly")
            if offset.nanos < pd.Timedelta(days=1).value:
                return FREQ_CLASSES.index("hourly")
        return FREQ_CLASSES.index("daily")
