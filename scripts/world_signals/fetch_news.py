"""Publication-time article counts from the pinned CC0 CryptoPanic archive."""

import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from .sources import _atomic_write, cached_download

# Resolved using GET /repos/soheilrahsaz/cryptoNewsDataset/commits/main on 2026-11-04.
CRYPTOPANIC_REVISION = "a3cac04997702dd5a05c312647e2038b3629caff"
NEWS_ASSUMPTIONS = [
    "Offset-free newsDatetime values are treated as UTC publication timestamps; "
    "historical timezone correctness is not independently verified.",
    "Coverage is inferred from first/last publication per coin (and ALL); "
    "zero inside that interval assumes archive completeness, not a verified ingestion audit.",
]
ARCHIVE_NAME = "news_currencies_source_joinedResult"


def download_archive(dest, *, revision, refresh=False, opener=None, raw_dir=None):
    """Download only the joined RAR (contains every field needed for counts).

    Cache and extracted CSVs are isolated by full commit SHA. ``raw_dir`` can
    override the cache location; extraction and the manifest remain under dest.
    No moving branch reference is accepted.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full lowercase commit SHA")
    directory = Path(dest) / revision
    directory.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://raw.githubusercontent.com/soheilrahsaz/cryptoNewsDataset/{revision}"
        f"/csvOutput/{ARCHIVE_NAME}.rar"
    )
    body = cached_download(
        url,
        raw_dir if raw_dir is not None else directory,
        suffix=".rar",
        refresh=refresh,
        opener=opener,
    )
    archive = directory / f"{ARCHIVE_NAME}.rar"
    _atomic_write(archive, body)
    members = subprocess.run(
        ["bsdtar", "-tf", str(archive)], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    # The pinned archive has one flat CSV; do not extract unexpected paths.
    if members != [f"{ARCHIVE_NAME}.csv"]:
        raise ValueError(f"Unexpected archive members: {members}")
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        subprocess.run(
            ["bsdtar", "-xf", str(archive), "-C", temporary],
            check=True,
            capture_output=True,
            text=True,
        )
        csv = Path(temporary) / members[0]
        if csv.is_symlink() or not csv.is_file():
            raise ValueError("Expected a regular CSV in archive")
        csv.replace(directory / members[0])
    metadata = {
        "revision": revision,
        "url": url,
        "license": "CC0-1.0",
        "sha256": hashlib.sha256(body).hexdigest(),
        "assumptions": NEWS_ASSUMPTIONS,
        "csv_sha256": hashlib.sha256((directory / members[0]).read_bytes()).hexdigest(),
    }
    _atomic_write(
        directory / "archive_manifest.json", (json.dumps(metadata, indent=2) + "\n").encode()
    )
    return [directory / name for name in members]


def _coins(value):
    if pd.isna(value) or str(value).strip().upper() in {"", "NULL"}:
        return []
    return sorted({coin.strip().upper() for coin in str(value).split(",") if coin.strip()})


def normalize_articles(frames):
    """Discard vote/text fields; merge duplicate IDs' tags, reject date conflicts."""
    parts = []
    for frame in frames:
        if frame["id"].isna().any() or frame["newsDatetime"].isna().any():
            raise ValueError("Article id and timestamp are required")
        parts.append(
            pd.DataFrame(
                {
                    "article_id": frame["id"].astype(str),
                    "timestamp_utc": pd.to_datetime(
                        frame["newsDatetime"], format="mixed", utc=True
                    ),
                    "coins": frame["currencies"].map(_coins),
                }
            )
        )
    if not parts:
        result = pd.DataFrame(
            {
                "article_id": pd.Series(dtype=str),
                "timestamp_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                "coins": pd.Series(dtype=object),
            }
        )
    else:
        joined = pd.concat(parts, ignore_index=True)
        if (joined.groupby("article_id").timestamp_utc.nunique() > 1).any():
            raise ValueError("Conflicting publication timestamp for duplicate article id")
        # Usually already unique: combine tags only for duplicate IDs.
        unique = joined.loc[~joined.article_id.duplicated(keep=False)]
        duplicate = joined.loc[joined.article_id.duplicated(keep=False)]
        merged = duplicate.groupby("article_id", as_index=False).agg(
            timestamp_utc=("timestamp_utc", "first"),
            coins=("coins", lambda rows: sorted({coin for row in rows for coin in row})),
        )
        result = (
            pd.concat([unique, merged], ignore_index=True)
            .sort_values(["timestamp_utc", "article_id"])
            .reset_index(drop=True)
        )
    result.attrs["assumptions"] = list(NEWS_ASSUMPTIONS)
    return result


def _article_days(articles, coin):
    selected = (
        articles if coin == "ALL" else articles.loc[articles.coins.map(lambda tags: coin in tags)]
    )
    return selected.timestamp_utc.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()


def coverage(articles, coins):
    """Empirical coverage per coin plus ALL, including NaT for absent tags."""
    rows = []
    for coin in dict.fromkeys([*coins, "ALL"]):
        days = _article_days(articles, coin)
        rows.append({"id": coin, "first_day": days.min(), "last_day": days.max()})
    return pd.DataFrame(rows)


def aggregate_daily(articles, coins, cov, lookbacks=(1, 3)):
    """Count half-open UTC days on the union coverage grid, without filling gaps.

    Zero is assigned only within an established coverage interval. The caller
    reindexes onto its desired panel extent; outside this grid values stay NaN.
    A trailing count needs every day of its window covered.
    """
    if not lookbacks or any(not isinstance(days, int) or days < 1 for days in lookbacks):
        raise ValueError("lookbacks must be positive integers")
    columns = ["date", "id", *[f"news_count_{days}d" for days in lookbacks]]
    if cov.first_day.isna().all():
        return pd.DataFrame(columns=columns).astype({"date": "datetime64[ns]"})
    grid = pd.date_range(cov.first_day.min(), cov.last_day.max(), freq="D")
    bounds = cov.set_index("id")
    result = []
    for coin in dict.fromkeys([*coins, "ALL"]):
        first, last = bounds.loc[coin, ["first_day", "last_day"]]
        counts = (
            _article_days(articles, coin).value_counts().reindex(grid, fill_value=0).astype(float)
        )
        counts = counts.where((grid >= first) & (grid <= last))
        part = pd.DataFrame({"date": grid, "id": coin})
        for days in lookbacks:
            part[f"news_count_{days}d"] = counts.rolling(days, min_periods=days).sum().to_numpy()
        result.append(part)
    return pd.concat(result, ignore_index=True)[columns]
