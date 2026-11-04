"""Build the private daily finance panel and portable, model-free fev bundle."""

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import fetch_calendars, fetch_news, fetch_prices, fetch_signals
from .fev_io import write_fev_parquet
from .sources import SOURCES

KNOWN_COLUMNS = ["holiday_us", "holiday_gf", "fomc_day"]
NEWS_COLUMNS = ["news_count_1d", "news_count_3d", "news_all_count_1d"]
GLOBAL_COLUMNS = [
    "fng",
    "gpr",
    "gpr_act",
    "gpr_threat",
    "vix",
    "dxy",
    "spx_ret",
    "wti_ret",
    "btc_ret",
    "eth_ret",
    "gold_ret",
]
CUTOFFS = {2022: "2022-01-03", 2023: "2023-01-03", 2024: "2024-01-02", 2025: "2025-01-02"}
HORIZON, NUM_WINDOWS, WINDOW_STEP = 13, 18, 18
ASSUMPTIONS = [
    *fetch_news.NEWS_ASSUMPTIONS,
    "F&G uses the provider's 00:00 UTC day labels; publication vintages are not verified.",
    *fetch_signals.GPR_ASSUMPTIONS,
    *fetch_calendars.FOMC_ASSUMPTIONS,
    "GDELT reconstructed raw counts/tone; ingestion-time vintages are not verified.",
    "Yahoo GC=F and SI=F are continuous futures proxies; close/settlement and roll semantics are unverified.",
    "Issuance is after UTC day T closes; past features use source labels through T without filling.",
    "Session returns use consecutive observed closes; non-positive endpoints invalidate the return.",
    "Session masks in evaluation come from valid Yahoo closes, not a prospective exchange calendar; "
    "compound metals price paths only over those slots. Unreported missing sessions cannot be identified.",
    "quote_missing means Yahoo explicitly returned a null close; it does not infer expected sessions.",
    "The shipped seasonal-naive baseline may fill its own inputs; the producer never does.",
    "Coverage train/test ratios include all calendar days in [build start, initial cutoff] and "
    "[initial cutoff + 1, last future day], including gaps between windows and pre-listing days.",
    "News no-lookahead checks hold archive coverage fixed; empirical coverage bounds are not an ingestion-vintage audit.",
]


@dataclass
class BuildInputs:
    prices: pd.DataFrame
    news: pd.DataFrame
    signals: pd.DataFrame
    calendar: pd.DataFrame
    gdelt: dict
    assumptions: list[str] = field(default_factory=list)


def feature_roles(keep_gdelt):
    return {
        "static": ["asset_class"],
        "known_dynamic": list(KNOWN_COLUMNS),
        "past_dynamic": [
            *NEWS_COLUMNS,
            *(fetch_signals.GDELT_COLUMNS if keep_gdelt else ()),
            *GLOBAL_COLUMNS,
        ],
    }


def _dates(frame, keys):
    data = frame.copy()
    data.attrs = {}
    data["date"] = pd.to_datetime(data.date)
    if (
        data.date.dt.tz is not None
        or data.date.isna().any()
        or not data.date.eq(data.date.dt.normalize()).all()
    ):
        raise ValueError("Expected nonmissing timezone-naive midnight date labels")
    if data.duplicated(keys).any():
        raise ValueError(f"Duplicate source keys: {keys}")
    return data


def build_long(raw, start, end):
    """Assemble BuildInputs into (panel with statics, scalar statics, price sidecar).

    Session returns precede daily reindexing. No join fills a missing observation.
    Each ID starts at its first valid close in the inclusive requested interval.
    """
    start, end = fetch_prices._day(start), fetch_prices._day(end)
    if end < start:
        raise ValueError("end precedes start")
    quotes = _dates(raw.prices, ["id", "date"])
    quotes = quotes.loc[quotes.date.between(start, end)].copy()
    if not np.isfinite(quotes.close.to_numpy(dtype=float)).all():
        raise ValueError("Input prices must contain only valid observed quotes")
    returns = fetch_prices.session_returns(quotes)
    news = _dates(raw.news, ["id", "date"])
    signals = _dates(raw.signals, ["date"])
    calendar = _dates(raw.calendar, ["date"])
    roles = feature_roles(raw.gdelt["keep"])
    globals_ = [c for c in roles["past_dynamic"] if c not in NEWS_COLUMNS]
    if raw.gdelt["keep"] and not fetch_signals.gdelt_preflight(signals, start, end)["keep"]:
        raise ValueError("GDELT keep decision disagrees with input coverage")
    all_news = news.loc[news.id == "ALL", ["date", "news_count_1d"]].rename(
        columns={"news_count_1d": "news_all_count_1d"}
    )
    panels, sidecars, statics = [], [], []
    missing = raw.prices.attrs.get("missing_quote_dates", {})
    for asset in sorted(fetch_prices.PANEL_SYMBOLS):
        observed = quotes.loc[quotes.id == asset, ["date", "close"]]
        if observed.empty:
            raise ValueError(f"No observed prices for {asset}")
        asset_class = "metals" if asset in {"GOLD", "SILVER"} else "crypto"
        grid = pd.DataFrame({"date": pd.date_range(observed.date.min(), end, freq="D")})
        price = grid.merge(observed, on="date", how="left", validate="1:1")
        price["id"] = asset
        price["close"] = price.close.astype(np.float64)
        price["session"] = price.close.notna()
        price["quote_missing"] = price.date.isin(pd.to_datetime(missing.get(asset, [])))
        if (price.session & price.quote_missing).any():
            raise ValueError(f"Conflicting valid/missing quote for {asset}")
        sidecars.append(price[["id", "date", "close", "session", "quote_missing"]])
        target = returns.loc[returns.id == asset, ["date", "ret"]].rename(columns={"ret": "target"})
        item = grid.merge(target, on="date", how="left", validate="1:1")
        per_asset = news.loc[news.id == asset, ["date", "news_count_1d", "news_count_3d"]]
        item = item.merge(per_asset, on="date", how="left", validate="1:1")
        if asset_class == "metals":
            item[["news_count_1d", "news_count_3d"]] = np.nan
        for source in (all_news, signals[["date", *globals_]], calendar[["date", *KNOWN_COLUMNS]]):
            item = item.merge(source, on="date", how="left", validate="1:1")
        if not item[KNOWN_COLUMNS].isin([0, 1]).all().all():
            raise ValueError("Known calendar flags must be 0/1 over the whole grid")
        numeric = ["target", *roles["known_dynamic"], *roles["past_dynamic"]]
        item[numeric] = item[numeric].astype(np.float64)
        if np.isinf(item[numeric].to_numpy()).any():
            raise ValueError("Infinite target/feature value")
        item["id"], item["asset_class"] = asset, asset_class
        item = item.rename(columns={"date": "timestamp"})
        panels.append(
            item[
                [
                    "id",
                    "timestamp",
                    "target",
                    *roles["static"],
                    *roles["known_dynamic"],
                    *roles["past_dynamic"],
                ]
            ]
        )
        statics.append({"id": asset, "asset_class": asset_class})
    return (
        pd.concat(panels, ignore_index=True),
        pd.DataFrame(statics),
        pd.concat(sidecars, ignore_index=True),
    )


def fetch_gdelt_with_retry(raw_dir, start, end, *, refresh=False):
    """One normal pass and, only after HTTP 429, one slower cached retry pass.

    Stop a pass on exhausted 429 retries: hitting the remaining history endpoints
    under a host rate limit cannot establish complete coverage. Never loop passes.
    """
    attempts = []
    for attempt in range(2):
        if attempt:
            print("GDELT 429: waiting 60s before the single slower retry pass", flush=True)
            time.sleep(60)
        frame = fetch_signals.fetch_gdelt_daily(
            raw_dir,
            start=start,
            end=end,
            refresh=refresh if not attempt else False,
            interval=5 if not attempt else 10,
            retry_wait=0 if not attempt else 60,
            stop_on_rate_limit=True,
        )
        result = fetch_signals.gdelt_preflight(frame, start, end)
        errors = frame.attrs.get("errors", [])
        attempts.append(
            {
                **result,
                "errors": errors,
                "interval_seconds": 5 if not attempt else 10,
                "retry_wait_seconds": 0 if not attempt else 60,
                "stopped_on_rate_limit": frame.attrs.get("stopped_on_rate_limit", False),
            }
        )
        if result["keep"] or not any(error.get("http_status") == 429 for error in errors):
            break
    return frame, {
        **result,
        "attempts": attempts,
        "errors": [e for a in attempts for e in a["errors"]],
    }


def fetch_inputs(raw_dir, start, end, *, refresh=False):
    """Fetch sequentially with the shared cache transport; no models imported."""
    raw_dir = Path(raw_dir)
    print("Fetching Yahoo panel and cross-market quotes", flush=True)
    prices = fetch_prices.fetch_panel(
        {**fetch_prices.PANEL_SYMBOLS, **fetch_prices.CROSS_SYMBOLS},
        start,
        end,
        raw_dir=raw_dir / "yahoo",
        refresh=refresh,
    )
    print("Loading pinned CryptoPanic archive", flush=True)
    files = fetch_news.download_archive(
        raw_dir / "news", revision=fetch_news.CRYPTOPANIC_REVISION, refresh=refresh
    )
    articles = fetch_news.normalize_articles(
        [
            pd.read_csv(path, dtype={"id": str}, usecols=["id", "newsDatetime", "currencies"])
            for path in files
        ]
    )
    coins = list(fetch_prices.PANEL_SYMBOLS)
    news = fetch_news.aggregate_daily(articles, coins, fetch_news.coverage(articles, coins))
    print("Fetching F&G, GPR, calendars and cross-market features", flush=True)
    signals = (
        fetch_signals.fetch_fng(raw_dir / "fng", refresh=refresh)
        .merge(
            fetch_signals.fetch_gpr(raw_dir / "gpr", refresh=refresh),
            on="date",
            how="outer",
            validate="1:1",
        )
        .merge(
            fetch_signals.cross_series(start, end, raw_dir / "yahoo", refresh=False),
            on="date",
            how="outer",
            validate="1:1",
        )
    )
    dates = pd.date_range(start, end, freq="D")
    calendar = fetch_calendars.holiday_flags(dates).merge(
        fetch_calendars.fomc_flags(dates, raw_dir / "fomc", refresh=refresh),
        on="date",
        validate="1:1",
    )
    print("Running GDELT historical preflight", flush=True)
    gdelt, decision = fetch_gdelt_with_retry(raw_dir / "gdelt", start, end, refresh=refresh)
    if decision["keep"]:
        signals = signals.merge(gdelt, on="date", how="outer", validate="1:1")
    return BuildInputs(prices, news, signals, calendar, decision)


def task_definitions(roles, arm):
    return [
        {
            "name": f"probe_{year}_{arm}",
            "dataset_path": f"panel_{arm}.parquet",
            "horizon": HORIZON,
            "num_windows": NUM_WINDOWS,
            "window_step_size": WINDOW_STEP,
            "initial_cutoff": cutoff,
            "seasonality": 7,
            "quantile_levels": [q / 10 for q in range(1, 10)],
            "eval_metric": "SQL",
            "extra_metrics": ["MASE", "WAPE", "WQL"],
            "id_column": "id",
            "timestamp_column": "timestamp",
            "target": "target",
            "static_columns": roles["static"],
            "known_dynamic_columns": roles["known_dynamic"],
            "past_dynamic_columns": roles["past_dynamic"],
        }
        for year, cutoff in CUTOFFS.items()
    ]


def _day_string(value):
    return None if pd.isna(value) else str(pd.Timestamp(value).date())


def _stats(values, start, end, *, numeric=True):
    values = values.reindex(pd.date_range(start, end, freq="D"))
    finite = np.isfinite(values.to_numpy(dtype=float)) if numeric else values.notna().to_numpy()
    dates = values.index[finite]
    return {
        "start": _day_string(start),
        "end": _day_string(end),
        "days": len(values),
        "finite_count": int(finite.sum()),
        "finite_ratio": float(finite.mean()) if len(values) else None,
        "first_finite_date": _day_string(dates.min()),
        "last_finite_date": _day_string(dates.max()),
    }


def coverage_report(panel, roles, start, end):
    result = {}
    for asset, frame in panel.groupby("id", sort=True):
        frame = frame.set_index("timestamp")
        result[asset] = {}
        for name in ["target", *roles["static"], *roles["known_dynamic"], *roles["past_dynamic"]]:
            numeric = name not in roles["static"]
            record = _stats(frame[name], frame.index.min(), end, numeric=numeric)
            record["folds"] = {}
            for year, cutoff in CUTOFFS.items():
                cutoff = pd.Timestamp(cutoff)
                last = cutoff + pd.Timedelta(days=(NUM_WINDOWS - 1) * WINDOW_STEP + HORIZON)
                record["folds"][str(year)] = {
                    "train": _stats(frame[name], start, cutoff, numeric=numeric),
                    "test": _stats(
                        frame[name], cutoff + pd.Timedelta(days=1), last, numeric=numeric
                    ),
                }
            result[asset][name] = record
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance(raw_dir):
    records = []
    for source in SOURCES:
        record = asdict(source)
        directory = Path(raw_dir) / {"cryptopanic": "news", "reference_crypto": "reference"}.get(
            source.name, source.name
        )
        record["cache"] = []
        for path in sorted(directory.rglob("*.meta.json")):
            meta = json.loads(path.read_text())
            body_path = Path(str(path).removesuffix(".meta.json"))
            if sha256(body_path) != meta["sha256"]:
                raise ValueError(f"Cache integrity mismatch: {body_path}")
            record["cache"].append({**meta, "cache_path": str(body_path.relative_to(raw_dir))})
        record["pinned_revision"] = (
            fetch_news.CRYPTOPANIC_REVISION if source.name == "cryptopanic" else None
        )
        record["source_manifests"] = [
            json.loads(path.read_text()) for path in sorted(directory.rglob("*_manifest.json"))
        ]
        record["role"] = (
            "reference_only_never_input" if source.kind == "reference" else "input_candidate"
        )
        records.append(record)
    return records


def write_bundle(raw, out, *, start, end, raw_dir=None, refresh=False):
    """Write seven payload files, then manifest and non-circular checksums."""
    start, end = fetch_prices._day(start), fetch_prices._day(end)
    last_future = pd.Timestamp(CUTOFFS[2025]) + pd.Timedelta(
        days=(NUM_WINDOWS - 1) * WINDOW_STEP + HORIZON
    )
    if end < last_future:
        raise ValueError(f"end must cover all folds through {last_future.date()}")
    panel, _, prices = build_long(raw, start, end)
    if (panel.groupby("id").timestamp.min() > pd.Timestamp(CUTOFFS[2022])).any():
        raise ValueError("Every id must have history at the first cutoff")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    roles = feature_roles(raw.gdelt["keep"])
    payloads = []
    profiles = {}
    for arm in ("on", "off"):
        twin = panel.copy()
        if arm == "off":
            twin[roles["known_dynamic"] + roles["past_dynamic"]] = np.nan
        name = f"panel_{arm}.parquet"
        write_fev_parquet(twin, roles, out / name)
        payloads.append(name)
        tasks = task_definitions(roles, arm)
        name = f"probe_{arm}.yaml"
        (out / name).write_text(yaml.safe_dump({"tasks": tasks}, sort_keys=False))
        payloads.append(name)
        profiles.update(
            {
                task["name"]: {"target_support": "real", "categorical_statics": ["asset_class"]}
                for task in tasks
            }
        )
    smoke = {**task_definitions(roles, "on")[0], "name": "probe_smoke"}
    (out / "probe_smoke.yaml").write_text(yaml.safe_dump({"tasks": [smoke]}, sort_keys=False))
    profiles["probe_smoke"] = {"target_support": "real", "categorical_statics": ["asset_class"]}
    (out / "profiles.yaml").write_text(yaml.safe_dump(profiles, sort_keys=False))
    prices.to_parquet(out / "prices.parquet", index=False)
    payloads.extend(["probe_smoke.yaml", "profiles.yaml", "prices.parquet"])
    coverage = coverage_report(panel, roles, start, end)
    symbols = {**fetch_prices.PANEL_SYMBOLS, **fetch_prices.CROSS_SYMBOLS}
    missing = raw.prices.attrs.get("missing_quote_dates", {})
    manifest = {
        "build_args": {
            "raw": str(raw_dir) if raw_dir is not None else None,
            "out": str(out),
            "start": str(start.date()),
            "end": str(end.date()),
            "refresh": refresh,
        },
        "built_at": datetime.now(timezone.utc).isoformat(),
        "repo_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip(),
        "sources": source_provenance(raw_dir) if raw_dir is not None else [],
        "assumptions": list(dict.fromkeys([*ASSUMPTIONS, *raw.assumptions])),
        "features": roles,
        "coverage": coverage,
        "gdelt_preflight": raw.gdelt,
        "targets": {
            asset: {
                "first_date": stats["target"]["start"],
                "last_date": stats["target"]["end"],
                "rows": stats["target"]["days"],
                "target_count": stats["target"]["finite_count"],
            }
            for asset, stats in coverage.items()
        },
        "missing_quote_dates": {
            symbol: missing.get(asset, []) for asset, symbol in symbols.items()
        },
        "output_sha256": {name: sha256(out / name) for name in sorted(payloads)},
        "checksum_convention": "manifest hashes all seven payloads; SHA256SUMS hashes payloads plus manifest. Neither file hashes itself, avoiding circular digests.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (out / "SHA256SUMS").write_text(
        "".join(f"{sha256(out / name)}  {name}\n" for name in sorted([*payloads, "manifest.json"]))
    )
    return manifest


def validate_bundle_windows(out):
    """Acceptance audit through the shipped specs/fev API, without any model."""
    from demand_forecast.bench.specs import load_task_specs

    out = Path(out)
    expected_ids = sorted(fetch_prices.PANEL_SYMBOLS)
    result = {"tasks": 0, "windows": 0, "series_windows": 0, "ids": expected_ids}
    for arm in ("on", "off", "smoke"):
        for spec in load_task_specs(out / f"probe_{arm}.yaml", out / "profiles.yaml"):
            result["tasks"] += 1
            for window in spec.to_task().iter_windows(num_proc=1):
                result["windows"] += 1
                past, future = window.get_input_data()
                truth = window.get_ground_truth()
                cutoff = pd.Timestamp(window.cutoff)
                expected = pd.date_range(cutoff + pd.Timedelta(days=1), periods=HORIZON).to_numpy(
                    dtype="datetime64[ns]"
                )
                assert list(past["id"]) == list(future["id"]) == list(truth["id"]) == expected_ids
                date_columns = ["id", "timestamp"]
                for history, forecast, target in zip(
                    past.select_columns(date_columns),
                    future.select_columns(date_columns),
                    truth.select_columns(date_columns),
                    strict=True,
                ):
                    assert pd.Timestamp(history["timestamp"][-1]) == cutoff
                    np.testing.assert_array_equal(forecast["timestamp"], expected)
                    np.testing.assert_array_equal(target["timestamp"], expected)
                    result["series_windows"] += 1
    return result


def crosscheck_reference(prices, reference):
    """Compare normalized reference id/date/close rows; never consumed by build."""
    reference = _dates(reference, ["id", "date"])
    result = {}
    for asset in ("BTC", "ETH"):
        pairs = (
            prices.loc[prices.id == asset, ["date", "close"]]
            .merge(
                reference.loc[reference.id == asset, ["date", "close"]],
                on="date",
                suffixes=("_yahoo", "_reference"),
                validate="1:1",
            )
            .dropna()
        )
        pairs = pairs.loc[pairs.close_reference > 0]
        relative = (pairs.close_yahoo - pairs.close_reference).abs() / pairs.close_reference
        result[asset] = {
            "overlap_count": len(pairs),
            "max_relative_difference": float(relative.max()) if len(pairs) else None,
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/world-signals/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/world-signals/built"))
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", required=True, help="Fixed inclusive end date, e.g. 2025-12-31")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    raw = fetch_inputs(args.raw, args.start, args.end, refresh=args.refresh)
    manifest = write_bundle(
        raw, args.out, start=args.start, end=args.end, raw_dir=args.raw, refresh=args.refresh
    )
    print(
        json.dumps(
            {"out": str(args.out), "targets": manifest["targets"], "gdelt_keep": raw.gdelt["keep"]},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
