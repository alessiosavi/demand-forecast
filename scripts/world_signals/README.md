# Finance world-signals probe

External producers only; the forecasting model is unchanged. Run Python from the
repository root so `scripts.world_signals` is importable. Raw downloads and bundles
belong under `data/world-signals/` (ignored); Yahoo-derived data stays private.
Fixtures contain synthetic observations in the providers' schemas.

Pipeline sequence: fetch prices → news → world-state signals → calendars → build
ON/OFF datasets → private bundle → Colab GPU runs → report. The builder, bundle and
report entry points are later handoffs; this handoff provides the fetcher APIs.
No local training. The eventual task profiles must set `target_support: real`;
the supplied recipe bases retain their original defaults for exact comparison.

## Fetcher contracts

- `fetch_prices.fetch_panel(PANEL_SYMBOLS, start, end, raw_dir=...)` returns
  observed sessions. Date ranges are inclusive. Exchange timestamps use New York
  dates (including DST); crypto uses UTC. The current local calendar day is
  conservatively excluded. `session_returns` runs before daily reindexing; it
  never fills missing prices, and a non-positive close invalidates both adjacent
  returns. Explicit null quote dates remain in `frame.attrs['missing_quote_dates']`;
  unreported expected sessions still require the later builder's session calendar.
- `fetch_news.download_archive(dest, revision=CRYPTOPANIC_REVISION)` extracts the
  joined CSV using `bsdtar -xf`. Read it with `pd.read_csv(..., dtype={'id': str})`,
  then call `normalize_articles`, `coverage`, and `aggregate_daily`. Dedupe by ID,
  union duplicate tags, and reject conflicting publication timestamps. Votes are
  excluded. Counts use half-open UTC days; zero is allowed only within coverage.
  Trailing counts require every day covered. Reindex the result for panel dates
  outside the union coverage grid; those cells must remain NaN.
- `fetch_signals.fetch_fng`, `fetch_gpr`, and `cross_series` return unfilled
  observations. GPR includes only GPRD/GPRD_ACT/GPRD_THREAT, never moving averages.
  Cross-market levels and returns stay on each series' session dates.
- `fetch_signals.fetch_gdelt_daily` returns daily raw counts and tone, mapping
  bitcoin/ethereum/gold to btc/eth/gold column names. Request chunks span at most
  one year; tails of a week or less are padded backwards to obtain daily bins,
  then trimmed. Unavailable/invalid chunks remain NaN with details in
  `frame.attrs['errors']`. `gdelt_preflight(frame)` checks all six columns against
  2018-01-01–2025-12-31. The builder must drop **all** GDELT columns from both twins
  unless `keep` is true (every column has at least 99% finite days).
- `fetch_calendars.holiday_flags(dates)` emits observed federal holidays and Good
  Friday. `fomc_flags(dates, raw_dir)` uses scheduled meeting `endDate`; it excludes
  unscheduled meetings and ignores cancellation/status and actual statement dates.

All network entry points accept `opener` and `raw_dir`; the opener follows
`urllib.request.urlopen(Request, timeout=...)`. Raw responses have adjacent
`*.meta.json` files containing URL, fetch time, HTTP status and SHA-256. Cache
integrity is checked on reuse; `refresh=True` deliberately refetches. Yahoo is
paced at one request per second, GDELT at one per five seconds, including retries.
There are at most three attempts, only for HTTP 429. Pacing is shared within one
process; run producers sequentially, not as concurrent processes.

## Provenance and limitations

CryptoPanic is pinned to commit
`a3cac04997702dd5a05c312647e2038b3629caff` (GitHub API resolution 2026-11-04).
The joined RAR alone contains all fields needed for these features. Its SHA,
revision, extracted CSV SHA, and assumptions are in `archive_manifest.json`.
The FOMC ZIP and embedded CSV hashes are in `fomc_manifest.json`; download times
are in their adjacent cache metadata. The future bundle manifest must incorporate
these records, `SOURCES`, and each frame's `assumptions`/`errors` attributes.

Offset-free news timestamps are **assumed UTC**. Coverage is empirical first/last
publication per coin (plus ALL); zeros between those bounds assume a complete
archive, which has not been independently audited. Metals lack CryptoPanic tags.
GPR is revised weekly-published history, not vintage-exact availability. F&G,
GDELT, and FOMC historical publication/schedule vintages are also not verified.
Yahoo GC/SI are continuous futures proxies; close/settlement and roll semantics
are not established. No halving, macro-release, dominance or vote features exist.
