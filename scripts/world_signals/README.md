# Finance world-signals probe

External producers only; the forecasting model is unchanged. Run Python from the
repository root so `scripts.world_signals` is importable. Raw downloads and bundles
belong under `data/world-signals/` (ignored); Yahoo-derived data stays private.
Fixtures contain synthetic observations in the providers' schemas.

Pipeline sequence: fetch prices → news → world-state signals → calendars → build
ON/OFF datasets → private bundle → Colab GPU runs → report.
No local training. The task profiles set `target_support: real`;
the supplied recipe bases retain their original defaults for exact comparison.

## Fetcher contracts

- `fetch_prices.fetch_panel(PANEL_SYMBOLS, start, end, raw_dir=...)` returns
  observed sessions. Date ranges are inclusive. Exchange timestamps use New York
  dates (including DST); crypto uses UTC. The current local calendar day is
  conservatively excluded. `session_returns` runs before daily reindexing; it
  never fills missing prices, and a non-positive close invalidates both adjacent
  returns. Explicit null quote dates remain in `frame.attrs['missing_quote_dates']`;
  the sidecar does not infer unreported expected sessions from a calendar.
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
are in their adjacent cache metadata. The bundle manifest incorporates
these records, `SOURCES`, source assumptions, and GDELT errors.

Offset-free news timestamps are **assumed UTC**. Coverage is empirical first/last
publication per coin (plus ALL); zeros between those bounds assume a complete
archive, which has not been independently audited. Metals lack CryptoPanic tags.
GPR is revised weekly-published history, not vintage-exact availability. F&G,
GDELT, and FOMC historical publication/schedule vintages are also not verified.
Yahoo GC/SI are continuous futures proxies; close/settlement and roll semantics
are not established. No halving, macro-release, dominance or vote features exist.

## Build the private bundle (no model)

```bash
./.venv/bin/python -m scripts.world_signals.build_dataset \
  --raw data/world-signals/raw --out data/world-signals/built \
  --start 2018-01-01 --end 2025-12-31
```

The inclusive `--end` is required and fixed; use `--refresh` only to deliberately
refetch sources. The builder uses cached successes, never fills missing values,
and puts each asset on its own daily grid from its first valid close through end.
Metals returns are computed between observed sessions before reindexing. The
first target per asset and all non-session metals targets are NaN.

Outputs under `built/`:

- `panel_on.parquet` and `panel_off.parquet`: one row per sorted ID, scalar string
  ID/asset class, timezone-naive nanosecond timestamp lists and float64 lists.
  OFF has identical columns and dates, with all world-signal dynamics set to NaN.
- `prices.parquet`: daily long `id,date,close,session,quote_missing`. `session`
  means a valid Yahoo close; `quote_missing` means an explicitly returned null
  quote. This retrospective mask is used for evaluation, not model features.
- `probe_on.yaml`, `probe_off.yaml`, `probe_smoke.yaml`, `profiles.yaml`: portable
  relative dataset paths, explicit roles, and fixed 2022–2025 folds (18 windows,
  stride 18, horizon 13, seasonality 7). Smoke is the full 2022 ON task.
- `manifest.json`: build arguments, checkout HEAD, source/cache identities and
  assumptions, GDELT decisions/errors, per-asset feature coverage and target
  counts, missing-quote dates, and hashes of the seven files above. The checkout
  SHA is the committed base; `repo_dirty` records tracked uncommitted changes
  (`git status --porcelain --untracked-files=no`). Commit the reviewed builder
  before remote execution.
- `SHA256SUMS`: hashes of the seven payloads plus the manifest. The manifest and
  checksum file cannot hash themselves, so neither includes a self digest.

GDELT is retained only if all six columns have at least 99% finite days over the
requested interval. Each pass stops on exhausted HTTP 429 retries. One slower
retry pass follows a 60-second wait, reusing cached successes, with at least
10 seconds between requests and 60 seconds between 429 retries. No further pass
is made; incomplete GDELT is omitted from both twins and YAMLs.

Coverage includes first/last finite dates, counts and ratios for each feature,
including target and static availability. Fold train periods are build start
through the inclusive initial cutoff; test periods run from the next day through
the last future day, including days between evaluation windows. Pre-listing days
count as unavailable in fold ratios; overall ratios cover each asset's own grid.
News coverage uses the pinned archive's empirical bounds: this is not an audit of
when the archive historically became available.

`build_long(BuildInputs(...), start, end)` is the pure assembly API used by tests;
`write_bundle` uses it for the real outputs. `fev_io.write_fev_parquet` accepts a
long frame and roles with `static`, `known_dynamic`, `past_dynamic` lists;
`read_fev_parquet` expands it back to a long frame. Arrow serialization preserves
IEEE NaNs within lists and rejects varying statics, duplicate dates and gaps.

```bash
(cd data/world-signals/built && shasum -a 256 -c SHA256SUMS)
HF_HOME=data/world-signals/hf-cache ./.venv/bin/python -c \
  'from scripts.world_signals.build_dataset import validate_bundle_windows; print(validate_bundle_windows("data/world-signals/built"))'
```

The second command audits all ON/OFF/smoke windows via the shipped fev task loader
without executing a model. It checks all ten IDs, inclusive past cutoffs and all
13 future timestamps. The optional reference test reads normalized
`raw/reference/closes.csv` (`id,date,close`, BTC/ETH); it reports discrepancies
without treating that reference as an input or an authoritative replacement.

## Offline rebuild and portable Colab bundle

```bash
./.venv/bin/python -m scripts.world_signals.build_dataset --offline \
  --raw data/world-signals/raw --out data/world-signals/offline-rebuild \
  --start 2018-01-01 --end 2025-12-31
./.venv/bin/python -m scripts.world_signals.colab_bundle build \
  --built data/world-signals/built --out data/world-signals/colab-bundle
./.venv/bin/python -m scripts.world_signals.colab_bundle verify \
  --bundle data/world-signals/colab-bundle
```

`--offline` uses cached response bytes only. A missing required source fails
with an offline cache-miss error; uncached GDELT chunks remain missing and count
against its coverage threshold. `--offline --refresh` is rejected. Match the
original start/end dates to reuse URL-addressed Yahoo caches. Parquet/YAML hashes
should match; manifest build time, paths, checkout provenance and offline GDELT
error evidence can differ, so manifest/SHA256SUMS hashes can differ too.

The bundle copier copies exactly the files listed in `SHA256SUMS`, plus that
checksum file. Verification rejects changed, missing or extra payload files,
unsafe checksum paths and symlink payloads. Keep reports and logs outside the
bundle directory. Matching partial copies can resume; conflicting destination
files fail without being overwritten. Copy this directory to the owner's private Drive; its relative
YAML paths survive the move. On Colab, after installing the pinned repo revision,
run from the repo root (replace the paths with the actual Drive locations):

```bash
python -m scripts.world_signals.colab_bundle verify \
  --bundle /content/drive/MyDrive/demand-forecast/world-signals/bundle
```

## Baselines, finalization and report (no model execution)

For local data-only checks, keep fev's datasets cache under the ignored data tree
and disable datasets/Hub network access. These environment settings also avoid
metadata lookups for local Parquet files:

```bash
export HF_HOME="$PWD/data/world-signals/hf-cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

Use the following after the Colab runs, replacing `python`/`demand-forecast` with
`./.venv/bin/python`/`./.venv/bin/demand-forecast` for local reporting:

```bash
PROBE_BUNDLE=/content/drive/MyDrive/demand-forecast/world-signals/bundle
PROBE_RUNS=/content/drive/MyDrive/demand-forecast/world-signals/runs
PROBE_REPORT=/content/drive/MyDrive/demand-forecast/world-signals/report

for arm in on off; do
  demand-forecast fev baseline --tasks "$PROBE_BUNDLE/probe_${arm}.yaml" \
    --profiles "$PROBE_BUNDLE/profiles.yaml" --out-dir "$PROBE_RUNS/baseline"
  python -m scripts.world_signals.rw_baseline --tasks "$PROBE_BUNDLE/probe_${arm}.yaml" \
    --profiles "$PROBE_BUNDLE/profiles.yaml" --out-dir "$PROBE_RUNS/baseline"
done

python -m scripts.world_signals.run_tools finalize \
  --run-dir "$PROBE_RUNS/l3-on" --dry-run
python -m scripts.world_signals.run_tools finalize --run-dir "$PROBE_RUNS/l3-on"

python -m scripts.world_signals.report --bundle "$PROBE_BUNDLE" \
  --runs "$PROBE_RUNS" --out "$PROBE_REPORT"
python -m scripts.world_signals.report --bundle "$PROBE_BUNDLE" \
  --runs "$PROBE_RUNS/smoke" --out "$PROBE_REPORT-smoke" --partial
```

Run the finalizer for each model run directory, after each completed task if
Drive space is tight. It extracts `history.json` from `checkpoints/latest.pt`
before pruning any `checkpoints/*.pt`; `progress.json` survives. It loads only
our trusted trainer checkpoints with `torch.load(weights_only=False)`. A summary
row, matching identities, bundle manifest, every expected window and saved
history are required for pruning. Incomplete/inconsistent tasks retain every
checkpoint and cause a nonzero exit; other complete tasks can still be finalized.
`--dry-run` loads and checks evidence but writes/deletes nothing.

Full reports require the six directories `scratch-on`, `scratch-off`,
`scratch-off-s1`, `l3-on`, `l3-off`, `l3-off-s1`, each with all four fold tasks.
`baseline/` summaries are cross-checked when supplied; seasonal naive and random
walk are always recomputed. Random-walk CLI results also save window NPZs and
resume by task/model, verifying identity and restoring missing prediction files.
Use a new baseline directory if changing the bundle/task definition.

`--partial` accepts a single run directory (including `probe_smoke`) or a parent
with missing runs/folds. Present tasks must still have complete, valid artifacts.
It writes no verdict and removes any stale verdict at that output path. Changed
identities, task definitions, window sets, fingerprints, or metric discrepancies
fail reporting. Supplied SQL/MASE/WQL/WAPE scores must match within relative
`1e-6`; L3 step-0 SQL must agree within relative `1e-4` when both histories exist.
Missing histories remain unknown. A failed step-0 comparison is written to the
report and causes a nonzero exit.

Outputs are seven CSV tables under `tables/`, `verdict.json` for full reports,
and `REPORT.md` embedding every table and the limitations. `return_cells.parquet`
and `price_paths.parquet` preserve ID/date/step-level calculation evidence.
Task SQL pools items/horizons within each window, then averages window scores as
fev does. Bootstrap SQL uses the mean finite per-cell scaled loss contribution;
all assets/horizons in one window move together (2,000 paired draws, seed 0).
Price WAPE pools each asset's errors/actual closes across the fold before macro
averaging; its bootstrap recomputes these ratios after sampling whole windows.
Price MASE uses the asset's session-close differences through the initial cutoff.
Direction excludes NaN/zero actuals and zero forecasts (RW is undefined, with a
zero denominator count). Price paths are derived point paths, never quantiles.

The decision uses L3 seed 0 only. A fold wins only when SQL improves, macroWAPE
does not worsen, and ON's best step is positive. Best step zero is `no-learning`.
`solid` requires at least three such wins, a pooled SQL interval wholly below
zero, and at least three winning SQL deltas strictly above their seed-noise
floors. Three wins without both gates are `suggestive`; at least three worsening
SQL folds and an interval wholly above zero are `negative`; otherwise the result
is `inconclusive`. Within-noise flags are separate from win/loss/no-learning.
ON's price skill versus flat-last-close random walk is reported separately.
Synthetic `run.json` artifacts with `synthetic: true` clearly mark the resulting
report/verdict as fake pipeline validation, not an experiment.
