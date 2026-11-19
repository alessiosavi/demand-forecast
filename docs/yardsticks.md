# Zero-shot reference models on fev

Run these models on the GPU benchmark host. Do not run real datasets on the Mac.
The runner consumes native fev YAML, including shifted definitions, multiple
simultaneous targets, and names such as `uk_covid_nation_1W/new`. No training role
profile is needed. `--names @names.txt` or `--names name1,name2` selects tasks.

## Separate host environment

For an NVIDIA L4 host with an NVIDIA driver supporting CUDA 12.8 (check
`nvidia-smi`), use Python 3.12 and the official CUDA 12.8 PyTorch wheel:

```bash
python3.12 -m venv ~/venvs/yardsticks
source ~/venvs/yardsticks/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-yardsticks.txt
python -m pip install -e '.[bench]'
python -m pip check
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
python -m pip freeze > yardsticks-environment.txt
```

Use the `cu126` wheel index instead if the installed driver supports CUDA 12.6
but not 12.8. The wheel supplies its CUDA runtime; match the driver's capability,
not a separately installed CUDA toolkit. Wheel commands are from
[PyTorch's versioned installation instructions](https://pytorch.org/get-started/previous-versions/#v280).
Requirements pin the inference packages and numerical stack; `pip freeze` records
remaining transitive dependencies. Keep this environment separate from our model.

```bash
python scripts/fev_yardstick.py --model chronos2 --weights amazon/chronos-2 \
  --revision 29ec3766d36d6f73f0696f85560a422f50e8498c \
  --tasks shifted.yaml --model-name chronos2 --out-dir results/chronos2 --resume --keep-going
python scripts/fev_yardstick.py --model timestral --weights mistralai/Timestral-5M \
  --revision bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f \
  --tasks shifted.yaml --model-name timestral5m --out-dir results/timestral5m --resume --keep-going
```

## Published settings and ablations

Settings follow the public fev adapters pinned at
[fev@9cf943c](https://github.com/autogluon/fev/tree/9cf943c388e5cfaf352175ea3c69e408b197d997/models).
Chronos uses `chronos-forecasting==2.2.2`; Timestral uses
[mistralai/timestral@a534bdb](https://github.com/mistralai/timestral/tree/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff).
Both use float32, joint targets, median point predictions and the nine task
quantiles directly from their native quantile heads. Chronos uses its published
quantile API; Timestral selects exact levels and uses linear interpolation only
for non-native levels, as its published script does. Neither uses static columns.

| Switch | Chronos-2 | Timestral-5M |
|---|---|---|
| default context | checkpoint cap: 8192 for pinned weights | checkpoint cap: 32768 for pinned weights |
| `--max-context N` | runner retains last N target and dynamic history values; passes N to pipeline context limit, capped by checkpoint | runner retains last N history values; forecaster's rolling cap becomes min(N, checkpoint cap) |
| `--covariates all` (default) | past-only and known history plus known future; nonnumeric/bool dynamics cast to strings as upstream | numeric, boolean and string dynamics; dataset-wide type/vocabulary inference; categorical codes target-encoded by the forecaster using only current context (joint multivariate targets retain codes) |
| `--covariates none` | targets only, no covariate history or future | targets only; skips type/vocabulary inference |
| `--cross-series on` (default) | cross-learning within each pipeline batch, including items, targets and covariates | joint targets within each item; separate batch items do not attend to each other |
| `--cross-series off` | splits each item into single-target groups and sets `cross_learning=False`; each target retains its own conditioning covariates | rejected: this ablation is registered for Chronos only |
| `--batch-size N` | variate budget, including covariates; raised to fit one group; default 100 | item limit, default 1024; length-sorted batches across all windows, budget 2^23 rows × padded steps; halves batches on OOM |

Future covariates keep the full forecast horizon. History trimming never shortens
the requested horizon. Context ablations bound information available to fixed
pretrained weights; they do not simulate retraining on shorter histories.
Timestral retains its forecaster defaults: symmetric averaging, nonnegative
clipping, block autoregression and forecast caps. Chronos retains its pipeline's
normalization, category encoding and long-horizon decoding defaults.

## Simons checkpoints: raw and published

Simons is an optional inference dependency, pinned to
[Abel-ai-lab/Simons@8e338e9](https://github.com/Abel-ai-lab/Simons/tree/8e338e915d26c8960713889ef12ab1cf5c08d591).
Install it in the benchmark host environment:

```bash
python -m pip install 'simons @ git+https://github.com/Abel-ai-lab/Simons.git@8e338e915d26c8960713889ef12ab1cf5c08d591'
```

Export our trained checkpoint with Simons's `save_pretrained` first. The runner
requires a local directory containing `config.json` and `model.safetensors`,
loaded with `Simons.from_pretrained`. It does not resolve Simons Hub IDs or accept
`--revision`. `--checkpoint` is an alias for `--weights` for every model.

Choose a processing mode explicitly. Use **raw** for checkpoint selection and
report **published** separately, with distinct labels and output directories:

```bash
python scripts/fev_yardstick.py --model simons --checkpoint /checkpoints/simons/export \
  --processing raw --tasks shifted.yaml --model-name simons25m-raw \
  --out-dir results/simons25m-raw --resume --keep-going
python scripts/fev_yardstick.py --model simons --checkpoint /checkpoints/simons/export \
  --processing published --tasks shifted.yaml --model-name simons25m-published \
  --out-dir results/simons25m-published --resume --keep-going
```

Both modes also accept the official fev task YAML. `--processing` is rejected for
Chronos-2 and Timestral; their settings and resume identities are unchanged.

**Raw** replaces all six inference-rule flags in the loaded frozen config with
`False`, even if the export enables them: `views`, `sign_clip`, `integer_snap`,
`fit_row`, `xreg`, and `xreg_guard`. This removes transformed views and pooled tail
distributions, sign clipping, integer snapping, the in-context ridge-fit row,
ridge residual/add-back processing and its backtest guard. Learned covariate
modules (including binary rows and learned reads), intrinsic normalization and
inversion, and quantile ordering remain enabled as configured. Horizons beyond
the model's single-pass prediction length roll forward on the median in raw mode.
**Published** leaves the exported config untouched, including any history-based
XReg limits; the actual defaults depend on the export.

Each item's targets are stacked together. Contiguous blocks of at most 32 targets
match upstream's group split; items never share a group. Known-future covariates
include the aligned historical values and exactly the forecast horizon.
`shared_categories=False` retains upstream's separate history/horizon category
encoding, and upstream limits known-future inputs to 16 usable columns. A
checkpoint needs learned covariate support to consume these inputs; otherwise
use `--covariates none`. Static columns are unused. Unsupported past-only
covariates are dropped. `dropped_past_only_covariates`, in each summary row and
`run.json`'s `input_metadata`, counts dropped column occurrences per item per
window, before target-group splitting. With `--covariates none`, this count is
zero because the shared ablation already removes every covariate.

`--max-context N` retains the last N target and covariate history values; upstream
also applies the checkpoint's context cap. `--covariates none` supplies targets
only. `--batch-size` defaults to upstream's fev wrapper value of 256 groups per
batch; `--cross-series off` is rejected. Following upstream's benchmark script,
the adapter sorts the nine quantile predictions (levels 0.1–0.9) and linearly
interpolates requested fev levels, clamping outside that range. The fifth
quantile (median) supplies the point forecast, even if fev omits 0.5.

The processing mode, SHA-256 content identity of `config.json` and
`model.safetensors`, and the pinned upstream commit enter the run identity.
The installed Simons Python source is also content-hashed, and its actual VCS
commit is recorded when installation metadata provides it. This detects changed
upstream commits and edited installations even if their package version is equal.
Unrelated export files, such as training logs, do not affect it. Changing a mode,
checkpoint or inference implementation requires a new output directory; resume
refuses mismatches before touching existing results.

The Simons conversion follows its Apache-2.0 benchmark implementation, Copyright
2026 Abel AI Lab. This product includes software developed at Abel AI Lab
(https://abel.ai).

## v5 exports

Pass a local export directory containing `config.json`, `model.safetensors`
and `parity.npz`. The adapter validates the export and checks parity on CPU
and on the selected device before scoring. The run identity includes the export
digest, package versions and a hash of the Python source under `demand_forecast/`.

```bash
python scripts/fev_yardstick.py --model v5 --checkpoint /checkpoints/v5/export \
  --processing raw --tasks shifted.yaml --model-name v5-full \
  --out-dir results/v5-full --resume --keep-going
```

v5 receives the full target and covariate histories provided by fev, the known
future covariates, timestamp axes, statics, and declared column types. A dated
panel uses each item's own forecast origin. Items outside the proven grid run
in individual panels without statics. A window with more than 16 dated origins
also uses individual panels. Native forecast quantiles are selected directly.
Other requested quantiles are interpolated linearly with constant tails.

`--processing raw`, `--covariates all` and `--cross-series on` are required.
The default encoder context is the largest context in the training sampler.
`--max-context` sets the encoder context, rounds down to a multiple of 16 and
caps it at 2,048. The minimum is 16. Scaling and panel statistics still use
the full available history up to the forecast origin. `--batch-size` counts
episodes and defaults to 64. Linux CUDA scoring uses 16 episode-building
workers when at least 18 CPU cores are available to the process.

The default `--variant full` uses the export's trained drops and summary
mask. Those settings apply to every variant. Ablations retain the full
variant's target layout, with explicit per-target layouts for `per_target`
and `no_siblings`.

| Variant | Scoring change |
|---|---|
| `full` | Baseline with the export's training settings |
| `no_features` | Remove all dynamic covariates from the panel |
| `no_known` | Remove known-future covariates, including their history |
| `no_past` | Remove past-only covariates |
| `no_dynamic_categories` | Remove categorical dynamic covariates |
| `no_statics` | Remove all static covariates |
| `no_static_categories` | Remove categorical static covariates |
| `no_static_numbers` | Remove numerical static covariates |
| `no_peers` | Drop peer rows |
| `no_panel_row` | Drop the whole-panel row |
| `no_typed_context` | Drop attribute tokens, static numbers, peer and whole-panel rows, and category value statistics and effects. A dynamic category is represented by flags for its most frequent values |
| `no_value_stats` | Replace a dynamic category's value statistics with flags and an effect row |
| `no_calendar` | Remove calendar features |
| `no_rolling` | Remove the rolling per-step family |
| `no_lags` | Remove the lag per-step family |
| `no_zeros` | Remove the zero-history per-step family |
| `no_step_families` | Remove rolling, lag and zero-history per-step families |
| `no_summary` | Mask all summary-vector entries and their flags |
| `per_target` | Forecast each target separately with siblings available |
| `no_siblings` | Use the per-target layout with sibling rows dropped |
| `gates_one` | Use gates fixed at one and disable pruning |
| `screening_alone` | Use screening-derived gates |
| `router_uniform` | Use uniform expert routing |

## Evidence and resume

`summaries.csv` uses fev's summary schema plus `task`, the adapter name, every CLI
inference setting, resolved weights revision and run identity. Training time is
zero and the trained-on-this-dataset flag is false (zero-shot use, not a new
claim about pretraining contamination). `inference_time_s` covers adapter
conversion and inference, excluding model loading, data loading and scoring.
Timestral batches windows together, so each NPZ receives an equal share of the
task's measured inference time; it is not a separately measured window latency.

Each task has `run.json` and `window_N.npz`: item IDs, target names, quantiles,
original-unit predictions `(items, targets, horizon, quantiles)`, identity and
time. Item and target order is fev's order. Failures, including fail-fast errors,
are recorded in `failures.json`; successful retries clear them. `--keep-going`
returns exit 1 if any selected task remains incomplete. Audit a yardstick run
with the task YAML it was run with: pass `--tasks` and the same `--names`
selection (comma-separated names or `@file`) to `fev_report.py`. For example:

```bash
python scripts/fev_report.py --run-dir results/chronos2 --model-name chronos2 \
  --tasks shifted.yaml --names @names.txt --out results/chronos2/report.md
```

Omit `--names` when the run includes every task in the YAML. The report verifies
identities, file hashes, coverage and recomputed fev scores. Yardstick audits do
not require synthetic training/bundle artifacts or a baseline.
`fev_wins.py standing` accepts these summaries directly. Isolated per-task runs
can be joined with `fev_merge_runs.py`, including names with `/`.

One output directory holds one yardstick configuration. Resume checks weights,
all switches, package versions, task definition, dataset fingerprint and window
hashes. Changed configurations are refused: choose a new output directory.
Missing/damaged window files trigger inference again; a missing/stale summary is
rebuilt from intact recorded windows. A model is never loaded for complete tasks.
Local weight directories are content-hashed; Hub aliases resolve to commit SHAs.
The default IDs above use the pinned SHAs even when `--revision` is omitted.

For v5, each task also has a `v5-report.json` and an
`evidence_window_N.npz` per forecast window. The report records counts of
items forecast alone, grid failures, missing time axes, short histories,
windows over the origin budget, mixed column kinds, and fallback target
rows. It also records the maximum distinct origins, worker count, and peak
resident memory of the process and its children in bytes.

Each evidence archive records the request's zero-based `item_position` in
the window, its `target`, and a boolean `joint` mask for requests covering
the item together. The `candidates` and `rules` arrays label the columns
and the fixed-rule codes. Each request-by-candidate matrix records
`status`, `rule`, `gate`, and screening `score`. Status codes 0 through 5
mean absent, kept, removed by a fixed rule, removed by a budget, zero gate
retained, and zero gate pruned. A rule index of −1 means no fixed rule. The
archive uses plain arrays and opens with `np.load(..., allow_pickle=False)`.
Both artifact types record `run_identity`. Both resume paths verify
`evidence_sha256` in `run.json` in addition to the forecast window digests.
The report supplies the v5 metadata when the summary is reconstructed.

## Cloud scoring jobs

`make_score_job.py` uses the same harness, saved forecast windows and completion
checks for Simons, Timestral, Chronos-2 and v5. Set `REPO_SHA` to the scoring code's
40-hex commit and `TASKS_SHA256` to the task YAML's SHA-256. For a development
subset, add `--names-uri` and `--names-sha256` to each command. The required
`--upstream-pin` retains the repository's Simons integration pin for all jobs;
reference jobs additionally record and verify their inference package pin in
`inputs.json` (Timestral commit `a534bdb`, Chronos `2.2.2`). Generation is offline:

```bash
python scripts/gcp/jobs/make_score_job.py --model simons --name simons-seed11 \
  --repo-sha "$REPO_SHA" --upstream-pin "$(cat third_party/simons/PINNED)" \
  --tasks-uri gs://bucket/tasks.yaml --tasks-sha256 "$TASKS_SHA256" \
  --checkpoint-uri gs://bucket/checkpoints/seed11/export --output /tmp/score-simons.sh
python scripts/gcp/jobs/make_score_job.py --model timestral --name timestral5m \
  --repo-sha "$REPO_SHA" --upstream-pin "$(cat third_party/simons/PINNED)" \
  --tasks-uri gs://bucket/tasks.yaml --tasks-sha256 "$TASKS_SHA256" \
  --hub-weights mistralai/Timestral-5M --hub-revision bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f \
  --output /tmp/score-timestral.sh
python scripts/gcp/jobs/make_score_job.py --model chronos2 --name chronos2 \
  --repo-sha "$REPO_SHA" --upstream-pin "$(cat third_party/simons/PINNED)" \
  --tasks-uri gs://bucket/tasks.yaml --tasks-sha256 "$TASKS_SHA256" \
  --hub-weights amazon/chronos-2 --hub-revision 29ec3766d36d6f73f0696f85560a422f50e8498c \
  --output /tmp/score-chronos2.sh
```

Reference jobs require Hub weights at a full commit SHA, refuse checkpoint URIs
and nondefault `--processing`, and save their native output in `raw` only.
`--covariates all|none` and `--max-context` apply to every model. To average these
runs with Simons, select `--processing raw|published|both`; references contribute
`raw` native forecasts in every mode, while Simons contributes that mode. Member tasks, covariates, context limits and scoring runtime must match. Model-specific
packages, native batching and Simons's dropped-covariate report may differ across
models, never between two runs of the same model. Runs of the same model must
share the full evaluation configuration and per-task evaluation identity,
including the inference implementation. The mean uses fixed, equal weights by default.

## Stage D static-attribute diagnostics

Run `python -m demand_forecast.bench.static_diagnostics --tasks TASKS.yaml --names @names.txt --out DIR`
for a subset, or omit `--names` for the entire YAML. The diagnostic reads fev's
past split for all measured values. For dated fev windows the date denotes the
last allowed observation. The builder origin is the first step after the split.
Diagnostics use scoring's `task.freq` and prove each item's time axis with the
same history timestamps, horizon timestamps, and task horizon as v5 scoring.
Only the timestamp column is read from fev's future split. The one-step
synthetic horizon is filled with missing targets and covariates.
The diagnostic therefore cannot see a future covariate cell that scoring would
refuse, such as a class code outside its declared labels.

Each `<task>.json` contains window summaries, static support, equal-static and
equal-target groups, dynamic-column relevance, and the static screening trace.
Numeric label support uses the builder's bins. For multilabel cells, support at
a given size means at least one label has that many members. Dependencies use
pairs of observed cells. Target equality compares ordered history values by
target name. `new_items` counts first appearances after the initial window.
`on_grid` includes items that fall back when the origin budget is exceeded.

`tasks.csv` has these columns in order.

```text
task,window,items,targets_per_item,history_min,history_median,history_max,short_items,on_grid,off_grid,items_alone,items_without_time_axis,grid_reasons,distinct_origins,over_origin_budget,new_items
```

`attributes.csv` has these columns in order. Support describes the full window.
`window_items` counts all items in the window. `panel_items` counts the items in the
dated part used to compute the score. `screen_kind`, removals and scores come from
that dated part at the given cutoff.
Cutoffs are integer offsets on that panel's grid. Lists and maps in CSV cells use JSON.

```text
task,window,cutoff,attribute,window_items,panel_items,kind,missing_share,labels,label_size_min,label_size_median,label_size_max,share_at_least_2,share_at_least_4,share_at_least_8,share_at_least_16,share_at_least_32,constant,identifier,unshared,determined_by,copied_from,screen_kind,removed,score,shuffled_scores
```

The diagnostic scores the three most frequent dated origins, favoring later
ones in ties, and takes the maximum relevance over the first 16 sorted targets.
`level.json` pools 20 permutations per kept attribute and cutoff in the first
and last windows when both `window_items` and `panel_items` are at least eight.
It uses the linear 95th percentile, rounds to nine decimals before taking the
ceiling to two decimals, and caps the level at 0.99. Fewer than 200 pooled scores
give 0.0. The report counts distinct tasks, task-windows and task-attributes in
the pool, and records the share of their real score rows strictly above the
level.

`scripts/gcp/jobs/make_diagnose_job.py` generates the CPU guest script offline.
The guest verifies the task YAML and optional names file by SHA-256 and publishes
the reports under `state/diagnose/<name>/`. It publishes `completed.json` last
and checks all recorded hashes on a retry.

## Averaging completed yardstick runs

`scripts/fev_ensemble.py` averages saved forecasts with fixed member weights (equal by default),
then recomputes fev scores from the averaged windows without loading models:

```bash
python scripts/fev_ensemble.py --members results/seed-1 results/seed-2 \
  --tasks shifted.yaml --names @names.txt --model-name average \
  --out-dir results/average
```

For a weighted mixed average, repeat `--member-weight W` once per `--members`
entry (or per `--member-uri` in the cloud generator), in the same order. Decimal
weights must be positive and sum exactly to one; for example `0.25`, `0.25`,
`0.5` combine two published-style Simons seeds with one native Timestral member.
All Simons members must share raw or published processing. Saved forecasts are
averaged directly, with no further clipping, snapping or processing: this is a
quantile average, not the quantiles of a probabilistic mixture. Members may have
seen different covariates (Simons drops past-only covariates); published mixed
outputs record each member's processing and input metadata. Explicit weights
cannot be combined with `--allow-identical-members`.

Supply at least two complete yardstick runs with matching tasks, dataset
fingerprints, quantiles, input ordering and evaluation settings. Each member
needs its root `yardstick.json`, matching per-task `run.json`, summary row and
saved windows with valid hashes, and no recorded task failure. Checkpoint
digests must differ unless `--allow-identical-members` is used for reproduction.
Members must have been scored by the same evaluation code, including adapters
and scorer: their recorded identities do not hash this repository's evaluation
code. Scoring and auditing still require access to the task data. Audit the
output with `fev_report.py` and the same task YAML and selection as above.

**Cloud route.** Generate an offline guest job with
`scripts/gcp/jobs/make_ensemble_job.py --name average --repo-sha <40-hex> --tasks-uri
gs://bucket/tasks.yaml --tasks-sha256 <64-hex> --member-uri
gs://bucket/state/score/seed-1 --member-uri gs://bucket/state/score/seed-2
--member-weight .25 --member-weight .75 --output average.sh`
(optionally `--names-uri` plus `--names-sha256`, and
`--processing raw|published|both`). It requires completed members with matching
evaluation code, task selection and evaluation settings, distinct names and
weight hashes, and verified manifests with no missing or unlisted mode files.
Before averaging, it scores every distinct member/source-folder pair against
itself with the job's scorer and checks every shared metric (NaN equals NaN; rtol `1e-9`, atol `1e-12`).
The job needs task data access but no GPU, model weights or Simons install.
Under `RUN_DIR/state/ensemble/<name>/`, it saves configuration, environment,
task inputs, member identity files, the hashed `resolved-inputs.json` folder mapping,
`parity/<member-index>-<source-folder>.json`, mode outputs/statuses
and a hashed `completed.json`. Downloaded member forecasts and parity outputs
stay in scratch. Restored jobs use the averaging script's own completion checks
to reuse partial results; a fully verified state exits before installation.

This differs from the older `scripts/fev_ensemble_runs.py`, which consumes the
training/bundle runner's saved format rather than yardstick artifacts.

## Reproduction limits

The published Chronos CSV records fev 0.8.0, and Timestral records 0.11.0.dev0;
our harness deliberately uses fev 0.10.0. Their CSVs do not record weight SHAs.
These defaults reproduce the public adapter settings; numerical parity remains
unverified until the preregistered five-task host check passes. Tests use only
synthetic local data and recording fakes; they download no model weights.

Simons tests use synthetic local data and a recording inference module; they
verify processing overrides, input isolation, oversized groups and resume
identity without downloading weights. Numerical parity with the pinned upstream
benchmark, forecast quality and GPU throughput require host runs with real
exported weights.
