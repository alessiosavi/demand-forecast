# Design: general probabilistic demand forecasting, measured on fev-bench (consolidated families + Timestral concepts) — rev 3

Date: 2026-10-30. Status: rev 3 after the user's steer ("consolidate, don't remove"; free-tier Colab); supersedes rev 2's
scope reductions. Resolutions of the builder check and Codex's design review remain in §12. Awaiting user approval.
Evidence: `2026-10-30-fev-bench-v2-analysis.md` (same directory) and its Codex reviews. Rev 3's changes are Claude's
and will be scrutinised again when Codex reviews the implementation plan.

## 1. Intent and why

Turn `demand-forecast` from a hackathon deliverable (Monday weeks, 16-week horizon, one store contract, point
forecasts) into a general library: anyone with a dataset of series, optional dynamic covariates and optional static
attributes can train a global neural model, tune it if they want, pick a family for their speed/quality trade-off, and
get calibrated quantile forecasts. Quality is measured on fev-bench (autogluon/fev), starting with the seven weekly
tasks `m5_1W, favorita_stores_1W, rossmann_1W, rohlik_sales_1W, walmart, hermes, hierarchical_sales_1W`, scored by SQL
over the nine quantiles 0.1–0.9. V2 is the reference backbone; every existing family (standard transformer, V2, TCN,
Mixer) is consolidated onto one data pipeline and one output contract; concepts from Mistral's Timestral (not its
weights, not its model) are ported where they raise quality, and may become a fifth family.

Why now: the analysis showed the V2 quantile head is never trained, no dynamic covariates exist, and every fev task
is rejected by the Monday checks before modelling starts. Published fev-bench rows put task-trained deep models at
0.67–0.72 of seasonal naive on these tasks and zero-shot foundation models at 0.46–0.50; the gap is largest where
covariates matter. The user's decisions (2026-10-30): train on Colab free tier via prompts; no Timestral weights or
model; delete only hackathon-specific contracts; keep and consolidate every working capability (model families,
Optuna tuning, clustering); V2 plus a possible new family.

## 2. Goals, non-goals, anti-goals

Goals
- One data path for user datasets (long CSV/Parquet) and fev tasks: ids, timestamps, target, known-future dynamic
  covariates, past-only dynamic covariates, static categorical and numeric attributes.
- One output contract for every family: sorted quantiles at configurable knots (default 0.1…0.9); point = median;
  trained with pinball. Families differ only in their temporal backbone and cost: `standard`, `v2`, `tcn`, `mixer`,
  later `dhf`. Export (ONNX/TorchScript/INT8) kept for the lightweight families, exporting the quantile output.
- Two ways to train: `train` with a family's standard configuration, or `tune` (Optuna) on inner validation.
- Clustering as an option: cluster label as a static feature of the global model (recommended), or one model per
  cluster (kept), or none (default).
- Missing data and variable-length histories are first-class: observation, padding and label masks everywhere.
- Frequency-agnostic date handling: a validated grid per panel; calendar features per frequency; weekly validated
  first, one daily task validated before `dhf`.
- Static categoricals stay learned embeddings; statics condition the temporal model (VSN + FiLM where the family has it).
- A fev harness that produces official `fev` summaries and a leaderboard, under a protocol that never tunes on the
  benchmark windows; a general `benchmark` command that compares families on any dataset under the same protocol.
- Training on Colab (free tier) driven by prompts; CPU inference; fast CPU test suite; a complete CPU-testable
  end-to-end path before anything runs on Colab.
- Provenance (run manifest) on every run; docs that describe only what exists.

Non-goals
- Timestral weights, zero-shot forecasting, fine-tuning any pretrained model.
- Daily/hourly benchmark coverage beyond one validation task in this iteration.
- The hackathon store contract (`product_id, origin, week_start, qty`), the AMD store/benchmark provenance protocol,
  Monday-anchored weeks, the fixed 16-step horizon and the catalogue-CSV coupling: deleted, not maintained (their
  general parts — provenance, baselines, per-horizon plots, family comparison — move into the new manifest, harness
  and `benchmark` command).
- Dynamic **categorical** covariates (string-valued dynamic columns) in this iteration: rejected with a message
  suggesting indicator encoding.
- Automatic holiday generation (supplied holiday covariates work as ordinary known covariates).

Anti-goals (a failure even if it "works")
- Any model selection, early stopping, tuning or ablation decision taken on the seven-task benchmark windows.
- A point-only code path, or quantiles that can cross, in any family.
- A hidden weekly or Monday assumption; a horizon literal anywhere; silent anchor conversion of timestamps.
- A fev run that drops eligible items, shortens horizons or re-orders predictions.
- Clipping a user's target because of an assumed domain rather than a declaration.
- A family that is "kept" but not on the shared contract (half-migrated code).

## 3. Constraints

- Python ≥ 3.10, PyTorch ≥ 2.x, pandas, numpy 2 allowed; `fev` pinned (extra `bench`); Optuna (extra `tune`);
  tsfresh optional (extra `cluster`); HF datasets via `fev`.
- Training runs on Colab **free tier**: T4 (16 GB VRAM), sessions of a few hours, idle disconnects, no persistence.
  Hence: one seed per run; `first_window` policy for the frozen run (per-window retraining only on the small tasks if
  time allows); mandatory checkpoint/resume to Drive; tuning budgets small and preregistered. The notebook clones the
  repo at an immutable commit SHA and installs `.[bench,tune]`. Every training request is a prompt the user pastes into
  VS Code (Colab extension + Claude plugin) that creates and runs a notebook on the Colab kernel (§7).
- No training on the developer Mac; unit tests and small-panel work run locally; m5_1W and the seven-task harness on
  Colab.
- Timestral code may be ported under Apache-2.0 with the licence text, retained notices and modification notices; no
  weights are downloaded.
- Trunk workflow: feature branches in worktrees, PRs, bot review, green CI; Codex implements, Claude reviews.
- Memory: m5_1W has 30,490 series; windows are never materialised; peak RAM/VRAM is measured before the frozen run.

## 4. Data model and configuration (decided)

### 4.1 Input schema and config
Long format (CSV/Parquet): one row per `(series key, timestamp)` with `target` and any dynamic columns. Optional static
table keyed by the same series key. fev tasks load from their ragged list-column format into the same in-memory
structure. Config declares column roles and the target's support; nothing is auto-detected except, optionally, the
frequency.

```yaml
data:
  source: long                       # long | fev
  path: data/sales.parquet           # long only
  id_columns: [product_id, store_id] # one or more; joined with "|" into a globally unique series id
  timestamp_column: timestamp
  target: sales
  target_support: nonnegative        # REQUIRED: nonnegative | real (no default; drives clamping at 0)
  known_dynamic: [price, promo]      # known over the horizon; numeric
  past_dynamic: [stock_out]          # context only; numeric
  static_path: data/products.csv     # optional; left-joined on id_columns; duplicate keys rejected
  static_categorical: [category, brand, store_id]   # numeric-coded ids allowed
  static_numeric: [size]
  freq: null                         # pandas offset alias with anchor (e.g. W-SUN, D, H, MS); null = infer (§4.2)
  missing_target: zero               # zero | nan: meaning of an absent row inside a series' span
  duplicates: error                  # error | sum: duplicate (series, timestamp) rows
  resample: null                     # optional explicit aggregation to a coarser offset (sum target, mean/last covariates)
forecast:
  horizon: 13
  quantiles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  max_context: 512                   # cap; batches pad to the longest context in the batch, not to the cap
  min_context: 1                     # observed steps required before an origin
clustering:
  mode: none                         # none | feature | per_model
  k: auto                            # auto (elbow/DBI on metafeatures, as today) | int
  features: fast                     # fast (in-repo metafeatures) | tsfresh (extra `cluster`)
model:
  family: v2                         # standard | v2 | tcn | mixer | dhf (phase 6)
  ...                                # family-specific knobs, as today
training: {...}                      # includes refit, max_steps, early stopping
tuning: {...}                        # Optuna study settings, search space per family, budget
```

Identity and joins: the series id is the `|`-joined tuple of `id_columns` (validated unique; null keys rejected).
Duplicate `(series, timestamp)` rows are an error unless `duplicates: sum`. Statics are left-joined; a series without a
static row gets the `missing` category and NaN numerics (+ indicator); duplicate static keys are an error.

### 4.2 Grid and frequency
Resolution order: (1) `freq` from config or from the fev task; (2) otherwise panel-level inference: pandas inference per
series, accepted only when every inferable series agrees; short or gapped series inherit the panel offset; if nothing
is inferable the loader stops and asks for `freq`. Every observed timestamp is validated against the grid: an off-grid
timestamp is an error that names the row and suggests `resample`; pinning never converts anchors silently. Each
series is placed on the grid from its first to its last row. Absent rows inside the span follow `missing_target`
(`zero`: no row = no sales; `nan`: unobserved). Positions before a series' first row are always NaN, as are explicit
NaN values. Dynamic covariates at absent rows are NaN + mask. Original timestamps are kept for output and for fev's
positional contract. Seasonal period `m` for scaling follows fev's convention per frequency (weekly 1, daily 7, hourly
24, monthly 12), configurable.

### 4.3 Future inputs and origins
`origin` is the first forecast timestamp; the context is strictly before it. Rows at or after the origin supply
known-future covariates; their target and past-only values are ignored (logged). Missing known-future values are
NaN + mask. For fev, `origin = cutoff + step` and `future_data` supplies the known covariates.

### 4.4 Preprocessing (stateless where possible)
- Target: no fitted scaler. Per sample: location = masked context median; scale = mean absolute `m`-lag difference over
  the **whole available past before the origin** (SQL's denominator), computed unfloored. A sample is loss-eligible
  only when its scale is positive and finite and it has at least one observed label; the floor applies only inside
  the arithmetic of eligible samples. A batch with no eligible sample contributes zero loss and is counted.
- Dynamic covariates: standardised per series over the available past with masks; NaN → 0 + mask channel; {0,1}-valued
  columns untouched. Calendar features per frequency (week-of-year, month; for sub-weekly data also day-of-week,
  day-of-month, day-of-year, hour) are generated as known covariates.
- Statics: categoricals through the vocabulary encoder with `unknown=0` and `missing` as its own slot (fit on training
  ids only); numerics standardised with NaN → 0 + indicator; numeric-coded categoricals are declared in config.
- Clustering (optional): metafeatures per series computed on training data before the inner-validation origin (`fast`:
  level, dispersion, zero share, intermittency, autocorrelation at `m`, trend slope; or tsfresh minimal); KMeans with
  `k` chosen as today (elbow + Davies–Bouldin) or fixed; labels saved in the bundle; new series at inference are
  assigned by nearest centroid. `feature` mode adds the label as a static categorical; `per_model` mode trains one
  model per cluster (same contract) and routes by label.
- No series filtering, no outlier clipping, no rounding on any path. Robust clipping remains an explicit flag, off by
  default, applied to training inputs only.

### 4.5 Sampling
Index-based sampler over `(series, origin)`: context = up to `max_context` steps before the origin, left-padded with
a padding mask and batched to the longest context in the batch (the `mixer` family pads to its fixed length); horizon
= `horizon` steps with a label mask. Training origins: every origin with at least `min_context` observed steps before
it and at least one observed label in the horizon, restricted to data before the inner-validation origin.
Uniform-series sampling per epoch. Nothing is materialised.

## 5. Models (decided)

### 5.0 Shared skeleton (consolidation)
`ForecastModel` base: static encoder (id embedding, per-column categorical embeddings, numeric projection, optional
VSN), input projection of the encoder/decoder channels, the sample normalisation frame, masks, the monotone quantile
head, and the fallbacks. A family supplies only its temporal backbone (`encode(context) -> memory`, `decode(memory,
known_future) -> hidden (B, H, D)`) and its conditioning hook (FiLM, additive, or none). Every family therefore gets
covariates, masks, variable context and quantiles from the same code, and the registry maps `model.family` to a
backbone. This replaces four copies of embedding/projection/head code.

### 5.1 Output contract
`forward(batch) -> quantiles (B, H, Q)` sorted along Q in the sample's affine normalised frame; the predictor
denormalises with the sample's location/scale and clamps at 0 only when `target_support: nonnegative`.
`point = quantiles[:, :, index(0.5)]`; 0.5 must be among the knots. Head: Timestral's monotone construction (median
free, `cumsum(width_factor · Δp · softplus(raw))`). Fallbacks (deterministic, finite, logged): no observed context →
all quantiles 0 when `nonnegative`, else the training panel's observed median (0 if none); zero/undefined scale →
empirical context quantiles.

### 5.2 Families
- `v2` (reference): pre-norm encoder/decoder with RoPE, VSN + FiLM + static enrichment kept. Changes: encoder channels
  `[target_norm, observed_mask, calendar…, known_cov_past…, past_cov…, cov_masks…]`, decoder channels
  `[calendar…, known_cov_future…, cov_masks…]`; padding mask in encoder self-attention and every decoder
  cross-attention layer; decoder self-attention **bidirectional** (known-future only; a later promotion must influence
  an earlier step); explicit horizon, no interpolation; monotone head replaces both old heads; patch embedding and
  series decomposition unavailable until mask-aware acceptance passes.
- `standard`: `nn.Transformer` encoder/decoder (GELU, optional Pre-LN, RoPE, FiLM, stochastic depth as today) on the
  shared channels; `src_key_padding_mask` and `memory_key_padding_mask` carry the padding mask; decoder self-attention
  bidirectional; per-step head over the decoder output (the "last position predicts all" head is dropped).
- `tcn` (lightweight): dilated causal TCN over the encoder channels with masked inputs; decoder = FiLM-conditioned
  known-future projection plus the context summary, per-step head; ONNX/TorchScript/INT8 export of the quantile
  output retained and tested.
- `mixer` (lightweight): MLP-Mixer over a fixed `max_context` with padding mask applied as zeroing + mask channel;
  known-future covariates enter through a decoder projection (today it ignores `future_time`); export retained.
- `dhf` (phase 6, own reviewed specification): causal patch transformer with Decoupled Horizon Forecasting ported
  from Timestral concepts (`[values, missing]` patches, per-patch causal normalisation, RMSNorm/SwiGLU/gated attention,
  partial RoPE, forecast-query tokens with two-stream masks, rows with roles and firewalls, variate attention), statics
  via FiLM/AdaLN. Its normalisation frame and multi-block training differ from the shared affine frame; the
  specification must define the frame conversion or a declared surrogate loss, and parity tests.
- Ensemble: the point-only `EnsembleWrapper` is replaced by a quantile-preserving weighted ensemble (non-negative
  weights, sorted output preserved), selectable in `benchmark` and `forecast`.

### 5.3 Loss and metrics
Training loss: doubled pinball over the knots in the affine frame, reduced over observed label cells of eligible
samples. Selection and reporting: exact SQL in original units with fev's denominator and exclusion rule, implemented
in-repo and cross-checked against `fev` on fixtures (missing, zero-scale, undefined-scale items, empty eligible
batches); also MASE, WAPE, WQL, per-knot pinball, 80% interval coverage/width. VSN weights logged as diagnostics only.

## 6. Training, tuning and evaluation protocol (decided)

- Trainer: AdamW, cosine schedule with warm-up, gradient clipping, AMP with auto-detected dtype, seeds, `max_steps`
  budget (scaled by panel size within declared bounds), early stopping on inner-validation SQL with a relative
  tolerance, best-checkpoint restore. Periodic checkpoints include model, optimizer, scheduler, AMP scaler, RNG and
  sampler progress so a run resumes exactly. One trainer serves every family and `per_model` clustering (one
  optimizer per model, parameters of that model only).
- Inner validation: for a given origin, the last `horizon` steps before it form the validation block; learned
  artefacts (vocabularies, covariate stats, cluster centroids) are fitted on data before that block; configurations and
  checkpoints are selected there. Per-item eligibility: items without enough history are excluded from the validation
  set only.
- Refit semantics (`training.refit: none | full`, default `none`): `none` keeps the selected checkpoint and extends its
  context through the validation block at forecast time; `full` refits on all data before the origin with the selected
  configuration and step count, refitting learned artefacts (no warm start).
- Configuration selection, two modes: **`train`** uses the family's standard configuration (fixed in the protocol
  document) plus early stopping; **`tune`** runs Optuna (rewired: objective = inner-validation SQL, per-family search
  spaces as today's `SearchSpace` minus hackathon presets, pruning on step checkpoints, study storage on Drive,
  preregistered budget in trials/time) and writes the best configuration, which `train` then uses. Tuning never sees
  benchmark windows. On the free tier, tuning budgets are small (single-digit trials per task) and optional; the
  context-length candidates (`2·H`, `max(64, 2·H)`, 256, 512) are part of the search space, with 512 the cap.
- Short-history exception (frozen): a window's inner validation runs when at least one item has
  `min_train_history` (default `horizon + 8`) observed steps before the validation origin and the validation SQL has at
  least one eligible contribution; the validation set is the eligible items only. Otherwise the window uses the
  family's standard configuration and a fixed step budget, fits on all available past without local early stopping,
  and falls back to §5.1's fallback where training is impossible. Official horizons and item coverage are never
  reduced; exclusions reported.
- Inference bundle (versioned directory): weights (all cluster models when `per_model`), architecture config, family,
  quantile knots, target support, column roles, frequency offset and `m`, vocabularies, fitted static preprocessing,
  cluster centroids/labels, training cutoff, selection vs final-model flag, package version and git SHA. `forecast
  --origin` loads the bundle, refuses origins earlier than the training cutoff, and reloads in a fresh CPU process with
  bit-identical outputs (test). Unseen ids/categories map to `unknown`; unseen series get the nearest cluster (tests).
- fev harness (`demand-forecast fev run --tasks weekly|<names> --family v2 --policy first_window|per_window --resume`):
  per task/window, build the Panel from `past_data`/`future_data`, fit (or reuse under `first_window`: same bundle,
  learned preprocessing untouched, deterministic context statistics updated, embedding ids never remapped), forecast,
  assert exact coverage and order, emit `{"predictions": q0.5, "0.1": …, "0.9": …}` per item, save completed-window
  predictions, `task.evaluation_summary(..., trained_on_this_dataset=False)`, collect summaries,
  `fev.leaderboard(..., baseline_model="seasonal_naive")` with explicit failure handling and bootstrap CIs. The
  seasonal-naive baseline is the official fev statsforecast wrapper vendored into the repo (Apache-2.0, pinned
  `statsforecast`). Parity with published rows is required only once task definitions, fingerprints, forecast counts and
  implementations are comparable, within a stated tolerance.
- General `benchmark` command (consolidated from `core/benchmark.py`): runs several families (and the ensemble) on any
  dataset or fev task under the same protocol, equal budgets, and produces the comparison table and plots (SQL/WQL by
  family and by horizon, accuracy vs cost, learning curves, example series) — the general parts of today's hackathon
  benchmark, without the store contract.
- Policy: `first_window` for development and for the free-tier frozen run; `per_window` on the small tasks
  (hierarchical, rossmann, favorita) if time allows; frozen before any benchmark score is viewed and stated in the
  report. The frozen run reports every task and window, failures, fallbacks, training and inference time, seed, hardware.
- Leakage and invariance tests: withheld targets, the horizon part of past-only covariates, padding content and padding
  length must not change valid outputs; a synthetic known-future covariate must show a later input changing an earlier
  forecast step; every family passes the same suite.
- User path: `train --config` (standard configuration; writes bundle + manifest), `tune --config` (Optuna on inner
  validation; writes best configuration), `forecast --config --origin <timestamp>` (long-format
  `series_id, origin, timestamp, q0.1 … q0.9, point`), `benchmark --config --families ...`, `export` for `tcn`/`mixer`.

## 7. Colab workflow (decided; free tier)

For each training need, Claude writes a prompt file the user pastes into VS Code (Colab extension + Claude plugin). It
instructs the assistant to create, run and save a Jupyter notebook on the Colab kernel that: runs a GPU preflight
(device, VRAM, dtype); mounts Drive; clones the repo at an immutable commit SHA and installs `.[bench,tune]` with pinned
dependencies; stages the fev dataset cache on Drive (revision/fingerprint recorded); runs the exact CLI commands
(tasks, family, policy, seed, budget) with `--resume` and checkpoints on Drive; and copies back summaries CSV, run
manifests (hardware, seed, SHA, dataset fingerprints), best bundles and logs, with a one-paragraph report format. An
interruption/resume test precedes the first full run. Free-tier sizing: one seed; `first_window`; m5_1W and tuning
runs split across sessions by task; the budget of every run is stated in steps/time before it starts. `docs/colab.md`
is the first phase-0 deliverable. Unit tests and small-panel profiling run on the Mac; anything touching m5_1W or the
full seven-task harness runs on Colab.

## 8. Migration inventory (decided; each item owned by a PR)

| Legacy item | Fate | PR |
|---|---|---|
| `core/forecast_run.py` | delete the store contract/latency protocol; provenance → run manifest (§6) | 3 |
| `core/benchmark.py`, `cli.benchmark` | **consolidate** into the general `benchmark` command (§6) | 4 |
| `core/tuning.py`, Optuna | **keep, rewire**: objective inner-validation SQL, new trainer/sampler, per-family spaces, hackathon preset removed | 4 |
| `core/pipeline.py` (Monday guards, catalogue grid, scalers, cluster routing) | replaced by Panel + trainer + bundle; cluster routing kept in the new `ModelWrapper` | 1–3 |
| `data/preprocessor.py` | `catalogue_weekly_grid`, `complete_week_bounds`, `filter_skus`, `ScalerManager` deleted; `resample_series` → explicit `resample`/`duplicates` loader options | 1 |
| `data/dataset.py`, `data/dataloader.py` | replaced by Panel + index sampler; torchtnt round-robin → in-repo round-robin for `per_model` | 1 |
| `utils/time_features.py` | replaced by frequency-aware calendar covariates | 1 |
| `utils/clustering.py`, `extract_metafeatures` | **keep, rewire** to the Panel; `fast` metafeatures added; tsfresh optional | 4 |
| `models/wrapper.py` (`ModelWrapper`, `EnsembleWrapper`) | **keep, rewire**: routing by cluster label; ensemble becomes quantile-preserving | 3–4 |
| `models/transformer.py` (`standard`) | **upgrade** onto the shared skeleton | 4 |
| `models/lightweight.py` (`tcn`, `mixer`, export, quantization) | **upgrade** onto the shared skeleton; export of quantile output tested | 4 |
| `models/transformer_v2.py` (`v2`) | upgrade (§5.2) | 3 |
| `models/components.py`, `models/losses.py` | keep; add monotone head, doubled pinball, masks; remove point-only losses that no path uses | 2–3 |
| `inference/predictor.py`, `inference/confidence.py` | `Predictor` **rewritten** as the bundle loader (same name); Gaussian/residual interval helpers retired once real quantiles exist (unused by any path) | 3 |
| `core/evaluator.py`, `utils/metrics.py` | rewritten around quantile metrics (SQL, WQL, MASE, WAPE, coverage); torchmetrics kept where still used | 2 |
| CLI | `train`, `tune`, `forecast`, `benchmark`, `export`, `fev run`, `generate-*`; `evaluate`/`predict`/`preprocess` folded into `benchmark`/`forecast`/`train` | 3–4 |
| `configs/hackathon*.yaml`, `configs/benchmark-w26.yaml` | delete; replaced by `configs/example-long.yaml`, per-family standard configs, per-task fev profiles | 3 |
| `synthetic/generator.py` | keep; generalise: explicit zero-demand rows, optional missingness injector, catalogue-free defaults | 1 |
| `debug_notebook.py`, `debug_quick.py` | delete (superseded by `docs/colab.md` and the CLI) | 3 |
| package exports | rewritten; package-import, CLI `--help` and synthetic round-trip checks in CI | 3 |
| dependencies | `fev`+`statsforecast` under `bench`, `optuna` under `tune`, `tsfresh` under `cluster`; drop torchtnt; lift `numpy<2` if tsfresh allows (else keep the pin under `cluster` only) | 1–4 |
| version | 2.0.0 | 3 |

## 9. Phases (one PR each; acceptance evidence per PR)

0. **Protocol, profiling, harness skeleton, Colab template.** Pin `fev`; `docs/colab.md`; profile all seven datasets
   (small panels locally, m5 via Colab prompt); vendored seasonal-naive baseline through the harness on all seven
   (Colab); protocol document (standard configuration per family, tuning budgets, lookback candidates, policy,
   exception, fallback) frozen. Evidence: profile table, baseline table with comparability notes, summaries CSV.
1. **Data core.** Panel, roles config, grid/frequency resolution, calendar covariates, index sampler with masks, scale
   and eligibility, loss, exact SQL + metrics with fixtures cross-checked against `fev`, fallbacks, generalised
   generator. Evidence: unit tests for W-SUN/W-WED/D/MS, off-grid and gapped timestamps, composite ids, duplicates,
   static joins; metric parity on fixtures; sampler memory test on an m5-sized synthetic panel.
2. **Shared skeleton + metrics.** `ForecastModel` base, monotone head, masks, evaluator/metrics rewrite. Evidence:
   head monotonicity and gradient tests; a toy backbone trains on the synthetic panel.
3. **`v2` + trainer + bundle + CLI + hackathon deletions.** §5.2 `v2`, §6 trainer, inner validation, refit, bundle,
   `train`/`forecast`, `Predictor` rewrite, migration items marked PR 3. Evidence: shape tests for horizons
   8/13/39/52, sorted quantiles, invariance tests, fresh-process reload parity, unseen-id test, CPU fixtures for
   Rossmann-like `29/H13`, Rohlik-like `8/H8`, mixed eligibility and no eligible contribution, a tiny offline fev task
   end to end, V2 CPU latency recorded, CI green.
4. **Other families, tuning, clustering, benchmark, export.** `standard`, `tcn`, `mixer` on the skeleton; `tune`
   rewired; clustering modes; quantile ensemble; general `benchmark`; `export`. Evidence: every family passes the
   same contract and invariance suite; export round-trip equals PyTorch output; tuning study runs on the synthetic
   panel; `per_model` and `feature` clustering train and reload; benchmark table on the synthetic panel.
5. **First fev runs (Colab) and daily validation.** Interruption/resume test; smoke on hierarchical_sales_1W, earliest
   Rossmann window, Rohlik's shortest contexts; optional small tuning on inner validation; frozen seven-task run with
   `v2` (other families on the small tasks as budget allows); `rossmann_1D` end to end; README/ARCHITECTURE/
   CONFIGURATION/API rewritten. Evidence: summaries, leaderboard with CIs, manifests, measured memory/time.
6. **`dhf` family** under its own reviewed specification; inner-validation comparison; frozen run if it wins.

## 10. Left to the builder

Module layout and names; tensor layouts; exact calendar feature set per frequency; standard configuration per family
(ranges: d_model 64–256, 2–6 layers, lr 1e-4–3e-3; TCN channels 32–128); `fast` metafeature definitions; search-space
bounds per family; test fixture generators; the Colab prompt wording; dataset cache layout on Drive; storage of
per-task fev role tables (hand-written from the analysis §1 table, numeric-coded ids declared categorical).

## 11. Calls made by Claude (user can override)

- Clustering default `none`; `feature` recommended when used; `per_model` kept. Rationale in the reply of 2026-10-30.
- Every family upgraded onto one skeleton rather than kept on its own code; the "last position predicts all" head and
  the point-only losses go because no path on the new contract can use them.
- `target_support` required (no clamping by default); `missing_target: zero` default; `training.refit: none` default.
- Lookback: 512 cap, context length tuned on inner validation (`train` uses the family standard).
- Free tier: one seed; `first_window` frozen run; tuning optional and small.
- Vendored statsforecast seasonal naive; daily validation task `rossmann_1D` before `dhf`.
- `forecast --origin` semantics as §4.3; results CSV and manifests committed; bundles on Drive.
- Interval helpers in `inference/confidence.py` retired once real quantiles exist (they are unused by any path today).

## 12. Review resolutions

Builder check (5 questions): refit semantics → §6; configuration search → §6 `train`/`tune`; Colab budget → §3, §7;
missing rows → §4.2; Mac vs Colab → §7. Codex design review (11 findings, all accepted): zero-scale eligibility →
§4.4/§5.1; clipping → `target_support` required; frequency resolution → §4.2; identity/joins/future inputs → §4.1,
§4.3; inference bundle and reuse → §6; `dhf` frame → §5.2 own specification, phase 6; Colab reproducibility → §3, §7;
lookback as candidate → §6; migration inventory and generator zeros → §8; provenance/aggregation preserved → §6, §8;
CPU-testable path → phase 3 evidence, mixed-panel exception in §6, daily/docs before `dhf`.
User steer (2026-10-30, second message): rev 2's scope reductions reversed — all families consolidated (§5.0, §5.2),
Optuna kept and rewired (§6), clustering kept as modes (§4.4, §6), `benchmark` consolidated (§6), free-tier sizing
(§3, §7).
