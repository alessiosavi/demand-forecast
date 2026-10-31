# demand-forecast → fev-bench weekly tasks: verification, gaps, and V2-centred redesign (rev 3)

Date: 2026-10-30. Author: Claude (orchestrator). Status: rev 3 after two Codex review rounds (resolutions in §7); analysis stage converged; awaiting the user's decisions in §8 before any planning or implementation.
Repos: `/opt/SP/Workspace/Python/demand-forecast` (main @ f60778a), `/opt/SP/Workspace/Python/timestral` (a534bdb, inference-only, Apache-2.0).
Scratch evidence (same directory): `fev/tasks.yaml`, `fev/metrics.py`, `fev/task.py`, `fev/utils.py`, `fev/analysis.py`,
`fev/datasets_README.md`, `fev/results/*.csv`, `fevdata/*/train.parquet` (3 weekly panels), `hf/Timestral-*.config.json`,
`hf/Timestral-209M.README.md`, `hf/Timestral-209M.LICENSE`, `spike/timestral_grad_check.py`, `spike/inspect_fev.py`,
`codex-review-output.md` (round 1).

## 0. Objective (user, 2026-10-30, two messages)

1. Verify two suspected limitations (weekly-only pipeline; mostly point forecasts), find other issues, and work out how to
   remove them.
2. Judge whether Mistral's Timestral architecture can be replicated here while keeping categorical features as learned
   embeddings; make the data pipeline state of the art (missing features, feature selection/importance).
3. Sharpened target: **fev-bench weekly tasks** `m5_1W, favorita_stores_1W, rossmann_1W, rohlik_sales_1W, walmart, hermes,
   hierarchical_sales_1W`. Output the **9 quantiles 0.1–0.9**. Price and promotions are **time-varying known-future
   inputs** on M5/Rossmann/Favorita; today price is static and only calendar features go forward. Horizons are 8–52 weeks,
   so `n_out` and the lookback window vary per task. **Build on V2** (`AdvancedDemandForecastModelV2`, `model_type: advanced`).

## 1. What fev-bench asks for (facts from `tasks.yaml`, `metrics.py`, `task.py`, `utils.py`, dataset card)

| task | horizon | windows | season. | series | known-future dynamic | past-only dynamic | static |
|---|---|---|---|---|---|---|---|
| m5_1W | 13 | 1 | 1 | 30,490 | sell_price, event_{Cultural,National,Religious,Sporting}, snap_{CA,TX,WI} | – | item_id, dept_id, cat_id, store_id, state_id |
| favorita_stores_1W | 13 | 10 | 1 | 1,579 | onpromotion | oil_price | store_nbr, family, city, state, type, cluster |
| rossmann_1W | 13 | 8 | 1 | 1,115 | Open, SchoolHoliday, StateHoliday, Promo | Customers | Store, StoreType, Assortment, CompetitionDistance, CompetitionOpenSince{Month,Year}, Promo2, Promo2Since{Week,Year}, PromoInterval |
| rohlik_sales_1W | 8 | 1 (cutoff 2023-12-15, min_context 8) | 1 | 5,243 (3,942 forecast in published runs after fev's length filter) | total_orders, sell_price_main, type_0..6_discount, holiday, shops_closed, winter_school_holidays, school_holidays | availability | product_unique_id, name, L1..L4_category_name_en, warehouse |
| walmart | 39 | 1 | 1 | 2,936 | MarkDown1..5, Unemployment, Fuel_Price, Temperature, CPI, IsHoliday | – | Store, Dept, Type, Size |
| hermes | 52 | 1 | 1 | 10,000 | external | – | country, category |
| hierarchical_sales_1W | 13 | 10 | 1 | 118 | – | – | – |

Protocol facts that constrain the adapter:
- **Scoring:** `eval_metric: SQL` with `quantile_levels: [0.1 … 0.9]`; extras MASE, WAPE(ε=1), WQL(ε=1).
  `SQL = mean over items, horizon, quantiles of 2·|(y−q)·(1[y≤q]−τ)| / seasonal_error_item`, with
  `seasonal_error_item = mean |y_t − y_{t−m}|` over the item's **whole past** and `m = seasonality = 1` on all weekly tasks
  (`metrics.py:461-490, 650-654`). Items with zero/undefined seasonal error are excluded from the metric, **not** from
  the required predictions. Scores are computed per window and averaged over windows (`task.py:995-1013`).
  The leaderboard skill score is `1 − geometric mean over tasks of (model SQL / seasonal_naive SQL)` with win rate and
  optional bootstrap CIs (`analysis.py:228-380`; CIs must be requested explicitly).
- **Predictions are consumed positionally** (`task.py:145-179`): one row per item in `test_data` order, each a dict with
  `"predictions"` (point; used by MASE/WAPE) and one key per quantile level as a string (`"0.1"` … `"0.9"`; used by
  SQL/WQL), every list exactly `horizon` long and finite. Item ids do not re-align a reordered batch.
- **Item eligibility** is fev's, not ours (`utils.py:435-453`): an item is kept when it has at least `min_context_length`
  **positions** (default 1; Rohlik 8) before the cutoff and `horizon` positions after it. A timestamp cutoff includes
  observations at or before it. We must forecast every kept item, including sparse and short ones.
- **Rolling windows:** `num_windows` cutoffs stepping by `horizon`; per window `past_data` (all columns up to the cutoff)
  and `future_data` (id, timestamps, known covariates over the horizon). Earlier windows' evaluation targets are
  legitimately part of later windows' history.
- `trained_on_this_dataset` describes **pretraining-corpus overlap** (fev `model.py`); published task-trained rows
  (TFT, DeepAR, PatchTST) carry `False`. Task-specific fitting on permitted history does not set it. The official
  AutoGluon wrapper supports first-window and per-window fitting and defaults to first-window.
- **Reproducibility:** the published result files span fev 0.7.0–0.11.0.dev0; pin `fev`, task definitions, dataset
  fingerprints and compare forecast counts before claiming exact reproduction. Published summaries carry
  `training_time_s` / `inference_time_s`; the fev-bench paper (arXiv 2509.26468, App. B) documents per-window deep
  refits, preprocessing search for boosted models and SeasonalNaive fallbacks when training data is insufficient.

### 1.1 Data facts from the parquet files (inspected 2026-10-30; independently re-read by Codex)

| dataset | series | length (weeks) | span | week anchor | target zero share | notes |
|---|---|---|---|---|---|---|
| hierarchical_sales_1W | 118 | 260 (all equal) | 2014-01-08 → 2018-12-26 | **Wednesday** | 1.2% | no covariates, no NaN |
| rossmann_1W | 1,115 | 133 (all equal) | 2013-01-13 → 2015-07-26 | **Sunday** | 3.2% | weekly covariates are day-aggregates: `Open` ∈ [0,1] fractions, `Promo` ∈ {0, 0.71}, `SchoolHoliday` ∈ {0..7}, `StateHoliday` ∈ {0,1,2}; statics 32–49% NaN (`CompetitionOpenSince*`, `Promo2Since*`, `PromoInterval`); `Store` is a float id |
| favorita_stores_1W | 1,579 | 240 (all equal) | 2013-01-13 → 2017-08-13 | **Sunday** | 17.6% | `onpromotion` is a weekly count (0–4,730); `oil_price` past-only; `store_nbr`/`cluster` float ids |
| walmart | 2,936 | not inspected | – | Friday (per Codex; not verified by me) | – | – |
| m5_1W, rohlik_sales_1W, hermes | – | not inspected | – | – | – | expected ragged/late starts (m5), fev-filtered items (rohlik) |

Consequences:
- **Week anchors are not Monday.** Every Monday check (`cutoffs`, `forecast_batches`, `validate_forecast_contract`,
  `complete_week_bounds`) rejects these tasks. The grid step and anchor must come from the data's timestamps.
  (Full sub-weekly frequency support can wait; honouring the given weekly timestamps cannot.)
- **Rolling windows eat the history.** rossmann: 8 × 13 = 104 of 133 weeks are evaluation, so the first cutoff leaves
  **29 weeks** of context; holding 13 more back for inner validation leaves 16 before the validation origin. favorita:
  110 weeks at the first cutoff; hierarchical: 130. The lookback must shrink to what exists (left-padding + mask), the
  sampler must accept incomplete windows, and training must pool across series (global model).
- **Weekly "indicators" are fractional** after aggregation; rules keyed on {0,1} values do not fire.
- **Statics can be NaN and numeric-coded categoricals** (`Store`, `store_nbr`, `cluster`): `filter_skus` drops rows with
  null string statics (`preprocessor.py:185-188`) and `CategoricalEncoder._log_values` raises on NaN continuous values
  (`feature_engineering.py:40-44`); both need "missing category" / "NaN → 0 + indicator" policies and a config switch to
  declare a numeric column categorical.

### 1.2 Reference bar: published fev-bench results on the seven tasks (SQL; `results/*.csv`; recomputed by Codex)

| model (class) | m5_1W | favorita | rossmann | rohlik | walmart | hermes | hierarch. | gm(SQL/naive) |
|---|---|---|---|---|---|---|---|---|
| timesfm-3 (zero-shot) | 0.878 | 2.253 | 0.265 | 1.113 | 0.688 | 0.599 | 0.615 | 0.457 |
| tabpfn-ts-3 (zero-shot) | 0.904 | 2.054 | 0.256 | 1.159 | 0.696 | 0.619 | 0.625 | 0.457 |
| timestral-209m (zero-shot) | 0.913 | 2.207 | 0.274 | 1.183 | 0.648 | 0.598 | 0.615 | 0.461 |
| chronos-2 (zero-shot) | 0.900 | 2.024 | 0.308 | 1.274 | 0.648 | 0.609 | 0.616 | 0.468 |
| timestral-28m / 5m (zero-shot) | 0.887 / 0.888 | 2.771 / 2.125 | 0.291 / 0.403 | 1.242 / 1.294 | 0.694 / 0.724 | 0.608 / 0.638 | 0.618 / 0.625 | 0.487 / 0.502 |
| tft (AutoGluon, task-trained) | 0.876 | 2.216 | 0.827 | 1.928 | 1.063 | 0.940 | 0.673 | 0.667 |
| stat_ensemble (statsforecast) | 0.936 | 2.220 | 0.501 | 1.646 | 1.217 | 1.416 | 0.746 | 0.672 |
| patchtst / deepar (AutoGluon) | 0.920 / 1.028 | 2.241 / 2.327 | 0.872 / 0.887 | 1.928 / 1.928 | 1.738 / 1.123 | 0.866 / 1.155 | 0.660 / 0.693 | 0.717 / 0.723 |
| catboost / lightgbm (mlforecast) | 1.13 | 2.4–2.5 | 0.67 | 1.9 | 1.2 | 1.2 | 0.78 | 0.73–0.74 |
| seasonal_naive (baseline) | 1.356 | 2.494 | 0.899 | 1.928 | 2.034 | 2.146 | 1.386 | 1.000 |

Reading, stated as references rather than feasibility bounds: the task-trained deep rows sit at 0.67–0.72 of naive, the
zero-shot foundation rows at 0.46–0.50, and the gap is largest where covariates matter (rossmann, walmart). The identical
1.928 of TFT/DeepAR/PatchTST on rohlik equals seasonal naive and is consistent with the paper's SeasonalNaive fallback, so
that row says nothing about architecture. These numbers do not isolate architecture from training corpus, optimisation
budget or fallback behaviour. The official SeasonalNaive is season-1 with probabilistic intervals (statsforecast); the
repo's point seasonal-52 baseline is not a substitute.

## 2. Verification of the two claims (against the code)

### Claim 1 — "Weekly only, Monday origins, fixed 26-week window and 16-week horizon"

**Confirmed, with two corrections.** Weekly/Monday arithmetic is spread across the pipeline. The lookback is configurable
(`timeseries.window`, default 52; hackathon configs set 26). The 16-step horizon is enforced only on the `forecast`
command and benchmark contract path; `train`/`predict` accept any `n_out`.

| Where | Hard-coded |
|---|---|
| `core/pipeline.py:124-137` (`cutoffs`) | `Timedelta(weeks=1)`, `weeks=n_out`, `weeks=validation_periods`; Monday check (catalogue path only) |
| `pipeline.py:265-270` (`create_datasets`) | `cutoff − 1 week`, `origin + (n_out−1) weeks`, `origins + 7·(n_out−1) days` |
| `pipeline.py:363-367, 403, 429-430, 481-483` | unconditional Monday check, `freq="W-MON"`, `week_start` export, `predict` requires W-MON |
| `core/forecast_run.py:46-53, 311, 317, 448-449` | Monday + W-MON contract, `n_out != 16` raises, `reshape(len(ids), 16)` |
| `core/benchmark.py:98-121, 131, 502` | baselines `W-MON`, `weeks=8`, seasonal naive `weeks=52`, horizon = days//7 |
| `data/preprocessor.py:128, 135, 241-270` | `W-MON` special cases; `complete_week_bounds`, `catalogue_weekly_grid` Monday-only |
| `utils/time_features.py:64-76, 88-100` | ISO week-of-year (÷53) and month only; future offset is `window·7` days |
| `data/dataset.py:66, 120`; `pipeline.py:298-299` | time-feature width literals: `x[:, :, 2:6]`, `reshape(count, n_out, 4)`, dims 5/4 |

`DataConfig.resample_period` accepts `D`, `M`, `Y` (`settings.py:14-16`) while window/cutoff arithmetic stays in weeks,
so non-weekly configs either fail at a later check or mis-slice. What blocks fev is the Monday anchor, the fixed
horizon on the contract path, and the `week_start` export, not the window size.

### Claim 2 — "Mostly point forecasts; quantiles only in V2, only 3; confidence.py uses past errors"

**Confirmed, and the V2 quantile head is never trained.**
- `models/transformer_v2.py:397-402`: quantile head, `num_quantiles=3`, default `[0.1, 0.5, 0.9]`, independent outputs
  (no non-crossing constraint); `get_prediction_intervals` (`:584-600`) hard-codes three.
- `core/trainer.py:128` sets `_criterion = nn.MSELoss()`; the `point_forecast()` call in `_forward_step` and in
  `core/evaluator.py:109` drops every head but `prediction`. `QuantileLoss`/`CombinedForecastLoss`
  (`models/losses.py:76, 112`) are used only in tests. Codex's independent backward pass left all four quantile-head
  parameter tensors without gradients. The repo documents it (`forecast_run.py:467-468`, `benchmark.py:308`).
- `inference/confidence.py`: Gaussian `z·std` bands (`:8-61`, per-horizon std possible) and an empirical residual-quantile
  helper (`:89-111`); the only caller, `inference/predictor.py`, is legacy (no `cont`, cannot run catalogue models). No
  probabilistic metric exists (`utils/metrics.py`, `benchmark.py:140`).
- Precisely: fev rejects missing quantile keys; a point model that repeats its point across the 9 symmetric knots scores
  SQL = MASE (Codex verified numerically). The deficiency is the absence of trained **conditional** quantiles, not an
  extra penalty.

## 3. Other findings that bear on fev (ranked)

1. **No dynamic covariates.** Inputs are `qty_scaled` + 4 calendar features; statics read from the last input step
   (`dataset.py:112-114`). fev weekly tasks carry up to 13 known-future and 1 past-only dynamic columns.
2. **No missingness concept.** Zero-filled grid (`preprocessor.py:267`), NaN rejected at inference (`pipeline.py:376`),
   no observation/padding mask reaches V2 (`transformer_v2.py:495-517`), patching and decomposition are not
   mask-aware (`components.py:331-339, 363-378`), no observed-target mask in any loss.
3. **Destructive preprocessing on the fev path:** sparse-item filtering (`pipeline.py:157-166`,
   `preprocessor.py:173-188`), target clipping applied to the whole grid including evaluation rows
   (`pipeline.py:191-197, 222-235`), rounding to 2 decimals on export. fev items, zeros and raw labels must be preserved.
4. **Scaling is a fitted `StandardScaler` per SKU or cluster on training rows** (`pipeline.py:199-203, 243-247`): a
   fitted global state is fragile for 30k series and for 29-week contexts. Mask-aware per-window normalisation inside the
   model is leakage-safe for V2 (one observed context per example) and needs no fitted state.
5. **Loss/metric mismatch:** MSE on standardised targets vs SQL (pinball in original units scaled by the item's seasonal
   error).
6. **Window materialisation does not scale:** `create_time_series_data` (`dataset.py:81-125`) loops per series and offset
   and requires complete `window + n_out` windows (`:106`). m5_1W has 30,490 series; the exact count of valid offsets and
   the memory depend on lengths, lookback, horizon and validation policy and must be **measured**, but materialising every
   window is not viable. Index-based access over ragged per-series arrays is required.
7. **V2's decoder self-attention is causal** (`transformer_v2.py:526-528`) although decoder inputs carry only known
   information (future calendar; no target feedback). An early forecast step therefore cannot see a promotion scheduled
   later in the horizon. Codex verified on a tiny V2 that perturbing the last future token changes step 1 by exactly 0.
   The mask serves no leakage purpose and should be removed (or a separate known-future encoder added).
8. **Statics assume log1p-able non-negative continuous columns** (`feature_engineering.py:40-44`); fev statics include
   arbitrary numerics and NaNs, plus high-cardinality ids (item_id 3,049; product_unique_id 5,243; Store 1,115).
9. **Single validation block** (`validation_periods == n_out`, `pipeline.py:133-134`): rolling tasks need an explicit
   per-window retrain or first-window reuse policy, with embeddings/vocabularies kept compatible on reuse.
10. **Output-length hacks:** V2 interpolates predictions to `n_out` when decoder length differs
    (`transformer_v2.py:546-552`); horizons 8/13/39/52 must be explicit.
11. **Training defaults are not a quantile recipe:** `num_epochs 10`, `learning_rate 1e-5`, `early_stop_min_delta 1.0`
    (`settings.py:145-150`); an absolute improvement threshold of 1.0 is meaningless for SQL values near 0.3–2.5.
12. Minor: `init_metrics` instantiates every torchmetrics regression metric per evaluation; RoPE slicing
    `cos[:tgt_len]/sin[:src_len]` is safe only because cross-attention disables RoPE.
    (Withdrawn after Codex review: a claimed AdamW weight-decay bug on inactive clusters — `zero_grad(set_to_none=True)`
    makes Adam skip parameters without gradients, verified on torch 2.15; and "tsfresh metafeatures are diagnostic only" —
    `pipeline.py:175-188` uses them for KMeans when `k > 1`.)

## 4. Timestral: what it is, what transfers to V2, what does not

Sources: `timestral/src/timestral/*.py`, model card, blog (2026-10-15), **technical report arXiv 2610.37585 (released
2026-10-27)**, weight LICENSE.

Architecture (in the released code):
- Patch 32; each patch embedded from `[values, missing]` by a residual MLP; learned `forecast_query` tokens (**DHF**:
  `[Enc ; Dec]`, context causal, forecast tokens read all valid context plus their causal block; ≤32 patches per block;
  longer horizons decoded block by block with the median fed back).
- **Causal per-patch normalisation** (Welford mean/std at each patch end, `asinh`), different rules for target/past-only
  rows and for known-future rows (whole-row stats; 0/1 rows skip asinh) (`model.py:122-151`); horizon patches re-framed
  by the "advancing refine frame" (`forecaster.py:458-483`).
- Pre-norm RMSNorm, SwiGLU, sigmoid-gated attention, partial RoPE on time; `time`/`time_variate` layers.
- **Rows with roles** and firewalls (`masks.py:104-118`): covariate rows never read targets; known-future rows are
  bidirectional in time and may not read **any** causal row (targets or past-only covariates).
- Categorical covariates **target-encoded per series** (univariate-target case; multivariate targets get raw codes)
  (`forecaster.py:246-276`). **No static embeddings**; the official fev Timestral wrapper ignores statics.
- **Monotone quantile head**: 21 knots, median free, `cumsum(width_factor·Δp·softplus(raw))`, width factor 3.0.
- Training (blog + report): masked pinball in normalised space over the 21 quantiles; synthetic-only pre-training then
  real+synthetic; forecast-block training with role assignment. The report describes the recipe; **no training
  implementation is released**. Fine-tuning is offered commercially (Forge).
- Test-time: clampsym (average with the forecast of the negated context with every covariate negated, quantile index
  reversed; clip ≥0 when the context is ≥0) and a quantile cap.
- Sizes: 5M = 8 layers d192, 28M = 12 d384, 123M = 20 d640, 209M = 24 d768.

**Licensing (decision-relevant):** code is Apache-2.0 (attribution + modification notices when ported). Weights are a
"Modified MIT" whose clause 2 denies all rights if the user's company **or employer** had global consolidated monthly
revenue above **US$20M** in the preceding month, extended to derivatives, fine-tunes and combined works. Code reuse and
weight reuse are separate decisions; weight eligibility must be confirmed before any zero-shot or fine-tuning route.

**Gradient check (narrow conclusion, reproduced by Codex):** the released module is a plain `nn.Module`; tiny random
config, `predict_block`, pinball loss, backward → finite non-zero gradients on 47/47 parameter tensors on both the plain
and covariate paths; sorted quantiles in train mode. Not covered: checkpoint loading, optimiser steps, adapter training,
the public `forecast()` (inference mode), advancing-frame parity, or benchmark quality. A pretrained-adapter route needs a
loaded-checkpoint fine-tune/parity spike before it is implementation-ready.

**What V2 already has that Timestral lacks and the user wants kept:** SKU + per-column `nn.Embedding`s, continuous static
projection, static VSN, FiLM on encoder/decoder inputs, static-enrichment GRN (`transformer_v2.py:301-371, 418-457`).

**What transfers into V2, as hypotheses to ablate on inner validation (not a package):**
1. Pinball objective over the task knots and a monotone 9-knot head (drop-in for V2's `quantile_head`).
2. `[values, missing]` channels and left-padding; known-future covariates as decoder inputs; past-only covariates as
   encoder channels.
3. Normalisation: start with explicit mask-aware per-window affine scaling; ablate asinh and per-patch causal statistics
   separately (per-patch Welford is not needed for V2's fixed-context setup).
4. Non-negative clipping enabled from **task semantics declared in config** (sales, orders and quantities have
   non-negative support by definition), never from profiling evaluation labels; clampsym's sign symmetrisation and the
   range cap only with inner-validation evidence (negated prices/flags are out of distribution for a retail-only V2; it
   doubles inference).
5. Rows-with-roles / variate attention: a measured alternative later, not dismissed — shared processing across covariates
   and explicit availability boundaries can matter within one task. DHF rewrite deferred (with patch 32 these horizons are
   1–2 forecast patches; DHF training is a different sampling/masking scheme).

## 5. Design (V2-centred, rev 2)

### 5.1 Evaluation protocol (frozen before the reported run)
- **Inner validation** per task and window: the last `horizon` steps before the window's cutoff form the validation
  origin; every learned preprocessing (vocabularies, scalers if any, clipping if any) is fitted on data before that origin;
  configurations, checkpoints and early stopping are selected on **exact fev SQL in original units** (same denominator:
  the item's whole available past, same exclusion rule) computed by our own implementation and cross-checked against
  `fev` on a fixture. Then refit under a declared policy up to the cutoff and forecast the official window.
- **Freeze** the configuration set and policies before the seven-task run; the run reports every task and window, with
  failures and fallbacks. The seven-task scoreboard is never used to choose between variants; if it ever is, it is
  disclosed as benchmark tuning.
- **Short-history exception (frozen before implementation):** official prediction eligibility (fev's
  `min_context_length`) is not inner-validation eligibility. Rossmann's first window has 29 weeks, so a 13-week inner
  holdout leaves training contexts of 1–3 observed steps (executable with left-padding, but seven items have zero scale at
  2–3 steps); Rohlik's 8-week eligible contexts leave nothing after an 8-step holdout. Rule: inner validation runs only
  where at least `min_train_history` (configurable; proposed `horizon + lookback_floor`) observations precede the
  validation origin and the validation SQL has at least one eligible contribution; otherwise the window uses a
  configuration and training-step budget fixed in advance (selected on the other windows/tasks), fits on all available
  past without local early stopping, and, where training remains impossible, emits the declared finite fallback
  forecast. Official horizons and item coverage are never reduced; validation exclusions are reported separately.
- **Leakage tests** (deterministic invariance, not shuffling): changing withheld evaluation targets, or the horizon part of
  past-only covariates, must leave predictions bit-identical; changing the content or length of padding must leave valid
  outputs bit-identical; a synthetic task with an informative known-future covariate must show that a **later**
  known-future input changes an **earlier** forecast step.
- **Reproducibility:** pin `fev`, record task definitions, dataset fingerprints, model revision, seeds, policy, training
  and inference time per task; use `fev.leaderboard(..., baseline_model="seasonal_naive")` with explicit failure handling
  and bootstrap resampling.

### 5.2 fev adapter and harness
- `demand_forecast/data/fev_source.py`: `fev.Task` windows → ragged per-series arrays (target, each dynamic covariate,
  timestamps) plus a static table; **no long-frame expansion**. Item order preserved exactly as `test_data`; exact
  coverage asserted per window (every kept item, `horizon` steps, finite values, 9 quantile keys, `predictions == q0.5`).
  Raw numeric forecasts (no rounding).
- `demand_forecast/core/fev_run.py` + CLI `fev-bench --tasks weekly|<names> --model advanced --policy per_window|first_window`:
  per task/window: fit preprocessing → train (or reuse) → forecast → `task.evaluation_summary(..., trained_on_this_dataset=False)`
  → summaries CSV → leaderboard. Baselines through fev's own wrappers (statsforecast SeasonalNaive; Timestral wrapper
  only if the weight licence allows).
- Per-task config: `horizon` from the task; lookback `min(k·horizon, available)` with configurable `k` and floor, left
  padded; week anchor derived from timestamps.

### 5.3 V2 model changes (minimum set for a first controlled result)
- Explicit `horizon`; no interpolation. 9 trained quantiles via a monotone head; `prediction` = 0.5 quantile; no separate
  point head.
- Decoder self-attention **bidirectional** over known-future inputs (remove the causal mask; the decoder has no target
  feedback), or a separate known-future encoder; keep cross-attention to the encoder.
- Inputs: encoder `[target, observed_mask, calendar, known_cov_past…, past_cov…, cov_masks…]`; decoder
  `[calendar, known_cov_future…, cov_masks…]`; channel counts from config. Padding mask into encoder attention; patching
  and decomposition disabled until mask-aware (or restricted to observed regions).
- Masks: observation (target NaN), padding (absent positions; distinct from present-but-missing), forecast-label (loss
  only on observed future targets, reduced over observed cells as fev does). The padding mask is applied in encoder
  self-attention **and in every decoder cross-attention layer** (V2's `PreNormDecoderLayer` cross-attention currently
  takes unmasked memory, `transformer_v2.py:159-167`; Codex measured a 0.32 output change from padding content without a
  memory mask). Non-finite inputs filled before arithmetic. Defined behaviour: a context with no observations at all gets
  a deterministic finite fallback (all quantiles = 0 for non-negative targets, logged); a context with observations but
  undefined scale gets its empirical context quantiles; all-keys-masked attention is prevented by the fallback path.
- Normalisation inside the model, mask-aware: location = context median or mean (cancels in pinball); scale = the
  item's mean absolute seasonal difference over the **whole available past before the origin** (SQL's own denominator;
  the cropped lookback is not the same quantity, and using it breaks the equivalence: Codex measured SQL 0.25 vs
  surrogate 1.0 on `[0,10,11,12]` with a 2-step crop). With that scale, an affine map and fev's factor of two,
  normalised pinball equals SQL exactly for items with a positive finite scale; items with zero/undefined scale are
  excluded from the training loss as fev excludes them from the metric (an ε floor is only numerical safety and is
  documented as a deviation). Checkpoint selection still uses exact original-unit SQL (§5.1). std and asinh are
  ablations. Covariates standardised per series over the context; indicators untouched.
- Statics: categoricals via `CategoricalEncoder` (unknown=0, NaN → "missing" category) + embeddings; numeric statics
  standardised with NaN → 0 + indicator; config declares numeric-coded categoricals. First-window reuse keeps
  vocabularies fixed (unseen ids → unknown slot).
- Sampling: index-based `(series, offset)` over ragged arrays with uniform-series sampling first (length-proportional
  sampling changes the effective weighting). Measure peak RAM/VRAM and batches/s on real m5_1W before choosing instances.
- Training recipe: AdamW with a tuned learning rate (not 1e-5), cosine schedule, early stopping on inner-validation SQL
  with a relative tolerance; seeds recorded. VSN selection weights reported as diagnostics only; predictive importance
  (grouped feature ablation on inner validation) deferred until the forecasting path is correct.

### 5.4 Retraining policy
Both `first_window` and `per_window` are legitimate (official AutoGluon wrapper default: first window). Default to
`first_window` for development (fewer fits; 32 task-windows × candidates otherwise), `per_window` for the frozen run if
budget allows; report the policy and times. On reuse, update the context only; never refit scalers or remap embeddings.

### 5.5 Frequency generalisation
Derive step and anchor from data now (needed for Sunday/Wednesday/Friday weeks). Sub-weekly calendar features and
daily/hourly tasks later.

## 6. Plan (one PR each) with acceptance evidence

0. **Freeze protocol, profile, baselines.** Pin fev and the StatsForecast wrapper/dependency versions; profile all seven
   datasets (lengths, anchors, NaN, negatives, zero-scale items, eligible counts, unseen categories); run SeasonalNaive
   through fev's wrapper. Parity with the published SQL is required **only** once task definitions, dataset fingerprints,
   forecast counts and the baseline implementation are shown compatible, within a stated tolerance; otherwise document the
   discrepancy and record our pinned baseline as the reference. Freeze the lookback rule, retraining policy,
   short-history exception and fallback. Evidence: profile table, baseline table with comparability notes, protocol doc.
1. **Thin V2 fev adapter.** Ragged arrays + indexed sampler, padding/observation/label masks (encoder and decoder
   cross-attention), explicit horizons 8/13/39/52, monotone 9-quantile head trained with doubled pinball, full-past
   scale normalisation, bidirectional known-future decoder, exact-SQL inner validation, checkpoint reload parity.
   Evidence: coverage/order assertions, finite ordered quantiles, `predictions == q0.5`, quantile-head gradients,
   tiny-batch learning test, metric fixtures with missing/zero/undefined scales cross-checked against `fev`, finite
   all-missing fallback test, padding-content and padding-length invariance, target/past-covariate invariance,
   memory/runtime log on real m5_1W.
2. **Hard-case smoke tests + typed dynamic covariates.** hierarchical_sales_1W, the **earliest rossmann window**
   (29-week context, exercising the short-history exception), rohlik's shortest eligible contexts; then
   known-future/past-only covariates with masks and full horizon access. Evidence: the exception executes as specified;
   synthetic test that a later known-future input changes an earlier forecast step; inner-validation SQL vs SeasonalNaive
   on those cases as quality evidence (not a correctness gate).
3. **Tune on inner validation, then the frozen seven-task run.** Small configuration set, seeds, chosen policy; report
   all tasks/windows, failures, fallbacks, times; leaderboard with CIs. Evidence: summaries CSV, run manifest.
4. **Optional additions, each via inner validation only:** asinh, per-patch statistics, clampsym/range cap, variate
   attention over covariate rows, gated attention/SwiGLU/RMSNorm, importance report, m5_1W scale-out.
Separate track (user decision, licence-gated): **pretrained Timestral adapter** — loaded-checkpoint fine-tune/parity
spike first; V2's static-embedding/FiLM path as the adapter; its own normalisation and masks preserved.

## 7. Codex round-1 resolution (verdict was "changes needed")

| # | Finding | Disposition | Change |
|---|---|---|---|
| 1 | Acceptance criteria tuned on the benchmark test windows; shuffle leakage test invalid | **Accept (blocking)** | §5.1 inner validation, freeze, disclosure; invariance tests + synthetic covariate test |
| 2 | fev contract imprecise (positional, eligibility, averaging, pinning, rounding) | **Accept** | §1 protocol facts (verified in `task.py:145-179`, `utils.py:416-453`, `task.py:995-1013`); §5.2 |
| 3 | Pinball in asinh space ≠ SQL; selection must use exact SQL | **Accept** | §5.1 selection on exact SQL; §5.3 scale = seasonal error so the surrogate matches; asinh an ablation |
| 4 | Target masks and destructive preprocessing missing | **Accept** | §3.2–3.3, §5.3 masks and policies; filtering/clipping/rounding off on the fev path |
| 5 | Short histories, index access, policy must precede; anchors contradict §5.4 | **Accept** | §6 reordered (now phase 1–2); §5.5 anchors now; m5 memory to be measured |
| 6 | Causal decoder blocks later known covariates | **Accept** (verified from `transformer_v2.py:526-528`; decoder has no target feedback) | §3.7, §5.3 bidirectional decoder |
| 7 | Welford/asinh/clampsym are hypotheses; clipping needs established support | **Accept with nuance** | §4 list; clipping on by task profile where targets are non-negative |
| 8 | Timestral facts: tech report released; firewalls; statics; gradient scope | **Accept** (report verified: arXiv 2610.37585) | §4 rewritten |
| 9 | Retraining policy and leakage-flag semantics | **Accept** | §1, §5.4 |
| 10 | Leaderboard ratios are references, not bounds; backbone is a separate track | **Accept** | §1.2 wording; §6 separate track; open question 6 |
| 11 | Weight licence has a US$20M revenue clause | **Accept** (LICENSE clause 2 verified) | §4 licensing; open question 2 |
| 12 | AdamW decay claim and tsfresh claim false | **Accept** (AdamW verified on torch 2.15) | removed from §3 |
| 13 | Wording/citations (fixed window, "silently", confidence.py, point-model penalty) | **Accept** | §2 reworded |
| 14 | VSN weights are not importance | **Accept** | §5.3 diagnostics only; importance deferred |

No finding rejected. Material revisions: §5.1 protocol, §5.3 normalisation/decoder/masks, §6 ordering, §4 licensing.

### Codex round 2 (verdict "changes needed"; five specification findings, all accepted; #6, #8–#14 closed)

| # | Finding | Disposition | Change |
|---|---|---|---|
| R2-1 | Scale must be the full-past denominator, factor of two, zero/undefined scales, observed-cell reduction, all-missing fallback | **Accept** (anticipated independently) | §5.3 normalisation and masks bullets |
| R2-2 | Short-history inner validation (Rossmann 16 weeks, Rohlik 0) needs a defined exception | **Accept** | §5.1 short-history exception |
| R2-3 | Padding mask missing from decoder cross-attention | **Accept** | §5.3 masks; phase 1 invariance evidence |
| R2-4 | Clipping must come from task semantics, not label profiling | **Accept** | §4 item 4 |
| R2-5 | Parity gate needs comparability prerequisites and tolerance; phases 1–2 need edge-case evidence | **Accept** | §6 phases 0–2 |

Codex found no further design blocker within scope. Remaining items are the user's decisions (§8) and execution
evidence (profiling the four uninspected panels, m5_1W memory/runtime measurement).

## 8. Open questions for the user

1. **Compute and policy:** EC2 instance/budget per task; `first_window` vs `per_window` for the frozen run; seeds.
2. **Timestral weights:** is the company (or employer) under the US$20M monthly-revenue threshold? Without that, no
   zero-shot Timestral baseline row and no fine-tuning route; code porting under Apache-2.0 stays possible.
3. **Lookback rule:** `k·horizon` with cap/floor (proposed) or explicit per-task values?
4. **Hackathon contract:** keep `forecast`/benchmark (W-MON, 16, `week_start`) as a profile, or retire?
5. **Clustering (`k`)**: start with `k=1` for scope; keep the wrapper for later comparison?
6. **Backbone track:** V2 from scratch only, or also the pretrained Timestral adapter track (licence permitting)?
