# Frozen fev protocol

This is the preregistered protocol for the new core path. Exact values come from the
[implementation plan's Global Constraints](superpowers/plans/2026-10-31-fev-core-v2.md),
with model and evaluation semantics from the [design, §§5–7](superpowers/specs/2026-10-30-fev-bench-v2-design.md).
The plan takes precedence where the earlier design differs. PR 0 supplies profiling,
seasonal-naive baselines and this specification; it does not implement the new trainer.
**Phase 5 owns all real-data profiling, baselines and model runs on Colab.** Local
verification uses synthetic fixtures only. No real-data results are claimed here.

## Tasks and scores

Use the seven definitions pinned to `fev==0.10.0` in
[`fev_bench_weekly.yaml`](../demand_forecast/bench/tasks/fev_bench_weekly.yaml).
Supplementary roles and required target support come from
[`profiles.yaml`](../demand_forecast/bench/tasks/profiles.yaml); all seven declare
`target_support: nonnegative`.

| Task | Horizon H | Windows | Seasonality m |
| --- | ---: | ---: | ---: |
| m5_1W | 13 | 1 | 1 |
| favorita_stores_1W | 13 | 10 | 1 |
| rossmann_1W | 13 | 8 | 1 |
| rohlik_sales_1W | 8 | 1 | 1 |
| walmart | 39 | 1 | 1 |
| hermes | 52 | 1 | 1 |
| hierarchical_sales_1W | 13 | 10 | 1 |

Keep every official cutoff, horizon, item and window. Rohlik uses the pinned initial
cutoff `2023-12-15` and `min_context_length: 8`; do not replace these with inferred
splits. Do not filter series, clip outliers, round forecasts or silently change anchors.

Quantiles are `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`, strictly increasing and
inside (0, 1). Point predictions are the median. Train with doubled pinball in the
sample's affine frame, over observed labels of loss-eligible samples. Select on exact
SQL in original units, with fev's seasonal denominator and exclusion rule. Report SQL,
MASE, WAPE and WQL (the latter two with `epsilon=1.0`), plus per-knot pinball and 80%
interval coverage/width when the new evaluator exists. Excluded contributions and
fallbacks must be counted; exclusions never remove required forecasts.

The reference is the vendored fev seasonal-naive implementation with
`statsforecast==2.0.1`. Official summaries use `trained_on_this_dataset=False`.
The leaderboard uses `baseline_model="seasonal_naive"` and 1,000 bootstrap resamples.
Published-score comparability requires matching task definitions, dataset fingerprints,
forecast counts and implementations, with a stated numerical tolerance. Matching a
published number alone is not evidence of parity.

## Standard v2 configuration and budget

These values are frozen for the new `ModelConfig`; they are not a configuration for
the legacy CLI still present in PR 0.

```yaml
seed: 0
model:
  family: v2
  d_model: 64
  nhead: 4
  num_encoder_layers: 2
  num_decoder_layers: 2
  dim_feedforward: 256
  dropout: 0.1
  cat_emb_dim: 16
  use_patch_embedding: false
  use_decomposition: false
forecast:
  max_context: 512
  min_context: 1
  quantiles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
training:
  refit: none
  base_steps: 3000
  eval_every: 200
  patience: 5
  rel_tol: 0.002
  learning_rate: 0.001
  weight_decay: 0.01
  warmup_steps: 100
  grad_clip: 1.0
  batch_size: 128
```

Horizon comes from the task. `use_patch_embedding=True` and `use_decomposition=True`
are rejected for `v2`. The monotone head uses `width_factor=2.0`; arithmetic uses
`min_scale=1e-6` only after eligibility is established. Defaults are
`missing_target: zero`, `duplicates: error`, and no clustering. Target support is
required; it has no default. Seasonal periods are weekly 1, daily 7, hourly 24,
monthly 12, otherwise 1, with the task's explicit seasonality used for fev scoring.

The frozen budget function is:

```text
resolve_budget(base_steps, n_series, eval_every) -> (max_steps, eval_every)
max_steps = int(clip(base_steps * n_series / 2000,
                     min(500, base_steps), 4 * base_steps))
eval_every = min(eval_every, max(max_steps // 5, 1))
```

`n_series = len(plan.fit_ids)`, the number of items with an observation in the permitted
fit split. Examples: `(3000, 118, 200) -> (500, 100)`;
`(3000, 30490, 200) -> (12000, 200)`; `(60, 3, 200) -> (60, 12)`.
Use AdamW with linear warm-up over 100 steps, then cosine decay to zero at `max_steps`.
Evaluate every resolved `eval_every` and at the final step when validation is available.
Restore the absolute best checkpoint. Early stopping fires after five evaluations
without improvement exceeding the relative tolerance 0.002.

One epoch draws exactly one origin per series: a permutation determined by `(seed,
epoch)`, then a uniform draw among that series' eligible origins. `Settings.seed=0`
is the single seed, set for Python, numpy and torch before model construction. Use
AMP on CUDA (bf16 when natively supported, i.e. compute capability ≥ 8.0;
otherwise fp16 with GradScaler), disabled on CPU.

## Selection, short histories and fallbacks

**Benchmark windows are never used for configuration selection, early stopping,
tuning or ablation decisions.** Full-history profile statistics are diagnostic only;
they include held-out values and must not guide those choices either.

For forecast origin `origin` (the first future timestamp), inner validation is the last
H steps strictly before it: `[origin - H * step, origin)`. Learned vocabularies, static
preprocessing and any clustering are fitted before that block. An item contributes to
validation only if it has at least `min_train_history = H + 8` observed values before
the validation origin, positive finite seasonal-naive scale over that past, and at
least one observed validation label. Use only these eligible items for validation.

If no item qualifies, the **short-history exception** uses the standard configuration
and the resolved fixed budget, fits on all available past before `origin`, and performs
no local early stopping or validation selection. Keep the final state. This also
covers long but constant or entirely missing histories with no eligible validation
contribution. Neither horizons nor official item coverage may be reduced.

When the permitted training split yields no origin (for example, singleton-only or
all-missing panels), return `FitResult(steps=0, stop_reason="no_training_data")` with
the initial model state and still save the inference bundle. The training-split
observed median supplies `fallback_level`, or 0.0 if no observation exists.

Forecast fallbacks are deterministic, finite and logged:

- No observed context: all quantiles are zero for `nonnegative`; for `real`, use the
  training split's observed median, or zero if none exists.
- Zero or undefined seasonal scale: empirical quantiles of the observed context.
- Otherwise: denormalize the sorted quantiles using the context median and full-past
  seasonal scale. Clamp at zero only for explicitly declared `nonnegative` support.

Standard training uses `max_context=512`. Optional future tuning may consider
`[2H, max(64, 2H), 256, 512]`, deduplicated with a cap of 512. Selection remains on
inner validation only; no tuning is authorized by prompt 00. Phase 4 owns tuning,
`refit: full` and `per_window`; preregister any later tuning trial/time budget before
running it. Other families and `dhf` need their own reviewed configurations.

The frozen policy is **`first_window`**, one seed (0). Train/select once on the first
window's available past; reuse that bundle and learned preprocessing for later
windows. Context extends with newly available observations, and deterministic context
statistics update without remapping embedding ids. `refit: none` keeps the selected
checkpoint and extends its context through inner validation at inference time.

**Item-specific origins.** fev integer cutoffs are positions: every item is split at its own
position, so items that end on different timestamps have different first future timestamps
(walmart: 11 origins, 37 of 2,936 items ending early). The window origin `O` for the run
identity, the inner-validation block and training is the latest item origin: the validation
block is `[O − H·Δ, O)` and the inclusive training cutoff is `O − (H + 1)·Δ` (except under
the short-history exception); items ending inside the block contribute a partial validation
horizon and items ending before it contribute none. Every item is forecast at its own first
future timestamp with its own past as context. The bundle's earliest permissible origin is the
earliest item origin, a global floor. Early-ending items are scored on periods that other
items' available past covers; this equals the information set fev gives every model (each
item's past is truncated at its own cutoff) and is not an out-of-time test for those items.
Shared-origin windows are the special case with one origin. Prompt 00's profile reports
`distinct_origins_first_window` per task.

**Quick validation of the pipeline, NOT the protocol: scores from this config must never inform a configuration decision.**
`configs/fev-v2-quick.yaml` caps every task at 300 optimizer steps (`max_steps_cap`),
evaluates every 50 steps with patience 2 and scores at most 1,024 seeded validation
items (`val_items_max`). It exists to exercise the pipeline on the seven tasks within
one free Colab session. Its summaries carry `model_name demand-forecast-v2-quick` and
a different run identity; its benchmark scores are reported only as a pipeline check
and must never inform a configuration, early-stopping, tuning or ablation decision.
The frozen configuration above is unchanged by these fields (both `null`).

### Amendment 2026-11-01 (non-finite training inputs)

The user's approved decisions, quoted verbatim: D1 **"D1a: mask it off (Recommended)"**;
D2 **"Yes (Recommended)"**; D3 **"Yes (Recommended)"**;
D4 **"Only if data shows it (Recommended)"**.

1. A dynamic covariate with no finite value before a sample's origin but at least one
   finite value in the sample's slice is unobserved for that sample in both context and
   future (mask off, value 0); an entirely missing slice counts zero; the count of such
   sample-columns is recorded (`cold_covariate_columns`). Future values never shape a
   sample's context transform, as before.
2. Dynamic covariate values are bounded to ±1e3 after either the indicator pass-through
   or the standardization; clipped entries are counted (`clipped_covariate_values`);
   non-finite source values (NaN, ±inf) are missing.
3. Training stops after 20 consecutive non-finite losses with `stop_reason` set to
   `nonfinite_losses`, after checkpointing; such a task has no bundle and no summary
   row and the run is failed.
4. A task with `fit_result.nonfinite_losses > 0` fails the report's completeness check.
   When that field is absent, the gate uses `timings.cumulative.nonfinite_losses`,
   which remains required. The three new covariate counters are optional for legacy
   evidence and display as `unknown` when absent.
5. `stop_reason` values: `max_steps`, `early_stop`, `exception_budget`, `no_training_data`,
   `interrupted`, `nonfinite_losses`. The standard configuration is unchanged;
   identities are unchanged. Batch tensors are unchanged for every sample without a
   cold column, without a floored or clipped value and without non-finite source values.
6. Standardization scale for dynamic covariates is `max(std, 1e-3 × max|past|)` when
   std > 0; floored sample-columns are counted (`floored_covariate_columns`). An exactly
   constant past retains the existing `std or 1.0` rule.

**D4 (target-frame floor): not triggered.** The 2026-11-01 diagnostic found rohlik
target scale minima 0.50–1.0 and maxima 983–1407; walmart's target scale was not small,
and no target-frame overflow was found. The failed quick run had 150/150 non-finite
losses on walmart and 99/150 on rohlik_sales_1W. The CPU diagnostic with fp16 emulation
found walmart's cold MarkDown columns in 128/128 rows, with raw maxima
97,741 / 20,834 / 34,348, and rohlik decoder covariates of
4.95e6 / 8.28e6 / 2.03e7 from tiny nonzero past standard deviations (sell_price_main
std < 1e-6 in 54 series, smallest 1.71e-8; type_4_discount smallest 1.63e-9).
Both tasks were fp32-finite and fp16-NaN. The two-task rerun in prompt 02 §5 is the
required T4 fp16 acceptance check for the 1e3 covariate bound; these diagnostic
observations do not claim that acceptance has passed.

## Persistence and execution evidence

The trainer persists model, optimizer, scheduler, scaler, RNG states, sampler
position, step, history, best state/score/step and patience counter. Resume restores RNG
states after construction without reseeding. Checkpoints, bundles and saved predictions
carry a SHA-256 run identity over canonical JSON of settings, seed, dataset fingerprint,
package version, resolved origin, and the task definition (`name`, `horizon`,
`num_windows`, `initial_cutoff`, `window_step_size`, `min_context_length`, `seasonality`,
`quantile_levels`). Mismatches raise `ValueError("run identity mismatch")`; only missing
artifacts are recomputed. A session interruption saves `checkpoints/latest.pt`, returns
`stop_reason="interrupted"`, and raises `InterruptedRun` before writing a bundle.

These trainer guarantees are implemented, including the `train` and `fev run` CLI
paths since PR #23. PR 0's profiler (`--resume`) and baseline (automatic resume)
save after each completed task, keyed by task name (and
model for baselines). They do not validate training run identities or checkpoint a
partially completed task. Use a new output directory whenever code, task definitions,
role profiles or data change. Preserve the dataset cache between resumed sessions.

Phase 5 must record the immutable code SHA, dependency lock, dataset revisions and
fingerprints, task definitions, seed, policy, budgets, hardware/dtype, failures,
fallback/exclusion counts, timings and measured peak RAM/VRAM. An interruption/resume
trial precedes the frozen model run, and no frozen v2 score is viewed before the
phase-5 prerequisites pass. Phase-5 evidence includes `checkpoints/progress.json`, per-window
NPZ diagnostics and the tables produced by `scripts/fev_report.py`. Use [the Colab template](colab.md) and
[prompt 00](colab/prompts/00-profile-and-baselines.md) for the first profiling/baseline
session. No Timestral weights are downloaded.

## Profile CSV contract

`python scripts/profile_fev.py --tasks weekly --out profile.csv` writes the following
columns in order. Selectors also accept comma-separated task names or a task YAML;
`--profiles roles.yaml` supplies the required roles for custom tasks. `--resume`
preserves completed rows and adds missing tasks. Without it, selected tasks replace
the output as they complete. Loading errors stop the command and preserve completed rows.
A profile written before `distinct_origins_first_window` exists cannot be resumed; use a
fresh output path. Older twelve-column CSVs are rejected and left unchanged.

| Column | Definition |
| --- | --- |
| task | Task name from `FevTaskSpec`. |
| n_series | Number of items in fev's full preprocessed dataset, before window filtering. |
| length_min | Minimum full target-history length in time steps, including missing targets. |
| length_median | Median full target-history length in time steps, including missing targets. |
| length_max | Maximum full target-history length in time steps, including missing targets. |
| anchor_weekday | Pandas `weekday()` of the first timestamp of fev's first item (id-sorted): Monday 0 through Sunday 6. This is not an anchor-distribution check. |
| nan_share | Count of missing target values divided by **all** target values, including missing values. |
| negative_share | Count of negative target values divided by **all** target values, including missing values. |
| zero_share | Count of zero target values divided by **all** target values, including missing values. |
| zero_scale_items | Items whose full-history `mean(abs(y[t] - y[t-m]))` is zero or undefined/non-finite; m is the task's seasonality. Missing lag pairs are omitted, and no valid pair means undefined. |
| eligible_items_first_window | Items returned in fev's first window with past length at least `min_context_length` and positive finite scale computed on that window's past alone. This is not the inner-validation eligibility test. An empty first window gives zero. |
| static_nan_share | Missing cells across all declared static columns and full-dataset items, numeric and categorical; 0.0 when there are no static columns. |
| distinct_origins_first_window | Number of distinct first future timestamps across the items of fev's first window; 0 for an empty first window. |

The seven tasks are univariate. For custom multivariate tasks, shares cover all target
columns, and scale pools observed lag differences across target columns within each
item; item counts still count dataset rows. Profiles never train or evaluate a model.
