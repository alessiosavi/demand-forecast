# fev batch performance and the EC2 frozen run — Implementation Plan (rev 4)

> **Rev 4 (2026-11-01).** The prompt 02 quick run at `2863356` found non-finite training losses
> on walmart (150/150 steps) and rohlik_sales_1W (99/150). The fp16 GradScaler skipped every one,
> so walmart never took an optimizer step. A frozen run at `2863356` would therefore be defective,
> and so would the old fallback. Changes:
> - Gate 0 adds the NaN fix (peer session demand-forecast-2d owns it) merged, plus a clean quick run.
> - The fallback and P3 reference become that fix's merge SHA, written **F** below.
> - `nonfinite_losses > 0` on any task blocks acceptance in P4 and in the frozen report.
> - This branch rebases onto F before its PR merges. If the fix touches `data/sampler.py` or
>   `core/evaluation.py`, the reference oracle and `BatchBuilder` are re-frozen on F.
>
> Measured scaler state (quick run, T4): start-of-run skips are 0–2 on the five healthy tasks.
> That supports D4.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build inner-validation batches once per fit and training batches without per-row pandas
work, with tensors bit-identical to `2863356`, then run the frozen seven-task protocol on one EC2
T4 instance as parallel per-task processes.

**Architecture:** `prepare_validation` builds the validation inputs once per fit session;
`evaluate_prepared` scores them. A `BatchBuilder` caches what `make_batch` derives from a series
alone (calendar features and grid per series start, static encodings, label cut-offs) and keeps
every per-row reduction and scalar conversion exactly as today. The EC2 run gives each task its
own output directory and merges them afterwards. Summary writing, run identity, sampling, model
and metrics are untouched.

**Tech Stack:** Python 3.12, numpy, pandas, torch (fp16 AMP + GradScaler on T4), fev 0.10.0,
AWS EC2 g4dn (eu-central-1, profile `veliu-staging`).

**Spec:** design draft of 2026-10-31 (session scratchpad
`quick/fev-performance-and-ec2-design-draft.md`; summarised in the ledger entry of 2026-10-31),
[`docs/protocol.md`](../../protocol.md) (binding), [prompt 01 rev 2](../../colab/prompts/01-fev-run-v2.md)
(prerequisites A–D), ledger `.superpowers/sdd/2026-10-31-fev-core-v2/progress.md`. Codex review of
rev 1: changes needed, 10 findings, all accepted (finding 8 in part, see D1). Codex re-review of
rev 2: changes needed, 7 findings (cut-off validity across grid growth, timezone in the cache key,
capacity evidence, full prerequisite D, executable P3 rule, cross-path continuation test, venue
wording), all accepted into rev 3 as specified.

## Decisions (user go: 2026-10-31)

| # | Decision | Basis |
| --- | --- | --- |
| D1 | **Performance first, then the frozen run on the merged SHA.** Order inside the PR: validation caching (Task 1, measured cost: m5 evaluation is 56% of its budget) before the `BatchBuilder` rewrite (Task 2, behind a profiling gate). If any EC2 parity gate fails, the same instance runs the frozen protocol at F (the NaN-fix merge SHA; rev 4) under the fallback procedure (Task 5). Gate 0: the Colab quick run (prompt 02) passes. | User chose this over the cheapest path (today's code, parallel, ≈ 5.6 h wall). Codex notes that path is viable for the earliest frozen result. The speed gains are estimates until P4 measures them. Correctness does not depend on them, because parity is exact. |
| D2 | **g4dn.4xlarge on-demand**: 1× T4 16 GB, 16 vCPU, 64 GiB, $1.505/h (eu-central-1, AWS Pricing API, 2026-10-31). | T4 keeps the fp16 + GradScaler path that prerequisites A–D and the quick run exercised. 16 vCPU (8 cores) covers the fallback's CPU-bound processes. g5/g6 (A10G/L4) select bf16 without a scaler: protocol-permitted but no real-data run has used it. The identity hashes the configured `device` (`null` here), not the resolved hardware or AMP dtype, so a cross-hardware resume is not detected. g4dn.12xlarge has four T4s and is not needed. Spot saves ≈ $3–6, but one interruption kills every process. |
| D3 | **Budget $50 (user).** Expected ≈ $7.5–9 (≈ 5–6 h including the gates and the quick dress rehearsal); fallback ≈ $15 (≈ 10 h). Twelve-hour compute guard: `shutdown -h +720` with `InstanceInitiatedShutdownBehavior=stop` (≈ $18 compute per guarded session). EBS stays billable after a stop until the volume is deleted. Transfer is budgeted from the measured artifact size. | Hourly price × hours. The guard is not a hard ceiling: setup delays, retries and the stop latency fall outside it. The remaining headroom covers one restart. |
| D4 | **No warm-up amendment before the frozen run.** Keep the frozen semantics: the LR schedule, `eval_every`, `checkpoint_every` and `max_steps` count attempted optimizer steps, including steps the fp16 scaler skips. Count skips (`timings["scaler_skips"]`) and document the semantics (Task 3). | Changing it introduces a second step counter, so cosine would not reach zero at `max_steps`. The step is not harmless to drop either: a zero-LR AdamW step still updates the moments and the step counter. Skip counts are unmeasured; prompt 02 reports start-of-run scales and Task 3 reports totals. A later amendment is phase-4 work with its own dated entry. |
| D5 | **No panel cache and no fast panel builder now.** `run_fev` builds a panel per window inside each task's own process (m5 ≈ 10 min; favorita/hierarchical 10 windows, rossmann 8). `fev_report.py` reads window data but does not call `panel_from_window`. Revisit in phase 4. | A cache adds an invalidation surface to save minutes inside a parallel run. |
| D6 | **One output directory per task** (`ROOT/<task>`), each running `fev baseline` then `fev run` for that task; `scripts/fev_merge_runs.py` builds `ROOT/merged` for `fev_report.py` and `fev leaderboard`. | Confirmed race: `run_fev` reads `summaries.csv` once (`bench/run.py:165`) and replaces it after each task (`:308-313`). Atomic replace prevents torn files, not lost updates. |
| D7 | **Stop rule: frozen budget unchanged (user).** Early stopping (patience 5, `rel_tol` 0.002) acts inside each task's resolved ceiling (500–12,000 steps). No Optuna in the frozen run: tuning is phase 4 and needs a preregistered budget. The report shows `steps`, `max_steps`, `best_step` and `stop_reason` per task. If inner validation shows tasks still improving at the ceiling, a preregistered budget study judged on inner validation only is the first phase-4 item. | The ~300-step cap belongs to the quick config only. hierarchical ended at its 500-step ceiling with its best score at the last evaluation. |

## Expected effect (estimates unless marked measured)

| Item | Today (measured, T4, 67d4bd7) | After Tasks 1–2 (estimate; measured in P4) |
| --- | --- | --- |
| Training step | 0.35–0.66 s, one core at 98%, GPU 0.3–1.9 GB | GPU-bound; value under seven-way GPU sharing unknown |
| m5 evaluation | 179–186 s × 60 | inference + `summarize` only |
| Parallel wall | ≈ 5.6 h (m5 5.33 h + build) | ≈ 1–2 h |

## Global Constraints

- Every `Batch` field is `torch.equal` to `2863356`'s `make_batch`, with the same dtype, shape and
  `cat` keys, for the same `(series, origin)` list and `label_upto`. Validation metrics equal
  `2863356`'s `evaluate` for the same model state (NaN equals NaN). Errors raised for the same inputs
  (types and messages) are unchanged, including float32 overflow on `loc`/`scale`.
- Do not add a `Settings` field, change the `core/run.py` identity payload or bump `__version__`.
  The pinned identities in `tests/test_bench/test_fev_run.py` stay unchanged and `2863356`
  checkpoints keep resuming.
- `WindowSampler` draw order, RNG streams and checkpointed state (`epoch`, `position`) unchanged.
- Model, losses, budget function, validation rule, metrics and fev scoring unchanged. Evaluation keeps
  512-item chunks in item order, float64 metric inputs and no autocast. Do not enable
  deterministic-algorithm settings.
- Local runs: fixture tests and micro-benchmarks of seconds only. No real data, no training, no full
  suite on the Mac (CI runs it). Codex workers obey the same rule.
- Codex implements in `.worktrees/perf-fev-batches-ec2` (branch `perf/fev-batches-ec2`); Claude commits.
- EC2: every torch command runs under `env -u LD_LIBRARY_PATH` (DL AMI cuDNN trap). All probes and the
  frozen run use the same venv lock, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2` and batch shapes. One
  hardware type and one dtype per run directory.

## Review Focus

1. Series whose grids do not share a phase (two `2D` series starting on consecutive days, hourly
   with DST, monthly, a late start), equal instants in different time zones, and a short request
   followed by a long one for the same start and `label_upto` (same-start series with unequal
   histories) → Task 2 fixtures `tz_collision`, `cutoff_growth`, `phase_mismatch`,
   `hourly_dst`, `monthly`, `late_start`.
2. Exceptional scalars: a constant finite target near `1e40` raises exactly as today;
   empty panel and empty item list behave as today → Task 2 fixtures `overflow_scale`, `empty`.
3. Builder state across calls: varying `label_upto`, and batch A rechecked after building batch B
   (no shared buffers) → Task 2 tests `test_label_upto_varies_across_calls`, `test_earlier_batch_unchanged`.
4. Items beyond `history_length` via `align_to_origin` (walmart early-ending items), with
   NaN-extended past-only covariates and retained known covariates → Task 2 fixture `ragged_aligned`.
5. Missing or unseen statics and binary/all-NaN dynamic pasts → Task 2 fixtures `unseen_statics`,
   `indicator_column`, `all_nan_past`.

---

### Task 1: inner validation prepared once per fit session

**Files:**
- Create: `tests/helpers/reference_batch.py`: verbatim copies of `2863356`'s `align_to_origin`,
  `_dynamic_values`, `make_batch` and `core/evaluation.evaluate`, renamed `reference_*`, with
  `reference_evaluate` calling the reference alignment and batching. Header comment names the SHA.
  Also `assert_batches_equal(a, b)` (field by field, dtype, shape, `cat` keys).
- Modify: `demand_forecast/core/evaluation.py`, `demand_forecast/core/training.py`
- Test: `tests/test_core/test_evaluation.py` (create), `tests/test_core/test_training.py`

**Interfaces:**
- Produces in `core/evaluation.py`: `@dataclass(frozen=True) PreparedValidation(batches: tuple[Batch, ...], y_true: np.ndarray, y_past: tuple[np.ndarray, ...], seasonal_period: int)` with `nbytes: int`;
  `prepare_validation(panel, items, origin, horizon, max_context, encoder, batch_size=512) -> PreparedValidation`;
  `evaluate_prepared(model, prepared, quantiles) -> dict[str, float]` (same `model.eval()` / `finally: model.train()`).
  `evaluate(...)` keeps its signature and returns `evaluate_prepared(model, prepare_validation(...), quantiles)`.
- Produces in `tests/helpers/reference_batch.py`: `reference_make_batch`, `reference_align_to_origin`,
  `reference_evaluate`, `reference_batches(sampler, batch_size)` (2863356's `WindowSampler.batches` loop over
  `reference_make_batch`), `assert_batches_equal`.

- [ ] **Step 1: Write failing tests.**
  - `test_evaluate_prepared_matches_reference`: fixture panel with an aligned item and a partial
    horizon; dict equal to `reference_evaluate`.
  - `test_prepared_batches_survive_cpu_evaluation`: evaluate twice on CPU; metrics equal and batches
    equal to clones taken before.
  - `test_fit_prepares_validation_once`: counter on `prepare_validation`; one call per `fit`
    session, one more after resume, none under `plan.exception`. Prepares the already-subsampled
    `plan.val_items`.
  - `test_fit_matches_reference_path` (CPU): `fit` with defaults vs `fit(evaluate_fn=reference_evaluate)`;
    histories equal and `best_state` tensors equal.
  - `test_fit_custom_evaluate_fn_keeps_old_call`.
  - `test_prepare_time_charged_once`: controlled clock; prepare time added once per session to
    cumulative `timings["eval_s"]`, never to per-evaluation telemetry `eval_s`.
- [ ] **Step 2: Run** `pytest tests/test_core/test_evaluation.py tests/test_core/test_training.py -q` → new tests FAIL.
- [ ] **Step 3: Implement.** In `Trainer.fit`, when `evaluate_fn is evaluate`, not
  `plan.exception` and `plan.val_items` is non-empty, prepare once after any resume and before the
  loop. Keep batches on CPU and move each to the device at evaluation. Log
  `validation prepared: items=%d batches=%d bytes=%d build_s=%.1f`.
- [ ] **Step 4: Run** `pytest tests/test_core tests/test_bench/test_fev_run.py -q` → PASS; existing fit-history parity tests and pinned identities unchanged.
- [ ] **Step 5: Commit** `perf: build inner-validation batches once per fit session`.

### Task 2: `BatchBuilder` with exact parity (behind a profiling gate)

**Files:**
- Modify: `demand_forecast/data/sampler.py` (`BatchBuilder`; `make_batch` delegates; `WindowSampler` owns one builder)
- Create: `scripts/bench_batches.py`
- Test: `tests/test_data/test_batch_builder.py`, `tests/test_core/test_training.py`

**Interfaces:**
- Consumes: `tests/helpers/reference_batch.py` (Task 1).
- Produces: `BatchBuilder(panel: Panel, horizon: int, max_context: int, encoder: StaticEncoder)`;
  `BatchBuilder.build(items, label_upto: pd.Timestamp | None = None) -> Batch`.
  `make_batch(panel, items, horizon, max_context, encoder, label_upto=None) -> Batch` keeps its
  signature, errors and messages and returns `BatchBuilder(...).build(items, label_upto)`.

- [ ] **Step 0: Profiling gate.** `python scripts/bench_batches.py --profile` builds a synthetic m5-shaped
  panel (2,000 series × 277 weekly steps, 8 known covariates, 5 categorical statics, seed 0), then
  cProfiles 20 reference batches of 128. Report the cumulative share of `grid_range` +
  `calendar_features`, `encoder.transform`, per-row torch slice assignment, `sample_frame` and
  `_dynamic_values`. **If the cacheable parts plus torch assignment are below 60%, stop and report to
  Claude before Step 1.**
- [ ] **Step 1: Write failing parity tests.**
  - `test_build_matches_reference[<fixture>]` for `shared_start`, `mixed_starts`, `tz_collision`,
    `cutoff_growth`, `phase_mismatch`,
    `hourly_dst`, `monthly`, `late_start`, `ragged_aligned`, `indicator_column`, `all_nan_past`,
    `label_cutoff`, `week53`, `daily`, `origin_zero`, `long_context`, `unseen_statics`:
    `assert_batches_equal(builder.build(items, label_upto), reference_make_batch(...))`.
  - `test_build_errors_match_reference[overflow_scale|empty|bad_items]`: same exception type and message.
  - `test_build_matches_reference_random_origins`: 20 batches of 128 from `build_origins` with
    `np.random.default_rng(0)`.
  - `test_label_upto_varies_across_calls` and `test_earlier_batch_unchanged`: one builder.
  - `test_sampler_batches_match_reference`: two epochs of `WindowSampler(seed=0)` vs `reference_batches`.
  - Extend `test_fit_matches_reference_path` (CPU) to patch `WindowSampler.batches` with
    `reference_batches` on the reference leg only; histories and `best_state` stay equal.
  - `test_reference_checkpoint_resumes_on_optimized_path` (CPU): interrupt a reference-path fit,
    then resume its checkpoint on the optimized path; compare with the reference continuation.
    Model/optimizer/scheduler state, history, sampler position, best step and `stop_reason` must be
    equal; timings are excluded.
- [ ] **Step 2: Run** `pytest tests/test_data/test_batch_builder.py -q` → FAIL (`BatchBuilder` undefined).
- [ ] **Step 3: Implement `BatchBuilder`.** Never use one panel-wide grid.
  - Cache per key `(start, str(start.tz))`, built lazily on first use at full length: the maximum
    `history_length` of the series sharing that key, plus `horizon`. The cache holds the grid and its
    `calendar_features`. `make_batch` already rejects `origin > history_length`, so a grid never
    needs to grow.
  - Cache `encoder.transform(series)` per series index.
  - Compute label cut-offs as `searchsorted(label_upto, side="right")` on that full-length grid,
    cached per `(key, label_upto)`. Keep per-row `sample_frame` and `_dynamic_values` on identical slices (no
  cumulative sums). Keep the per-row scalar tensor assignment for `loc` and `scale`, which preserves
  torch's overflow check. Fill other fields as fresh numpy float32/bool arrays per call, converted
  once with `torch.from_numpy`.
- [ ] **Step 4: Run** `pytest tests/test_data tests/test_core tests/test_bench/test_fev_run.py -q` → PASS.
- [ ] **Step 5: Micro-benchmark.** `python scripts/bench_batches.py` prints reference vs builder ms per
  128-row batch over 20 batches. Report the ratio; it is not a pass/fail gate.
- [ ] **Step 6: Commit** `perf: cache series features in BatchBuilder with exact batch parity`.

### Task 3: scaler-skip telemetry and protocol record

**Files:**
- Modify: `demand_forecast/core/training.py`, `demand_forecast/bench/run.py` (summary `extra_info`),
  `scripts/fev_report.py` (training table next to `nonfinite_losses`), `docs/protocol.md`
- Test: `tests/test_core/test_training.py`, `tests/test_bench/test_fev_report.py`

**Interfaces:**
- Produces: `timings["scaler_skips"]: int | None`: `0` in `_empty_timings()`, +1 when the scaler is
  enabled and `get_scale()` after `update()` is below the value before it. When neither the checkpoint
  timings nor the ZIP-footer `timings.json` has the key, a resume sets it to `None` (unknown, never
  zero) and it stays `None`.

- [ ] **Step 1: Write failing tests.**
  - `test_scaler_skips_counted`: stub scaler with decreasing, unchanged and increasing scales and
    skipped `optimizer.step` calls → only decreases count.
  - `test_scaler_skips_cumulative_across_resume`.
  - `test_scaler_skips_zero_without_scaler` (CPU).
  - `test_scaler_skips_unknown_after_legacy_resume`: checkpoint and footer both lack the key. Also
    extend Task 2's `test_reference_checkpoint_resumes_on_optimized_path` with the key removed from
    both; the continuation stays equal and `scaler_skips` is `None`.
  - `test_report_shows_scaler_skips`: `None` renders as unknown.
- [ ] **Step 2: Run** the tests → FAIL.
- [ ] **Step 3: Implement**, then edit `docs/protocol.md` (dated 2026-10-31).
  - (a) Clarification under the budget section: the schedule, `eval_every`, `checkpoint_every` and
    `max_steps` count attempted optimizer steps. A step the fp16 scaler skips still advances the
    schedule; torch warns `lr_scheduler.step() before optimizer.step()`. Skips are reported as
    `scaler_skips`; bf16 and CPU runs never skip.
  - (b) Venue, also updating the introduction's "model runs on Colab" sentence: Colab holds the
    earlier profiling and baseline evidence and the quick runs. EC2 g4dn.4xlarge (T4, fp16),
    authorised by the user on 2026-10-31, hosts the frozen phase-5 run together with its
    environment-validation baselines, prerequisites and rehearsals.
  - (c) The D7 stop rule, restated as unchanged.
- [ ] **Step 4: Run** `pytest tests/test_core/test_training.py tests/test_bench -q` → PASS.
- [ ] **Step 5: Commit** `feat: report fp16 scaler skips; record attempted-step schedule and EC2 venue`.

### Task 4: run merge, real-data parity probe, EC2 launcher and runbook

**Files:**
- Create: `scripts/fev_merge_runs.py`, `tests/test_bench/test_fev_merge_runs.py`,
  `scripts/ec2/check_real_parity.py`, `scripts/ec2/frozen_parallel.sh`, `docs/ec2.md`

**Interfaces:**
- Produces: `python scripts/fev_merge_runs.py --run-root ROOT --tasks weekly [--profiles P] [--allow-partial]`.
  It writes `ROOT/merged` only, using `float_precision="round_trip"` for reads.
  - Each `ROOT/<task>/summaries.csv` may contain only rows of that task.
  - It refuses duplicate `(task, model_name)`, foreign-task rows, missing tasks (unless
    `--allow-partial`) and an existing non-empty `ROOT/merged`.
  - It writes `summaries.csv` atomically plus relative symlinks `merged/<task> -> ../<task>/<task>`.
  - Exit 0 on success, 2 on refusal.
- Produces: `python scripts/ec2/check_real_parity.py --tasks m5_1W,hermes,walmart --batches 200 --out FILE`
  (EC2 only; imports `tests.helpers.reference_batch`).
  - Per task, it builds the first-window panel once and compares against the reference: 200
    `WindowSampler(seed=0)` batches vs `reference_batches`; prepared validation vs the reference;
    the window's inference batches vs `reference_make_batch`.
  - It also compares `evaluate_prepared` with `reference_evaluate` for a seed-0 initialised model on
    the GPU.
  - For walmart it prints prerequisite D's origin facts (distinct, earliest, latest, `train_upto`,
    `val_origin`).
  - One line per check; non-zero exit on any mismatch.
- Produces: `scripts/ec2/frozen_parallel.sh --venv V --run-root ROOT --config C --tasks t1,… [--stop-after N]`,
  with `set -euo pipefail`.
  - Writes provenance first: executing package SHA and interpreter, `pip freeze`, lock file, config,
    task definitions, exact commands, `nvidia-smi -q`, dataset pin manifest.
  - Each task runs under `setsid nohup env -u LD_LIBRARY_PATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    /usr/bin/time -v` as `fev baseline` then `fev run --policy first_window --resume` into
    `ROOT/<task>`, with per-task `run.log` and `exit` files.
  - Pollers sample every 1 s: per-PID GPU memory (`nvidia-smi --query-compute-apps=pid,used_memory`),
    device-wide `memory.used` and per-PID RSS. They write CSVs, and the report labels the results
    "sampled maxima".

- [ ] **Step 1: Write failing merge tests:**
  - `test_merge_matches_combined_dir`: one combined fixture dir with baseline + model rows, split by
    copying into per-task dirs and merged; `fev_report` output equal to the combined dir's (custom
    task/profile YAML).
  - Refusal tests: `test_merge_refuses_duplicate_rows`, `test_merge_refuses_foreign_task_rows`,
    `test_merge_refuses_missing_task`, `test_merge_refuses_nonempty_out`.
  - `test_merge_allow_partial`.
- [ ] **Step 2: Run** `pytest tests/test_bench/test_fev_merge_runs.py -q` → FAIL.
- [ ] **Step 3: Implement** the merge script, the parity probe and the launcher. Check the launcher with
  `bash -n` and `shellcheck` if available.
- [ ] **Step 4: Write `docs/ec2.md`**, a runbook following Task 5's gates:
  - instance, AMI (DL Base OSS Nvidia Driver, Ubuntu), key pair and SG (SSH from the operator IP
    only), shutdown guard and restart handling;
  - Python 3.12 venv with the Colab constraints lock (`pip install -c`);
  - online dataset pinning and a sequential HF-cache warm-up as in [docs/colab.md §2](../../colab.md),
    with the cache left writable (datasets takes file locks);
  - copy-back verification and teardown, including EBS volumes.
- [ ] **Step 5: Run** `pytest tests/test_bench -q` → PASS. **Commit**
  `feat: per-task EC2 launcher, run merge and real-data batch parity probe`.

### Task 5: EC2 frozen run (Claude, after the PR merges; not a code task)

Every gate's artifacts are kept under `ROOT/../probes-<UTC>` and copied back with the run.

- [ ] **Gate 0:**
  - The NaN fix is merged at F, and a quick run at F or later shows `nonfinite_losses == 0` on all
    seven tasks.
  - This PR is rebased onto F and merged, with green CI and a reviewed bot report.
- [ ] **Provision:** g4dn.4xlarge on-demand, eu-central-1, 150 GB gp3 (`DeleteOnTermination=true`),
  shutdown behaviour *stop*, `sudo shutdown -h +720` at boot. Record the instance and volume IDs in the ledger.
- [ ] **Setup:** venv at the merged SHA, lock enforced, pins verified, HF cache warmed one task at a time.
- [ ] **Prerequisite A** (prompt 01 §3): seven `fev baseline` rows equal prompt 00's to six significant
  figures; an idempotent rerun leaves hashes unchanged.
- [ ] **P1 parity:** `check_real_parity.py` passes for m5_1W, hermes, walmart, including prerequisite D's
  origin facts (11 distinct, earliest 2011-10-14, latest 2012-02-03, `train_upto` 2011-04-29,
  `val_origin` 2011-05-06).
- [ ] **Prerequisite B** (prompt 01 §4, every check), on hierarchical_sales_1W with the PR code:
  - `--stop-after 120` exits 1 with no premature run.json, bundle or NPZ;
  - the resume reports `resumed_from_step 120`, steps 500, `sessions 2`, 10 NPZ diagnostics;
  - an idempotent rerun prints the skip line with equal hashes.
- [ ] **Prerequisite D** (prompt 01 §5b, every check), fresh trial directory: `fev run --tasks walmart … --stop-after 50`.
  - Exits 1; the checkpoint step is 50.
  - Distinct origins 11, earliest 2011-10-14, latest 2012-02-03; `train_upto` 2011-04-29,
    `val_origin` 2011-05-06.
  - `n_fit_ids` 2936, `n_val_items` 2864, `max_steps` 4404, `eval_every` 200, AMP float16.
  - No run.json, bundle or NPZ.
- [ ] **P3 repeatability** (same venv lock, threads and shapes): hermes `--stop-after 200` with the PR
  code once and with **F** twice (F replaces `2863356` throughout P3), run sequentially. Compared quantities: the full-precision
  inner SQL at step 200 from `progress.json`/history, and the `model` tensors in `latest.pt`.
  - If the two F runs are bitwise equal, the PR run must be bitwise equal to them.
  - Otherwise the PR run passes iff `|SQL_PR − SQL_ref1| ≤ 3 × |SQL_ref1 − SQL_ref2|`.
  - A failure with a bitwise-repeatable reference is a code difference → **fallback**.
  - If the reference pair disagrees, rerun the reference once more before assigning any cause.
    Variability that remains unexplained blocks the launch.
  - m5 once with the PR code: its inner SQL@200 is compared with Colab's 0.8674 as corroboration
    only, never a gate.
- [ ] **P4 quick-config dress rehearsal and capacity bound:** run the launcher with
  `configs/fev-v2-quick.yaml` on all seven tasks concurrently, in a trial root.
  - Pipeline-check identity; its scores never inform a decision, as the protocol says.
  - It exercises training at batch 128, inference at batch 512, every window's panel build, the
    baselines, the merge and the report.
  - Sample each PID every 1 s for GPU memory (`nvidia-smi --query-compute-apps=pid,used_memory`)
    and RSS; label the results "sampled maxima".
  - Capacity bound = Σ over tasks of per-process maxima. It is conservative whatever the overlap,
    because CUDA contexts and allocators are per process.
  - Host term: replace each task's quick validation cache with its frozen size,
    `nbytes per item × n_val_items` from the `validation prepared` log. Evaluation moves one
    512-item batch at a time, so inference already bounds the GPU validation peak.
  - Add a 10% margin. Launch all seven together if the bound is ≤ 14 GB VRAM and ≤ 48 GB RSS.
    Otherwise split into waves whose own bound fits, and keep reducing concurrency. If even one task
    does not fit, block.
- [ ] **Non-finite gate:** any task with `nonfinite_losses > 0` blocks acceptance, in P4 and in the
  frozen report. `scaler_skips` beyond start-of-run calibration (measured 0–2 per healthy task) is
  investigated before launch.
- [ ] **Failure handling — fallback** (second venv at **F**, never `2863356`; fresh directories):
  - Provenance records the executing SHA and interpreter.
  - Prerequisites A, B and D and P4 are repeated with that code; `scaler_skips` is reported unknown.
- [ ] **Launch:** fresh `ROOT=runs/ec2-frozen-<sha>-<UTC>`; tasks per the P4 decision; Monitor on the exit files; cost guard checked hourly.
- [ ] **Report:**
  - All `exit` files 0. Then `fev_merge_runs.py --run-root ROOT --tasks weekly` (no `--allow-partial`).
  - `python scripts/fev_report.py --run-dir ROOT/merged --tasks weekly --baseline seasonal_naive --out ROOT/merged/report-tables.md`
    exits 0 with no completeness issue.
  - `fev leaderboard ROOT/merged/summaries.csv --baseline seasonal_naive`.
  - Ledger: hardware/dtype, `scaler_skips`, `steps`/`max_steps`/`best_step`/`stop_reason` per task,
    timings and peak RSS/VRAM.
- [ ] **Copy back and tear down:**
  - `rsync -a` ROOT and the probes to the Mac; compare a sha256 manifest on both sides.
  - Then terminate, delete the key pair and SG, and verify no G instance and no orphan volume remains.
