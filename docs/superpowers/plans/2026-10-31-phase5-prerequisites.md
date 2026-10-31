# Phase-5 prerequisites and the frozen v2 Colab prompt — implementation plan (PR #25, rev 2)

> **For agentic workers:** one Codex `-p worker` dispatch implements Tasks A–D in the worktree
> `.worktrees/feat-phase5-prereqs` (branch `feat/phase5-prereqs`, based on `fix/colab-findings`
> @ db4c3e8 = PR #24). Claude reviews, commits, opens the PR.

**Goal:** make the first real `fev run` on Colab observable, resumable without corrupting its
evidence, and reportable per the frozen protocol, and ship the Colab prompt that executes it.

**Spec:** `docs/protocol.md` (frozen; §"Tasks and scores" l.36 requires per-knot pinball and 80 %
interval coverage/width, §"Persistence and execution evidence" requires timings, peak RAM/VRAM,
fallback/exclusion counts), `docs/colab.md` (template, PR #24 version). Rev 2 incorporates the
Codex review of rev 1 (10 findings) and the Colab operator's findings B1–B10.

## Global Constraints (in addition to the merged plan's, which still bind)
- Nothing trains on the developer Mac: tests use tiny configs (d_model ≤ 16, ≤ 150 optimizer
  steps, CPU). CUDA paths are covered by monkeypatched tests only.
- No protocol number changes. Telemetry must not change training results: `history`, RNG
  consumption, selection and optimizer behaviour stay identical, and the resume-parity tests in
  `tests/test_core/test_training.py` (whole-history comparisons at l.341/605/619) and
  `tests/test_bench/test_fev_run.py` stay green **unchanged**, except tests that intercept
  `Predictor.forecast_arrays`, which may be re-pointed at `forecast_with_diagnostics` (say so).
- Diagnostics use `logging.getLogger(__name__)` at INFO. The CLI-facing progress line
  `Skipping completed task <name>` is `print(..., flush=True)` in `bench/run.py`, matching
  `run_baseline`; no other `print` in library code.
- Every file written during a run is published atomically (temp file + `os.replace`).
- `json.dumps(..., allow_nan=False)` stays in `_write_run`: non-finite diagnostic values are
  serialised as `null`.
- fev extras: `Task.evaluation_summary(..., extra_info=...)` keys must not shadow fev keys; use
  the names below. Bench imports stay inside functions (bench extra optional).
- Commits by Claude, conventional prefixes, no AI attribution.

## Task A — trainer telemetry, cumulative timings, progress file
**Files:** `demand_forecast/core/training.py`, `demand_forecast/core/run.py`; tests in
`tests/test_core/test_training.py`, `tests/test_cli.py`.
**Why:** with `--stop-after`, `train_panel` raises before writing `run.json`; nothing in `core/`
or `bench/run.py` logs; `run.json` timings cover one session (operator B2; Codex Q2/Q3/Q6).
**Produces:**
- `Trainer.timings: dict` = `{"train_s", "eval_s", "checkpoint_s", "sessions", "peak_cuda_mem_bytes",
  "nonfinite_losses"}`; `train_s` = optimizer-step wall time (batch fetch + forward/backward/step),
  `eval_s` = evaluation wall time, `checkpoint_s` = checkpoint writes; all cumulative across
  resumed sessions, persisted in the checkpoint as elapsed durations (never timer origins) and
  restored by `resume` (peak = max across sessions). `peak_cuda_mem_bytes` =
  `torch.cuda.max_memory_allocated(device)` sampled after every evaluation, at every checkpoint
  and at completion, after `torch.cuda.reset_peak_memory_stats(device)` at the start of each
  `fit` session (CUDA only; 0 otherwise). `nonfinite_losses` counts non-finite step losses,
  cumulative. **`history` is unchanged** (numerical records only).
- `Trainer.telemetry: list[dict]` — one record per evaluation
  `{"step", "elapsed_s", "train_s_per_step" (since the previous evaluation, excluding evaluation
  and checkpoint time), "eval_s", "peak_cuda_mem_bytes"}`; persisted in the checkpoint; not part
  of `history`, model state or run identity. `FitResult.timings` and `FitResult.telemetry`.
- INFO logs: fit start `fit: max_steps=%d eval_every=%d fit_ids=%d val_items=%d origins=%d
  device=%s amp=%s`; per evaluation `eval: step=%d/%d inner_SQL=%.4f best=%.4f@%d patience=%d
  train_s_per_step=%.4f eval_s=%.1f elapsed_s=%.0f peak_cuda_mem_MiB=%d`; `checkpoint: step=%d
  path=%s`; `fit done: stop_reason=%s steps=%d`.
- `checkpoints/progress.json` (atomic) written at every checkpoint and at completion:
  `{run_identity, step, max_steps, eval_every, n_fit_ids, n_val_items, exception, device,
  amp_dtype, timings, telemetry_tail: last 3 telemetry records}`.
- `run.json` gains `resolved_budget: {max_steps, eval_every}`, `n_fit_ids`, `n_val_items`,
  `amp_dtype`, `timings.cumulative` (the trainer's timings) next to the existing session
  timings, `telemetry` (full list), `fit_result.nonfinite_losses`.
- `run_fev` logs INFO `panel: task=%s window=%d series=%d build_s=%.1f` around `panel_from_window`
  and `identity: task=%s %s` once per task; `train_panel` logs `plan: train_upto=%s val_origin=%s
  exception=%s`.
**Tests (TDD, tiny, controlled clock via monkeypatched `time.perf_counter` where sums are
asserted):** `test_stop_after_writes_progress_json_and_resume_accumulates_timings` (stop_after 3 →
progress.json step 3, `timings.sessions == 1`; resume → `sessions == 2`, `train_s` monotone,
run.json has `resolved_budget`, `timings.cumulative`, `telemetry`); `test_timing_categories_sum_
with_controlled_clock`; `test_nonfinite_losses_counter_is_cumulative` (monkeypatched loss);
`test_telemetry_without_validation` (plan.exception → no eval records, progress.json still
written); `test_fit_logs_budget_and_evaluations(caplog)`; `test_run_json_serialises_nonfinite_as_null`.
Resume-parity tests unchanged and green.

## Task B — fev run: verified skip of completed tasks, canonical timings, GPU inference, fallback diagnostics
**Files:** `demand_forecast/bench/run.py`, `demand_forecast/inference/predictor.py`,
`demand_forecast/models/base.py`, `demand_forecast/core/run.py` (`resolve_device`); tests
`tests/test_bench/test_fev_run.py`, the predictor and skeleton test modules.
**Why:** `run_fev` re-times a completed task on every `--resume` and replaces its row (operator
B1, Codex P1); after a crash before the summary row is written, reused predictions would yield
≈0 timings (Codex 4); inference runs on CPU regardless of CUDA (Codex Q3); the protocol requires
fallback counts (operator (e), Codex Q6).
**Produces:**
- `ForecastModel.fallback_flags(batch) -> dict[str, Tensor]`: `{"no_context": ~batch.has_context,
  "undefined_scale": batch.has_context & ~batch.eligible}` (the second includes zero and
  non-finite scales); `predict` uses them (identical outputs; existing predict tests pin it).
- `Predictor.forecast_with_diagnostics(panel, origin, batch_size=512) -> (ids, arrays,
  diagnostics)` with `{"fallback_no_context": int, "fallback_undefined_scale": int, "n_items": int,
  "inference_time_s": float, "device": str}`; `forecast_arrays` delegates (tests that intercept
  `forecast_arrays` are updated deliberately).
- `core/run.py`: public `resolve_device(settings) -> torch.device` (renamed `_device`).
- `bench/run.py::run_fev`:
  - `Predictor.from_bundle(task_dir / "bundle", device=resolve_device(settings))`.
  - Per-window NPZ carries the diagnostics as extra scalar arrays (`inference_time_s` float64,
    `fallback_no_context` int64, `fallback_undefined_scale` int64, `device` unicode, next to
    `ids`, `arrays`, `identity`; all readable with `allow_pickle=False`, no object dtype), so
    diagnostics are published atomically with the predictions through the existing temp-file
    writer; a legacy NPZ without them yields `None` diagnostics in memory ("unknown", never
    zero, never written back).
  - `train_panel` is always called under `--resume` (its identity/bundle verification is the
    recovery path); `reused = resume and (task_dir / "bundle/manifest.json").exists()` evaluated
    before the call; prediction identity checks stay as today.
  - Canonical timings come from persisted measurements, not this invocation's timers:
    `training_time_s = run.json timings.cumulative.train_s + eval_s + checkpoint_s`,
    `inference_time_s = sum of the windows' persisted inference_time_s` (`None` if any window
    lacks it → the summary value is `None` and the report says unknown).
  - Skip condition: `reused` and every `window_{i}.npz` exists with diagnostics and identity, and
    `summaries.csv` holds **exactly one** row with `(task, model_name)` whose `run_identity`
    column equals the identity → `print(f"Skipping completed task {spec.name}", flush=True)`,
    keep the row untouched. Otherwise (zero rows, duplicate rows, stale identity, missing window
    or diagnostics) re-evaluate with canonical timings and replace the rows as today.
  - `extra_info={"run_identity": identity, "fallback_no_context": sum|None,
    "fallback_undefined_scale": sum|None, "inference_device": str, "resolved_max_steps": int,
    "n_fit_ids": int, "n_val_items": int, "nonfinite_losses": int, "sessions": int}` and
    `per_quantile_scores=True` to `task.evaluation_summary`.
**Tests (TDD, fixture tasks, tiny settings):** `test_resume_of_completed_task_skips_without_fit`
(third run: `Trainer.fit` not called via monkeypatch, skip line printed, summaries.csv sha256
unchanged); `test_completed_task_with_stale_identity_row_is_recomputed`;
`test_duplicate_rows_are_replaced_not_skipped` (existing l.293 behaviour kept);
`test_deleted_row_is_rebuilt_from_persisted_timings` (row rebuilt, `training_time_s` equals
run.json cumulative, `inference_time_s` equals the npz sum, not ≈0);
`test_legacy_npz_without_diagnostics_reports_unknown`; `test_npz_carries_fallback_counts`
(fixture with an all-NaN series → `fallback_no_context == 1` per window; constant series →
`fallback_undefined_scale ≥ 1`); `test_extra_info_columns_present_and_baseline_rows_nan`;
`test_predictor_diagnostics_match_arrays`; `test_run_fev_uses_settings_device(monkeypatch)`;
`test_completed_task_with_mismatched_artifact_identity_raises` (bundle manifest or one NPZ with a
foreign identity while every artifact exists → `ValueError("run identity mismatch")`, never a
skip); `test_npz_publication_failure_leaves_no_partial_file` (inject an exception inside the
writer → no `window_i.npz`, no stray temp file in the task dir); existing missing-window recovery
test (l.92) unchanged (NPZ files are only ever created, never rewritten).

## Task C — `scripts/fev_report.py` (bench extra required)
**Files:** `scripts/fev_report.py` (new), `tests/test_bench/test_fev_report.py` (bench-marked).
**Why:** protocol numbers are spread over `run.json`, NPZ diagnostics, `progress.json`,
`summaries.csv`, and some (per-knot pinball, 80 % coverage/width, scoring exclusions) need the
task's labels (Codex 5/6; operator (e)).
**Produces:** `python scripts/fev_report.py --run-dir DIR [--tasks weekly] [--profiles P]
[--baseline seasonal_naive] [--model-name demand-forecast-v2] [--allow-partial] --out DIR/report-tables.md`:
loads each selected task (`load_task_specs`), verifies `dataset_fingerprint(task)` equals the
fingerprint in both the baseline and model rows, aligns each `window_{i}.npz` by id to
`task.iter_windows()` (`get_input_data()` past + `get_ground_truth()`), and writes Markdown:
(1) per task: resolved max_steps/eval_every, n_fit_ids, n_val_items, exception, steps,
best_step, best_score (inner SQL), stop_reason, resumed_from_step, sessions, cumulative
train_s/eval_s/checkpoint_s, peak_cuda_mem_MiB, amp_dtype, nonfinite_losses (from `run.json`);
(2) per task: SQL, MASE, WAPE, WQL, num_forecasts (item-window forecasts), v2/baseline SQL ratio
(only when fingerprint and num_forecasts agree, else "not comparable"), per-quantile SQL columns,
fallback counts per window and total, inference device/time, and from the recomputation with
`demand_forecast.metrics.summarize`: per-knot pinball (labelled "doubled, unscaled pinball" —
the helper returns fev's 2× convention), 80 % interval coverage and width
(fev's equal-window averaging), **exclusions**: item-window pairs with non-finite scaled
contributions (zero/undefined seasonal error) and excluded horizon cells (missing labels), SQL's
quantile dimension stated; (3) artifact sizes per task (`checkpoints/`, `bundle/`, NPZ) and
totals; (4) completeness: every selected task has exactly one model row with matching identity,
one baseline row, all expected windows with diagnostics; missing/unknown items listed; required
metrics finite (model-only diagnostic columns may be NaN on baseline rows). Exit 1 on
incompleteness or non-finite required metrics unless `--allow-partial`.
**Tests:** run `run_baseline` + `run_fev` on two fixture tasks (tiny settings), then the script
with `--tasks <fixture yaml>`: tables contain both tasks, ratio, coverage/width, exclusions
(fixture with a constant-past series → ≥ 1 excluded pair; a window with a missing label → ≥ 1
excluded cell), sizes, exit 0; reordered ids in one NPZ still align; a dir missing one task
exits 1 and names it; a fingerprint mismatch marks the ratio "not comparable".

## Task D — frozen config, prompt 01, template additions, prompt 00 alignment
**Files:** `configs/fev-v2-standard.yaml` (new; appendix verbatim),
`docs/colab/prompts/01-fev-run-v2.md` (new; appendix verbatim, then align only CLI flags/keys
with what Tasks A–C implement), `docs/colab.md` (§4: the detached launcher with an `EXIT` trap
writing the status file, the VRAM `Popen` poller, the exit-1 `PIPESTATUS` pattern, model size
depends on fitted vocabularies, the online dataset-pinning rule: verify the Hub commit before any
load and the parquet sha256 manifest after it — `HF_HUB_OFFLINE` must NOT be used because `datasets`
4.0.0 offline mode cannot rebuild an empty `/content` datasets cache from the Hub snapshot, and
`FEV_DATASETS_PREFIX` is not used either because it changes the task's `dataset_path`, hence the
task definition recorded in `summaries.csv` and compared against the published fev tasks),
the detached launcher without `tee`,
`docs/colab/prompts/00-profile-and-baselines.md` item 2 (datasets cache default `/content/hf-datasets`,
bot note on PR #24), `docs/protocol.md` (one sentence naming `progress.json`, NPZ diagnostics and
`scripts/fev_report.py` as the phase-5 evidence; no numbers change).
**Acceptance:** `Settings.from_yaml("configs/fev-v2-standard.yaml")` loads and equals the
protocol's standard values (test in `tests/test_config.py`); ruff format passes on Markdown
fences; neither doc claims CSV byte-identity after re-evaluation nor uses "background cell".

Appendix files next to this plan: `fev-v2-standard.yaml`, `01-fev-run-v2.md`.
