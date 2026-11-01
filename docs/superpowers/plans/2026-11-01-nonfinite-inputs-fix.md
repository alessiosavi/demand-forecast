# Non-finite training losses (cold covariates, fp16 range) Implementation Plan — rev 2.2

Scoped re-review (rev 2): Task 2 ready with implementation details preserved below; Task 3 three
corrections accepted (no exact GradScaler scale gate; `timings.cumulative.nonfinite_losses` stays
required and feeds the gate when `fit_result` lacks the copy; amendment wording matches Task 1).

Colab diagnostic (2026-11-01, CPU runtime, fp16 emulated; files nan-diagnostic-*.txt in
runs/02-quick-v2-2863356…/): both tasks are fp32-finite and fp16-NaN. walmart: decoder columns 4/6/8
(MarkDowns) have an entirely empty past in 128/128 rows and raw maxima 97,741 / 20,834 / 34,348 →
the identity-transform path, fixed by D1a. rohlik: three offending rows with decoder covariates
4.95e6 / 8.28e6 / 2.03e7 from tiny nonzero past standard deviations (sell_price_main std < 1e-6 in
54 series, smallest 1.71e-8; type_4_discount smallest 1.63e-9) → **tiny-std inflation**, which D1a
does not cover and a bare clamp would merely saturate: rev 2.1 adds a relative std floor. Target
scales were 0.50–1.0 minimum (no overflow) → **D4 not triggered**.

Codex plan review (rev 1): changes needed; eight findings accepted: finite-first sanitization and
bound after both branches; fail-fast entry guard on resume and a persisted terminal reason; stop
sequence records history and diagnostics before checkpointing and `_result` publishes the reason;
skip backward/optimizer on a non-finite loss when the scaler cannot protect parameters; optional
new NPZ counters (legacy NPZ stay valid); 1e3 provisional with forward/backward finiteness tests;
D4 pending until demonstrated overflow in the actual computation; fixture origin and oracle notes.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Training never silently proceeds on non-finite losses. Covariate columns without an
observed past are treated as unobserved for that sample; standardized covariates are bounded;
the trainer fails fast after consecutive non-finite losses; the report and the prompts gate on
`fit_result.nonfinite_losses == 0`.

**Architecture:** All input guards live in the shared sampler helper `_dynamic_values`
(called per row by `make_batch` and by the performance branch's `BatchBuilder`), so both
builders inherit them. The trainer counts consecutive non-finite losses next to the existing
`nonfinite_losses` counter, checkpoints, writes `progress.json` with the stop reason and returns
`stop_reason="nonfinite_losses"`; `train_panel` raises `NonFiniteTraining` (a `RuntimeError`,
sibling of `InterruptedRun`) before any bundle write; the CLI maps it to exit 1.
`scripts/fev_report.py` treats a positive `nonfinite_losses` as a completeness failure.

**Tech Stack:** Python 3.10+, numpy, torch, pytest; fixtures only.

**Spec:** `docs/protocol.md` (binding; this plan adds a dated amendment block). Incident:
prompt 02 quick run at 2863356 (2026-10-31): walmart 150/150 and rohlik_sales_1W 99/150
non-finite training losses; fp16 GradScaler skipped every such step (scaler scale 4.6e-41 and
1.0e-25), parameters finite, walmart effectively untrained; the other five tasks clean. User
decisions (2026-11-01, relayed verbatim from the Colab session's dialog): D1 "D1a: mask it
off", D2 "Yes", D3 "Yes", D4 "Only if data shows it".

**Mechanism (fixture-confirmed, real-data diagnostic pending on Colab):** a known covariate
whose past is entirely missing receives the identity transform in `_dynamic_values`, so raw
future values enter `dec_cov`. walmart MarkDown1–5 are missing before 2011-11 and are dollar
amounts up to ~1.4e5 afterwards; training windows with origins after ~2011-02 reach them in
their 39-step future; 65504 is fp16's maximum, so the autocast forward overflows and the loss
is NaN while fp32 stays finite (probe: `dec_cov` max 1e5, 4 entries > 65504, fp32 loss finite,
fp16-cast inputs → 36 non-finite outputs). Candidate for rohlik: a tiny nonzero past standard
deviation inflating standardized values; the diagnostic's per-column maxima decide the bound.

## Global Constraints

- No `__version__` bump; no new `Settings` fields (the fail-fast threshold and the covariate
  bound are protocol constants in code). Run identities therefore do not change.
- Batch tensors are byte-identical to today for every sample whose covariate columns all have
  an observed past, whose standardization scale is not floored, whose values lie inside the bound
  and whose source values are all finite.
- Helper-level only: no inline logic in `make_batch`'s body beyond calling the helper and
  storing its diagnostics (the performance branch's `BatchBuilder` must inherit the fix by
  calling the same helper).
- Never reduce items, horizons or windows; inference uses the same helper, so the few walmart
  items whose own origin precedes 2011-11 are forecast with MarkDowns masked off.
- `ruff check .` and `ruff format --check .` clean; full suite passes in the Colab-stack venv.

## Review Focus

1. **Cold column at inference:** an item whose forecast origin precedes a column's first value
   must get finite predictions with that column masked off (fixture with a late-starting known
   covariate; `forecast_with_diagnostics` output finite; `cold_covariates` counted).
2. **Bound is reached only by pathological inputs:** a standardized value with a tiny but
   nonzero past std is clamped and counted; an ordinary z-score of 50 is untouched.
3. **Fail-fast evidence survives:** after the stop, `checkpoints/latest.pt` holds the step and
   `progress.json` says `stop_reason: nonfinite_losses`; no `bundle/` and no `run.json`.
4. **Resume after a fail-fast stop** with unchanged code reproduces the stop (same step), and
   `--resume` on the task does not skip it as complete.
5. **Report gate:** `nonfinite_losses > 0` is a completeness issue (exit 1 without
   `--allow-partial`), shown in the Training table.

---

### Task 1: Cold covariates masked off, standardized covariates bounded (sampler helpers)

**Files:**
- Modify: `demand_forecast/data/sampler.py` (`_dynamic_values`, `Batch`, `make_batch`)
- Modify: `demand_forecast/core/training.py` (cumulative diagnostics only, see Interfaces)
- Modify: `demand_forecast/core/run.py` (`run.json` keys)
- Test: `tests/test_data/test_sampler.py`, `tests/test_core/test_training.py`, `tests/test_core/test_run.py`,
  `tests/test_bench/test_fev_run.py` (inference case)

**Interfaces:**
- Produces:
  - `COVARIATE_BOUND: float = 1e3` (module constant in `sampler.py`; the diagnostic may move it,
    the plan review decides the final value).
  - `_dynamic_values(source, start, origin, stop) -> tuple[np.ndarray, np.ndarray, DynamicDiagnostics]`
    where `DynamicDiagnostics = NamedTuple(cold: bool, clipped: int, floored: bool)`; a module-level
    `_add_row_diagnostics(batch: Batch, row: int, diag: DynamicDiagnostics) -> None` adds the counts into
    `batch.cold_covariates[row]`, `batch.clipped_covariates[row]`, `batch.floored_covariates[row]` so the
    performance branch's `BatchBuilder` calls the same code. Rules, in order:
    (0) validity = `np.isfinite(source)` everywhere (NaN and ±inf are missing); every later step
    uses only valid values and the validity mask is preserved through the transform.
    (1) `past` = valid values before `origin` (the WHOLE past, not the cropped context); if
    `past.size == 0` and the slice `[start, stop)` holds at least one valid value, the column is
    **cold**: return zeros and an all-False mask for the whole slice, `cold=True`. An entirely
    missing column stays as today (zeros, all-False mask, `cold=False`); past-only columns cannot be
    cold (`stop == origin`). (2) Indicator classification and standardization as today, on valid
    past values, with the standardization scale `max(past.std(), REL_STD_FLOOR * max(abs(past)))`
    when `past.std() > 0` (an exactly constant past keeps today's `or 1.0` rule), where
    `REL_STD_FLOOR = 1e-3`; a sample-column whose scale was raised by the floor counts as
    `floored` (third field of `DynamicDiagnostics`). (3) `values = np.clip(values, -COVARIATE_BOUND, COVARIATE_BOUND)` applied AFTER
    both branches (an indicator past can be followed by a large non-binary future value), on
    float64 before the float32 conversion so in-range values keep their bytes; `clipped` = number
    of valid entries moved by the clip. (4) invalid entries → value 0, mask False.
  - `Batch.cold_covariates`, `Batch.clipped_covariates`, `Batch.floored_covariates`: `Tensor (B,) int64`
    per-row counts, filled by `make_batch` through `_add_row_diagnostics`; `Batch.to/clone/_map_tensors`
    cover them generically.
  - Trainer: cumulative `timings["cold_covariate_columns"]` (sample-column occurrences, i.e. the
    sum of `Batch.cold_covariates`), `timings["floored_covariate_columns"]` and
    `timings["clipped_covariate_values"]` (persisted like the
    other timings, restored on resume) and the same keys in `FitResult.timings` → `run.json`
    `timings.cumulative`; `progress.json` carries them. Update the exact timings-dict assertion in
    `tests/test_core/test_training.py` (~line 811) and the explicit diagnostics payload in
    `tests/test_bench/test_fev_run.py` (~line 858).
  - Predictor diagnostics: `forecast_with_diagnostics` adds `cold_covariate_columns` and
    `clipped_covariate_values` (summed over the call); `forecast_per_item_origins` sums them.
    NPZ: the two keys are written by new runs but are OPTIONAL on read: `prediction_diagnostics`
    returns the existing four fields for a legacy NPZ and `None` only when one of those is
    missing; absent counters are `None` and `extra_info` sums per key over windows where present
    (never a manufactured 0). Regression test: NPZ with today's four fields and neither counter
    keeps its inference time, fallbacks and device on resume.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_data/test_sampler.py
def test_cold_known_covariate_is_masked_off_in_context_and_future(ragged_panel_with_late_covariate):
    # a known covariate NaN before position 30 with raw values 1e5 from 30 on; origin 28, H 4 → the
    # slice [start, 32) holds valid values while the past has none → cold
    batch = make_batch(panel, [(0, 28)], 4, 32, encoder)
    col = N_CALENDAR + 0
    assert not batch.enc_cov_mask[0, :, col].any() and not batch.dec_cov_mask[0, :, col].any()
    assert batch.dec_cov[0, :, col].abs().max() == 0 and batch.cold_covariates[0] == 1
    batch2 = make_batch(panel, [(0, 34)], 4, 32, encoder)      # past has observed values → not cold
    assert batch2.cold_covariates[0] == 0 and batch2.dec_cov_mask[0, :, col].all()


def test_tiny_std_past_is_floored_and_counted(panel_with_tiny_std_covariate):
    # past: 100.0 repeated with one 100.0 + 1e-6 (std ~1e-7, max|past| 100 → scale 0.1); future: 110.0 →
    # z = 100 (not 1e8), floored_covariates[0] == 1, nothing clipped
    batch = make_batch(panel, [(0, origin)], 4, 32, encoder)
    assert abs(batch.dec_cov[0, 0, col] - 100.0) < 1e-3 and batch.floored_covariates[0] == 1
    assert batch.clipped_covariates[0] == 0


def test_standardized_covariate_is_bounded_and_counted(panel_with_zero_mean_tiny_std_covariate):
    # past: 0.0 repeated with one 1e-9 (max|past| 1e-9 → floor negligible); future: 0.5 → z huge → clipped
    batch = make_batch(panel, [(0, origin)], 4, 32, encoder)
    assert batch.dec_cov[0, :, col].abs().max() == COVARIATE_BOUND and batch.clipped_covariates[0] == 4
    assert batch.enc_cov.abs().max() <= COVARIATE_BOUND


def test_ordinary_covariates_are_byte_identical_to_the_previous_helper(weekly_panel):
    # golden: a batch built on the fixture before this change (store the tensors in the test as literals
    # for two rows, or compare against tests/helpers/reference values if present) is unchanged
    ...


def test_inf_in_a_covariate_is_missing(panel):
    # inf inside the past → excluded from mean/std and from indicator classification; mask False, value 0 at
    # that position; inf in the future with an indicator past → missing, never clipped to the bound
    ...


def test_existing_all_missing_past_expectation_is_updated():
    # tests/test_data/test_sampler.py ~line 185 currently expects the identity transform for an all-missing
    # past with future values; under the cold rule those futures are masked off — update that assertion


def test_fp16_cast_inputs_and_model_pass_stay_finite_after_the_guards(panel_with_cold_and_tiny_std):
    batch = make_batch(...)
    for name in ("target", "enc_cov", "dec_cov", "labels"):
        assert torch.isfinite(getattr(batch, name).half()).all()
    # representability is not model safety: also run forward + pinball_loss + backward in fp32 on the
    # tiny model and assert finite outputs, loss and gradients; where torch supports it, repeat the forward
    # under torch.autocast("cpu", dtype=torch.bfloat16) and record the result (T4 fp16 is checked by the
    # Colab rerun in Task 3, not here)
```

```python
# tests/test_bench/test_fev_run.py
def test_item_with_cold_known_covariate_at_its_origin_gets_finite_forecasts(tmp_path):
    # fixture: known covariate NaN for the first 36 steps, values 1e5 after; ragged lengths so one item's
    # origin precedes the covariate's first value → cold at inference; run_fev completes; that item's NPZ
    # row is finite; diagnostics cold_covariate_columns >= 1 in the window NPZ and in summaries extra_info
    ...
```

Trainer/run tests: `timings.cumulative.cold_covariate_columns` and `clipped_covariate_values` present
in `run.json` (0 for the aligned fixture), restored across `--stop-after` → `--resume`.

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_data/test_sampler.py -k "cold or bounded or inf or fp16" -v`.

- [ ] **Step 3: Implement** per the Interfaces; keep `_dynamic_values`'s existing docstring rule
  ("future values cannot change the context transform") and extend it with the cold rule.

- [ ] **Step 4: Full suite + ruff.**

- [ ] **Step 5: Commit**

```bash
git add demand_forecast tests
git commit -m "fix: mask off cold covariates, bound standardized covariates, count both"
```

### Task 2: Fail-fast on consecutive non-finite losses

**Files:**
- Modify: `demand_forecast/core/training.py` (`NONFINITE_STOP_AFTER = 20`, `Trainer.fit` loop,
  `_write_progress`, `FitResult.stop_reason` values, `NonFiniteTraining`)
- Modify: `demand_forecast/core/run.py` (`train_panel` raises), `demand_forecast/cli.py` (exit 1),
  `demand_forecast/bench/run.py` (propagates; no summary row)
- Test: `tests/test_core/test_training.py`, `tests/test_core/test_run.py`, `tests/test_cli.py`, `tests/test_bench/test_fev_run.py`

**Interfaces:**
- Produces: `NONFINITE_STOP_AFTER: int = 20`; `class NonFiniteTraining(RuntimeError)` with
  message `"Training stopped after 20 consecutive non-finite losses at step {step}; checkpoint {path}"`;
  `Trainer.consecutive_nonfinite: int` and `Trainer.terminal_reason: str | None`, both checkpointed
  and restored (legacy checkpoints without the fields restore as 0 / None).
  Step loop, in this order: compute the loss under autocast; **if the loss is non-finite, do not call
  backward or the optimizer** (zero_grad only; the scheduler still steps and `self.step` still
  increments, so the LR schedule stays aligned with the step count; the cumulative
  `nonfinite_losses` and `consecutive_nonfinite` increment); a finite loss resets
  `consecutive_nonfinite` to 0 and proceeds as today. After the step's `history.append(record)`
  and the diagnostics updates, if `consecutive_nonfinite >= NONFINITE_STOP_AFTER`: set
  `terminal_reason = "nonfinite_losses"`, checkpoint to `checkpoint_dir / "latest.pt"` (when
  `checkpoint_dir` is None, resolve `Path("checkpoints")` under the current working directory and log
  it; never skip the checkpoint), and `return self._result("nonfinite_losses")`. This failure check
  runs BEFORE the evaluation, the normal-completion and the `--stop-after` handling of the same step,
  so failure wins when boundaries coincide. `_result(reason)` passes `reason` to `_write_progress`,
  so `progress.json["stop_reason"]` is the published reason (null only while running).
  Entry guard: after `resume()` restores state and before the loop, if `terminal_reason ==
  "nonfinite_losses"` (or the restored counter is already at threshold), return
  `self._result("nonfinite_losses")` without training, so the checkpoint itself is the failed-task
  marker and `--resume` never retrains the task. `train_panel`: `if result.stop_reason ==
  "nonfinite_losses": raise NonFiniteTraining(...)` before writing the report or the bundle. CLI
  `train` and `fev run` catch it like `InterruptedRun` (message to stderr, exit 1). `run_fev` lets it
  propagate (no summary row for the task). Checkpointing finite parameters that never changed is
  correct: the checkpoint preserves model, scaler, scheduler, RNG and sampler state as evidence.
  Implementation details to preserve: on an invalid-loss step skip ALL scaler operations including
  `update()` (an update without recorded gradient checks can assert; unchanged scaler state is
  checkpointable); the scheduler still advances (a skipped first optimizer update triggers PyTorch's
  ordering warning: document and test that expected warning); skipped steps stay in the shared
  timing/history/diagnostic accounting (no early `continue`); initialise the new fields before any
  early return; place the terminal entry guard before the `no_training_data` return; initialise
  `progress_path` when resolving the fallback checkpoint directory; cover the fallback-directory and
  legacy-resume paths with tests.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_core/test_training.py
def test_fit_stops_after_consecutive_nonfinite_losses_with_checkpoint_and_progress(tmp_path, monkeypatch):
    # monkeypatch demand_forecast.core.training.pinball_loss to return (torch.tensor(float("nan"), requires_grad=True) * pred.sum(), n)
    result = trainer.fit(panel, plan, forecast_cfg, encoder, checkpoint_dir=tmp_path / "ck")
    assert result.stop_reason == "nonfinite_losses" and result.steps == 20
    assert result.timings["nonfinite_losses"] == 20
    progress = json.loads((tmp_path / "ck" / "progress.json").read_text())
    assert progress["stop_reason"] == "nonfinite_losses" and progress["step"] == 20
    assert (tmp_path / "ck" / "latest.pt").exists()


def test_finite_loss_resets_the_consecutive_counter_and_parameters_are_not_poisoned(...):
    # scaler disabled (CPU): alternate 19 NaN then 1 finite then 19 NaN ... never stops within max_steps;
    # parameters stay finite and change only on finite-loss steps (NaN losses skip backward/optimizer)


def test_resume_after_nonfinite_stop_returns_the_terminal_result_without_training(tmp_path, ...):
    # fit → stop at 20; new Trainer, fit(..., resume=True): returns stop_reason nonfinite_losses at step 20
    # immediately; progress.json stop_reason preserved; history length == 20 in the checkpoint


def test_failure_wins_over_stop_after_and_completion_boundaries(...):
    # max_steps == 20 == threshold → stop_reason nonfinite_losses, not max_steps; stop_after == 20 → same
```

```python
# tests/test_core/test_run.py
def test_train_panel_raises_nonfinite_training_without_bundle_or_report(tmp_path, ...):
    with pytest.raises(NonFiniteTraining, match="20 consecutive non-finite losses"):
        train_panel(...)
    assert not (out / "bundle").exists() and not (out / "run.json").exists()
    assert (out / "checkpoints/latest.pt").exists()
```

```python
# tests/test_cli.py / tests/test_bench/test_fev_run.py
def test_cli_fev_run_exits_1_on_nonfinite_training_and_writes_no_summary_row(...): ...
```

- [ ] **Step 2–5:** run (fail) → implement → full suite + ruff → commit
  `git commit -m "feat: stop training after 20 consecutive non-finite losses; NonFiniteTraining"`.

### Task 3: Report gate, protocol amendments, prompts

**Files:**
- Modify: `scripts/fev_report.py` (completeness issue on `nonfinite_losses > 0`; Training table columns
  `nonfinite_losses`, `cold_covariate_columns`, `clipped_covariate_values` when present)
- Modify: `docs/protocol.md` (dated amendment block), `docs/colab/prompts/01-fev-run-v2.md` (§6/§7 gate),
  `docs/colab/prompts/02-quick-validation.md` (§3 gate; new §5 rerun of walmart and rohlik_sales_1W at the fix SHA)
- Test: `tests/test_bench/test_fev_report.py`

**Interfaces:**
- Produces: `fev_report.py`: for each task the gate value is `report["fit_result"]["nonfinite_losses"]`
  when present, else the already-required `timings.cumulative.nonfinite_losses` (a missing cumulative
  count remains the existing completeness issue); `value > 0` →
  `issues.append(f"{task}: {n} non-finite training losses")`. Only the new covariate counters are
  optional (`"unknown"` in the table, no issue). Tests cover both legacy shapes.
- `docs/colab.md`: one sentence on the failed-run artifacts (checkpoint with `progress.json`
  `stop_reason: nonfinite_losses`, no bundle, no summary row).
- Protocol amendment block (append to "Selection, short histories and fallbacks" or a new
  "Amendments" section), dated 2026-11-01, user decisions D1a/D2/D3 quoted:
  > **Amendment 2026-11-01 (non-finite training inputs).** (1) A dynamic covariate with no finite
  > value before a sample's origin but at least one finite value in the sample's slice is unobserved
  > for that sample in both context and future (mask off, value 0); an entirely missing slice counts
  > zero; the count of such sample-columns is recorded (`cold_covariate_columns`). Future values never
  > shape a sample's context transform, as before. (2) Dynamic covariate values are bounded to ±1e3
  > after either the indicator pass-through or the standardization; clipped entries are counted
  > (`clipped_covariate_values`); non-finite source values (NaN, ±inf) are missing. (3) Training stops after 20 consecutive non-finite losses with `stop_reason
  > nonfinite_losses`, after checkpointing; such a task has no bundle and no summary row and the run
  > is failed. (4) A task with `fit_result.nonfinite_losses > 0` fails the report's completeness check.
  > (5) `stop_reason` values: `max_steps`, `early_stop`, `exception_budget`, `no_training_data`,
  > `interrupted`, `nonfinite_losses`. The standard configuration is unchanged; identities are unchanged. Batch tensors are unchanged
  > for every sample without a cold column, without a floored or clipped value and without
  > non-finite source values.
  > D4 (target-frame floor): not triggered (diagnostic 2026-11-01: target scale minima 0.50–1.0,
  > no target-frame overflow). (6) Standardization scale for dynamic covariates is
  > `max(std, 1e-3 × max|past|)` when std > 0; floored sample-columns are counted
  > (`floored_covariate_columns`).
- Prompt 01 rev 2 §6/§7 and prompt 02 §3/§4: "every task must show `fit_result.nonfinite_losses == 0`;
  a `nonfinite_losses` stop or a positive count is a failed run, not a configuration question".
- Prompt 02 new "## 5. Rerun of walmart and rohlik_sales_1W after the fix": at the fix SHA, a new run dir
  `runs/02b-quick-v2-<SHA>/`, §1–§2 as before, seasonal naive for `--tasks walmart,rohlik_sales_1W`,
  then the quick command with `--tasks walmart,rohlik_sales_1W --model-name demand-forecast-v2-quick`,
  gates: `nonfinite_losses == 0`, `cold_covariate_columns`, `floored_covariate_columns` and
  `clipped_covariate_values` recorded,
  `stop_reason` early_stop or max_steps, GradScaler scale reported diagnostically (not gated: two
  legitimate overflow back-offs give 16384 and growth needs 2,000 steps), finite parameters and
  predictions, and evidence of actual optimizer updates from the checkpoint's optimizer state; report with
  `--tasks walmart,rohlik_sales_1W --model-name demand-forecast-v2-quick`; §2's seven-row/seven-skip
  expectations become two for this rerun; the five clean tasks are not rerun. This rerun is also the
  T4 fp16 acceptance check for the 1e3 bound.

- [ ] **Steps:** tests (report issue + columns; legacy run.json without keys) → implement → docs → full suite + ruff → commit
  `git commit -m "docs/feat: non-finite loss gate in fev_report, protocol amendment 2026-11-01, prompt gates and rerun"`.

### Task 4 (conditional on the Colab diagnostic): relative floor for the target frame — NOT TRIGGERED

Diagnostic 2026-11-01: rohlik target scales 0.50–1.0 minimum and 983–1407 maximum, walmart scale not
small, no target-frame overflow in the fp16 forward. The protocol amendment block records
"D4 not triggered" with these numbers. The rest of this section is kept for the record of the
trigger rule: trigger only on demonstrated overflow in the ACTUAL computation (labels stay float32 and the pinball
loss is computed in float32; a half-cast label or a small scale alone is not evidence): the
diagnostic's fp16 forward on real walmart/rohlik batches must show non-finite outputs attributable
to the normalized target/context inputs after Task 1's guards. If triggered: replace `min_scale=1e-6`
in `sample_frame` by `max(1e-6, RELATIVE_FLOOR × max|context − loc|)` with a named constant, keep the
seasonal-scale eligibility rule, handle empty/missing context explicitly, add a protocol amendment
line and fixture tests. Until the Colab result arrives the amendment block records D4 as
**pending**, never as "not triggered". The user's decision: "Only if data shows it".
