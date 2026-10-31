# Quick-validation run (step cap, validation subsample) Implementation Plan — rev 2

Codex plan review (rev 1): changes needed; seven findings accepted: identity pin belongs to the tiny
test settings (frozen-config identity pinned separately); exception tasks stop with `exception_budget`;
concrete quick-resume test; report composed with the label and counts; prompt 02 enumerates its
overrides of prompt 01; `tests/test_config.py:82` and `tests/test_cli.py:254` adapt to the two new
fields; dedicated fixture for the train_panel test; colab.md §4 link.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A labelled quick-validation configuration (`configs/fev-v2-quick.yaml`) that runs the
seven fev weekly tasks end to end on one free Colab session in about an hour: a flat cap of
300 optimizer steps, evaluation every 50 steps, patience 2, and a seeded subsample of at most
1,024 validation items. The frozen protocol configuration and its identities are untouched.

**Architecture:** Two optional `TrainingConfig` fields with `None` defaults. `max_steps_cap`
is applied by `resolve_budget` after the frozen clip and before the `eval_every` rule.
`val_items_max` subsamples the planner's eligible validation items with a seeded permutation in
`train_panel` (`subsample_validation`), leaving `train_upto`, `exception` and `fit_ids`
unchanged. Both appear in the run identity through `settings`, so quick runs can never be
mistaken for protocol runs. Prompt 02 drives the run on Colab and labels every number.

**Tech Stack:** Python 3.10+, pydantic settings, numpy, torch, fev 0.10.0, pytest.

**Spec:** `docs/protocol.md` (binding; this plan adds one paragraph). User decision 2026-10-31
(relayed from the Colab session): the 11.4 h frozen run is too long for free Colab; run a
labelled quick validation now ("~300 max steps with a smart early stop"), the frozen protocol
later on EC2. The "smart early stop" is the existing inner-SQL early stopping with a small
patience; tasks under the short-history exception (rossmann_1W: `n_val_items 0`) have no
validation and run to the cap, as the protocol prescribes.

## Global Constraints

- `configs/fev-v2-standard.yaml` unchanged; `tests/test_config.py::test_frozen_fev_standard_config_matches_protocol`
  passes unchanged; `resolve_budget` with `max_steps_cap=None` returns exactly today's values
  (worked examples `(3000, 118, 200) -> (500, 100)`, `(3000, 30490, 200) -> (12000, 200)`,
  `(60, 3, 200) -> (60, 12)`).
- With both new fields `None`, `train_panel` writes the same `run.json` as today plus the new
  top-level keys (`val_items_max: null`, `n_val_items_eligible`) and the two nested null fields
  inside the embedded `settings.training` dump; identities are unchanged because the identity
  payload pops exactly those two keys when `None` (never `exclude_none=True`, which would drop
  existing null fields). Skipped legacy artifacts stay unchanged and lack the new keys.
- No `__version__` bump. No change to `demand_forecast/bench`, the sampler, the evaluator or
  the model.
- `ruff check .` and `ruff format --check .` clean; full suite passes in the Colab-stack venv.
- Docs edited: `docs/protocol.md` (one paragraph), new `docs/colab/prompts/02-quick-validation.md`.

## Review Focus

1. **Identity stability for existing runs.** `run_identity` hashes `settings.model_dump(mode="json")`.
   New fields would alter every existing identity (rev-1/rev-2 artifacts, the d683bf7
   compatibility evidence). The identity payload must therefore drop the two new keys when they
   are `None`: `train_panel`/`run_identity` build the settings dump and `pop` `training.max_steps_cap`
   and `training.val_items_max` when `None`. Tests: (a) the aligned fixture under
   `tiny_settings(base_steps=20, d_model=16)` still hashes to
   `0109546d94a7e19ca25a3d8e427e46af981b5f04ee5a5ca772cc610d0397e03a` (recorded at 67d4bd7 by the
   compatibility check); (b) the same fixture under `configs/fev-v2-standard.yaml` hashes to the
   value the worker computes once with the 67d4bd7 code (`git archive 67d4bd7` into a temp dir,
   same recipe; Codex's read-only probe reported `576741a4…c0a0250`, to be confirmed) and pins it.
2. **Cap after clip, eval rule after cap:** `(3000, 30490, 200, cap=300) -> (300, 60)` and with
   `eval_every=50 -> (300, 50)`; `(3000, 118, 200, cap=300) -> (300, 60)`; cap larger than the
   clip is a no-op: `(3000, 118, 200, cap=10000) -> (500, 100)`.
3. **Subsample determinism and ordering:** same seed → same sorted items; `len <= max`;
   unchanged when eligible count ≤ max; never changes `exception`/`train_upto`/`fit_ids`.
4. **Resume with a quick config:** extend the existing resume-parity e2e
   (`tests/test_bench/test_fev_run.py` around line 117) with an active cap and actual subsampling:
   interrupted-then-resumed predictions equal the uninterrupted run, the subsampled indices are
   identical across sessions, the completed run skips idempotently, and changing EITHER new field
   raises `run identity mismatch`. Note `--stop-after` yields exit 1 only when it fires before
   completion or early stop (completion takes precedence).
5. **Report truthfulness:** `run.json` records `val_items_max`, `n_val_items` (after) and
   `n_val_items_eligible` (before); `fev_report.py --model-name demand-forecast-v2-quick` works.

---

### Task 1: `max_steps_cap`, `val_items_max`, quick config

**Files:**
- Modify: `demand_forecast/config/settings.py` (`TrainingConfig`)
- Modify: `demand_forecast/core/training.py` (`TrainConfig`, `resolve_budget`, Trainer call site)
- Modify: `demand_forecast/core/validation.py` (`subsample_validation`)
- Modify: `demand_forecast/core/run.py` (`run_identity` payload, `train_panel` plan/report)
- Create: `configs/fev-v2-quick.yaml`
- Modify: `scripts/fev_report.py` (training table gains `val_items_max` and `n_val_items_eligible` when present in `run.json`)
- Test: `tests/test_core/test_training.py`, `tests/test_core/test_validation.py`,
  `tests/test_core/test_run.py`, `tests/test_config.py` (also line ~82: exclude `val_items_max`
  from the TrainingConfig/TrainConfig comparison), `tests/test_cli.py` (line ~254: expected identity
  payload drops the two keys when `None`), `tests/test_bench/test_fev_run.py` (quick resume e2e),
  `tests/test_bench/test_fev_report.py` (new columns)

**Interfaces:**
- Produces:
  - `TrainingConfig.max_steps_cap: int | None = Field(None, ge=1)`, `TrainingConfig.val_items_max: int | None = Field(None, ge=1)`.
  - `TrainConfig.max_steps_cap: int | None = None` (dataclass); `train_panel` builds `TrainConfig(**settings.training.model_dump(exclude={"refit", "min_train_history", "val_items_max"}))`.
  - `resolve_budget(base_steps, n_series, eval_every, max_steps_cap: int | None = None) -> tuple[int, int]`:
    frozen clip, then `max_steps = min(max_steps, max_steps_cap)` when given, then
    `eval_every = min(eval_every, max(max_steps // 5, 1))`; `ValueError` on `max_steps_cap < 1`.
  - `subsample_validation(plan: ValidationPlan, max_items: int | None, seed: int) -> ValidationPlan`:
    returns `plan` itself when `max_items is None` or `len(plan.val_items) <= max_items`; else a
    copy with `val_items = np.sort(np.random.default_rng(seed).choice(plan.val_items, size=max_items, replace=False))`.
  - `run_identity(...)`: settings dump with `training.max_steps_cap`/`training.val_items_max`
    removed when `None` (helper `_identity_settings(settings) -> dict`).
  - `train_panel`: `eligible = plan; plan = subsample_validation(plan, settings.training.val_items_max, settings.seed)`;
    report gains `"val_items_max"`, `"n_val_items_eligible": len(eligible.val_items)`;
    `plan:` INFO line gains `val_items=%d/%d` (after/eligible); `n_val_items` keeps meaning
    "items evaluated".
  - `configs/fev-v2-quick.yaml`: byte-for-byte the frozen file except the header comment
    ("Quick validation of the pipeline, NOT the protocol: scores from this config must never
    inform a configuration decision") and `training`: `eval_every: 50`, `patience: 2`,
    `checkpoint_every: 50`, `max_steps_cap: 300`, `val_items_max: 1024`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_core/test_training.py
@pytest.mark.parametrize(
    "args, expected",
    [((3000, 118, 200), (500, 100)), ((3000, 30490, 200), (12000, 200)), ((60, 3, 200), (60, 12)),
     ((3000, 30490, 200, 300), (300, 60)), ((3000, 30490, 50, 300), (300, 50)),
     ((3000, 118, 200, 300), (300, 60)), ((3000, 118, 200, 10000), (500, 100))],
)
def test_resolve_budget_cap_applies_after_clip_and_before_eval_rule(args, expected):
    assert resolve_budget(*args) == expected


def test_resolve_budget_rejects_nonpositive_cap():
    with pytest.raises(ValueError):
        resolve_budget(3000, 118, 200, 0)
```

```python
# tests/test_core/test_validation.py
def test_subsample_validation_is_seeded_sorted_and_noop_when_small(plan_with_12_val_items):
    plan = plan_with_12_val_items
    assert subsample_validation(plan, None, 0) is plan and subsample_validation(plan, 12, 0) is plan
    a, b = subsample_validation(plan, 5, 0), subsample_validation(plan, 5, 0)
    assert a.val_items.tolist() == b.val_items.tolist() and len(a.val_items) == 5
    assert a.val_items.tolist() == sorted(a.val_items.tolist())
    assert set(a.val_items) <= set(plan.val_items)
    assert subsample_validation(plan, 5, 1).val_items.tolist() != a.val_items.tolist()
    assert (a.train_upto, a.val_origin, a.exception, a.fit_ids) == (plan.train_upto, plan.val_origin, plan.exception, plan.fit_ids)
```

```python
# tests/test_core/test_run.py
def test_quick_fields_subsample_validation_and_are_reported(tmp_path, quick_panel_and_settings):
    # dedicated fixture: >= 4 eligible validation items and an uncapped budget above 7
    # (tiny_panel_and_settings has three series and base_steps=2, which cannot serve here)
    panel, origin, settings = quick_panel_and_settings
    quick = settings.model_copy(deep=True); quick.training.val_items_max = 2; quick.training.max_steps_cap = 7
    bundle = train_panel(panel, origin, quick, tmp_path / "q", dataset_fingerprint="fp")
    report = json.loads((tmp_path / "q" / "run.json").read_text())
    assert report["val_items_max"] == 2 and report["n_val_items"] == 2 and report["n_val_items_eligible"] >= 4
    assert report["resolved_budget"]["max_steps"] == 7 and report["fit_result"]["steps"] <= 7


def test_identity_ignores_unset_quick_fields_and_changes_when_set(settings_fixture):
    base = run_identity(settings_fixture, 0, "fp", "2.0.0", "2023-01-01", None)
    dumped = settings_fixture.model_dump(mode="json")
    assert "max_steps_cap" in dumped["training"]                   # the field exists on the model ...
    payload = _identity_settings(settings_fixture)
    assert "max_steps_cap" not in payload["training"] and "val_items_max" not in payload["training"]  # ... but not in the hash
    quick = settings_fixture.model_copy(deep=True); quick.training.max_steps_cap = 300
    assert run_identity(quick, 0, "fp", "2.0.0", "2023-01-01", None) != base
```

Plus two identity pins in `tests/test_bench/test_fev_run.py`: (a) aligned fixture +
`tiny_settings(base_steps=20, d_model=16)` → `0109546d94a7e19ca25a3d8e427e46af981b5f04ee5a5ca772cc610d0397e03a`;
(b) aligned fixture + `Settings.from_yaml("configs/fev-v2-standard.yaml")` → the value computed once
with the 67d4bd7 code (see Review Focus 1). Both use `settings_for_task(spec, settings, task.freq)`,
the window-0 origin and `spec.task_definition()` exactly as the e2e test does.

```python
# tests/test_config.py
def test_quick_config_differs_from_frozen_only_in_declared_fields():
    frozen = Settings.from_yaml(root / "configs/fev-v2-standard.yaml").model_dump()
    quick = Settings.from_yaml(root / "configs/fev-v2-quick.yaml").model_dump()
    assert quick["training"].pop("max_steps_cap") == 300 and frozen["training"].pop("max_steps_cap") is None
    assert quick["training"].pop("val_items_max") == 1024 and frozen["training"].pop("val_items_max") is None
    for key in ("eval_every", "patience", "checkpoint_every"):
        assert (frozen["training"].pop(key), quick["training"].pop(key)) == ({"eval_every": (200, 50), "patience": (5, 2), "checkpoint_every": (200, 50)}[key])
    assert quick == frozen


def test_training_config_rejects_nonpositive_quick_fields():
    with pytest.raises(ValidationError):
        TrainingConfig(max_steps_cap=0)
    with pytest.raises(ValidationError):
        TrainingConfig(val_items_max=0)
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_core tests/test_config.py -k "cap or subsample or quick or identity" -v`.

- [ ] **Step 3: Implement** the interfaces. In `run_identity`, replace
  `settings.model_dump(mode="json")` with `_identity_settings(settings)` that pops the two keys
  when `None`; keep everything else byte-identical. In `Trainer.fit`, pass
  `self.cfg.max_steps_cap` to `resolve_budget`.

- [ ] **Step 4: Full suite + ruff** — `pytest -q --no-cov -p no:cacheprovider`; `ruff check . && ruff format --check .`.

- [ ] **Step 5: Commit**

```bash
git add demand_forecast configs/fev-v2-quick.yaml tests
git commit -m "feat: quick-validation knobs (max_steps_cap, val_items_max) and configs/fev-v2-quick.yaml"
```

### Task 2: Protocol paragraph and prompt 02

**Files:**
- Modify: `docs/protocol.md` ("Selection, short histories and fallbacks": one paragraph after the item-specific origins paragraph)
- Create: `docs/colab/prompts/02-quick-validation.md`

- [ ] **Step 1: Protocol paragraph**
  > **Quick validation is not the protocol.** `configs/fev-v2-quick.yaml` caps every task at 300
  > optimizer steps (`max_steps_cap`), evaluates every 50 steps with patience 2 and scores at
  > most 1,024 seeded validation items (`val_items_max`). It exists to exercise the pipeline on
  > the seven tasks within one free Colab session. Its summaries carry `model_name
  > demand-forecast-v2-quick` and a different run identity; its benchmark scores are reported
  > only as a pipeline check and must never inform a configuration, early-stopping, tuning or
  > ablation decision. The frozen configuration above is unchanged by these fields (both `null`).

- [ ] **Step 2: Prompt 02** (`docs/colab/prompts/02-quick-validation.md`), structured like prompt 01 rev 2 but short:
  0. Rules: labelled quick validation, not the protocol; the same launcher/poller/exit-1
     patterns; T4 + runtime 2026.07; fresh-VM restart after the first editable install; score
     exposure: quick scores may be read in §4, and the report must carry the label sentence from
     the protocol paragraph verbatim; `FEV_RUN_DIR=$FEV_DRIVE_ROOT/runs/02-quick-v2-<SHA>`,
     `TRIAL` unused; provenance under `provenance-$FEV_SHA/`.
  1. Setup: prompt 01 rev 2 §1–§2 by reference WITH these overrides stated explicitly: `FEV_RUN_DIR`
     is the quick directory and `TRIAL` is not used; metadata `scope: "quick_validation_v2"`; the
     dataset revision must equal `d3e7e720f4158eab03ebcb021f1cce779cd92acc` exactly (a different
     revision blocks the run, unlike prompt 01's fresh-experiment allowance); prompt 01 §6's rerun
     policy does not apply; the A–D evidence being reused is the rev-2 directory
     `runs/01-fev-run-v2-67d4bd78…/` (cite it, do not rerun A–D); keep the restart rule, the lock,
     the online revision/parquet verification, the launcher, the poller and the RSS/VRAM gates.
     Copy `configs/fev-v2-quick.yaml` into `provenance-$FEV_SHA/` and export the notebook to the run dir.
  2. Seasonal naive into this run dir: `demand-forecast fev baseline --tasks weekly --out-dir "$FEV_RUN_DIR"`
     (seven rows; prompt 01 §3's equality check against prompt 00 to 6 s.f.).
  3. The run, detached with the launcher, `/usr/bin/time -v` and the poller:
     `demand-forecast fev run --tasks weekly --config configs/fev-v2-quick.yaml --policy first_window --out-dir "$FEV_RUN_DIR" --resume --model-name demand-forecast-v2-quick`
     Before launching, state in the notebook that the quick configuration is fixed and that no
     score-driven variant selection follows this run. Expected per task in the log:
     `fit: max_steps=300 eval_every=50 … val_items=<n>` with
     `n <= 1024` (rossmann: `val_items=0`, exception, runs to 300 with `stop_reason=exception_budget`);
     other tasks end with `early_stop` or `max_steps`;
     projected wall about 1 h (unverified projection; includes baseline, inference and reporting), dominated
     by panel builds (rev-2 measurements: m5 ≈ 600 s, hermes ≈ 205 s, favorita ≈ 25 s × 10 windows).
     `/usr/bin/time -v` around the whole command gives whole-command wall and peak RSS; per-task wall
     comes from the log timestamps and per-task peaks are not measured. Gates: RSS/VRAM as prompt 01 §1.
     After a disconnect: §1 then the same command (`--resume`).
  4. Report: `python scripts/fev_report.py --run-dir "$FEV_RUN_DIR" --tasks weekly --baseline seasonal_naive --model-name demand-forecast-v2-quick --out "$FEV_RUN_DIR/report-tables.md"`
     (generated tables), then compose `$FEV_RUN_DIR/report.md` by hand: first line the protocol's label
     sentence verbatim, then model name `demand-forecast-v2-quick`, config path and SHA, per task
     `n_val_items`/`n_val_items_eligible` and `val_items_max` read from `run.json`, then the tables.
     Any table shared elsewhere carries the label line. Paste back per task: steps, stop_reason, best_step, `n_val_items`/`n_val_items_eligible`,
     wall, `panel:` build_s, `train_s_per_step`, `eval_s`, inference s, peak RSS/VRAM, SQL ratio
     vs seasonal naive, 80% coverage, fallback counts, `max_distinct_origins`; the dataset
     manifest check; the label sentence; and the measured per-stage timings that the EC2
     performance plan needs (panel build per task, first-evaluation batch-build seconds if the
     log shows them, per-step seconds).

  Also add one sentence to `docs/colab.md` §4 pointing at prompt 02 as the quick, non-protocol run.

- [ ] **Step 3: Commit**

```bash
git add docs/protocol.md docs/colab.md docs/colab/prompts/02-quick-validation.md
git commit -m "docs: quick-validation protocol note and Colab prompt 02"
```
