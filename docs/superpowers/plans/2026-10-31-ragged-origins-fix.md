# Item-specific fev origins (walmart) Implementation Plan — rev 2.1

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `demand-forecast fev run` completes fev windows whose items have different first
future timestamps (fev integer cutoffs over ragged series ends, as in `walmart`), without
changing run identities, training semantics, fitted state or prediction values for windows with
one shared origin.

**Architecture:** `run_fev` derives one origin per item from `future_data`. The latest item
origin `O` is the window origin used for the run identity, the validation plan and training:
the inner validation block is `[O − H·Δ, O)` and the inclusive training cutoff is
`plan.train_upto = O − (H + 1)·Δ` (except under the short-history exception). Items ending
inside the block contribute a partial validation horizon; items ending before it contribute no
validation labels. At inference every item is forecast at its own origin with its own past as
context: the predictor groups items by origin and delegates each group to the existing
single-origin path. The bundle records the earliest item origin as its permissible floor, a
**global** floor (not a per-item guarantee).

**Tech Stack:** Python 3.10+, pandas, numpy, torch, fev 0.10.0, pytest. Bench tests use the
local fixture writer (`tests/helpers/fev_fixture.py`), never real datasets.

**Spec:** `docs/protocol.md` (binding), `docs/superpowers/specs/2026-10-30-fev-bench-v2-design.md`
§4.3 and the forecast-command sentence near line 244 (amended by Task 2). Incident: prompt 01
frozen run at `d683bf7` exited 2 on walmart with `Invalid value: windows with item-specific
origins are not supported` (bench/run.py `window_origin`). fev resolves walmart's absent
`initial_cutoff` to `-horizon` per item (`fev/task.py`, `y[:cutoff]`), and walmart series end on
different weeks. Colab diagnostic (2026-10-31, revision d3e7e720): 11 distinct first-future
timestamps in walmart window 0, all Fridays; 2,899 items at 2012-02-03 and 37 items ending early
between 2011-10-07 and 2012-01-20; no gaps in any past grid; every future is exactly 39
consecutive weeks after the item's own last observation; all early origins lie inside the
shared validation block. Every other task and window has exactly one origin (rohlik window 0
keeps 3,942 of 5,243 series after fev's `min_context_length` filter).

**Information set.** Early-ending items are forecast for calendar periods that other items'
available past (and selection labels) already cover. No item's own future is used: fev truncates
each item at its own cutoff and the loader reads only known covariates from future rows. This
equals the information set fev gives every model and is not an out-of-time test for those items.

Codex plan review (rev 1.1): changes needed; all eight findings accepted and folded in below.
Scoped re-review (rev 2): three corrections accepted — `plan.val_items` are integer positions;
`summaries.csv` byte preservation holds only when every task skips (values of existing rows are
preserved otherwise, legacy rows get NaN in new columns); `commit.txt` also goes under the
provenance folder.

## Global Constraints

- No `__version__` bump and no change to `configs/fev-v2-standard.yaml`: the run identity
  payload (`settings`, `seed`, `dataset_fingerprint`, `package_version`, `origin`,
  `task_definition`) is unchanged, so shared-origin identities are equal before and after.
- Shared-origin windows: unchanged identities, training semantics, fitted state and prediction
  values under controlled execution. Fresh artifacts may differ only in provenance (producing SHA,
  creation time), timings, and the declared new metadata (`earliest_origin` in `run.json` and in
  the manifest's embedded report, `max_distinct_origins` in summaries). **Existing task
  artifacts reused by `--resume` are never rewritten** (byte preservation), including legacy ones
  that lack the new keys; skipped legacy tasks do not acquire them. `summaries.csv` is rewritten
  only when a task completes: its existing rows keep their values (legacy rows get NaN in new
  columns), and it is byte-preserved when every task skips.
- Never reduce items, horizons or windows; fev's positional prediction order is unchanged.
- No real-data download or training in tests or on the developer Mac; fixtures only.
- `ruff check .` and `ruff format --check .` clean; the full suite passes in a venv with the
  Colab stack (`numpy 2.0.2`, `pandas 2.2.2`, `torch 2.11`, `fev 0.10.0`); tests under
  `tests/test_data` must not import bench-only packages (CI runs them without the extra).
- `print` stays limited to the two existing "Skipping completed task" CLI lines; new runtime
  messages use `logging`. Type imports from `collections.abc`.
- Docs edited in this plan: `docs/protocol.md`, the design spec (§4.3 and the forecast-command
  sentence), `docs/colab.md`, `docs/colab/prompts/01-fev-run-v2.md` (becomes rev 2). Historical
  plans are not edited.

## Review Focus

1. An item whose own origin lies **before the training cutoff** must be forecast at its own
   origin, not rejected by the floor — Task 1 ragged test, lengths `(40, 38, 40, 28, 38, 40)`,
   `H = 4`: series `3` ends 12 steps early (walmart has no such item; the fixture covers it).
2. Grouping by origin must return forecasts in **panel order** — the ragged test asserts
   `ids == item_ids(panel)` and exact equality against each origin group forecast with identical
   batching; singleton comparisons use `atol=1e-5` (PyTorch batching is not bitwise stable).
3. Partial validation horizons: series `1` and `4` end 2 steps inside the block — the ragged test
   asserts `plan.val_items` has five entries (all but series `3`) and `run.json["n_val_items"] == 5`.
4. Legacy compatibility: a run directory in the d683bf7 shape (no new keys) resumes with a skip
   and byte-identical files, including `summaries.csv`.
5. `scripts/fev_report.py` on a ragged run directory with its seasonal-naive row — exit 0 and
   `Complete`.

---

### Task 1: Per-item origins in `run_fev`, grouped inference, earliest permissible origin

**Files:**
- Modify: `demand_forecast/bench/run.py` (`window_origin` → `window_origins`, `resolved_origin`,
  `run_fev` window loop and `extra_info`)
- Modify: `demand_forecast/inference/predictor.py` (new `Predictor.forecast_per_item_origins`)
- Modify: `demand_forecast/core/run.py` (`train_panel(..., earliest_origin=None)`, report key,
  reuse check)
- Modify: `tests/helpers/fev_fixture.py` (`lengths` parameter)
- Test: `tests/test_bench/test_fev_run.py`, `tests/test_core/test_run.py`,
  `tests/test_data/test_sampler.py`, the module holding `Predictor` tests, `tests/test_bench/test_fixture.py`

**Interfaces:**
- Consumes: `panel_from_window(past, future, spec, freq) -> Panel` (per-item `history_end`),
  `Panel(series, roles, freq, seasonal_period, target_support)`, `item_ids(panel)`,
  `Predictor.forecast_with_diagnostics(panel, origin, batch_size=512) -> (ids, arrays, dict)`,
  `run_identity(settings, seed, fp, version, origin, definition)`, `save_bundle(..., earliest_origin=)`,
  `load_bundle(dir).panel_metadata["earliest_origin"]`, `plan_inner_validation(panel, origin, H, min_train_history)`,
  `run_baseline(specs, out_dir)`, `tiny_settings(base_steps=20, d_model=16)` (existing test helper).
- Produces:
  - `window_origins(future_data, id_column: str, timestamp_column: str = "timestamp") -> dict[str, pd.Timestamp]`
    — `str(row[id_column])` → `pd.Timestamp(row[timestamp_column][0])`; `ValueError("empty future rows for item <id>")`
    when a row has no future timestamps; `ValueError("duplicate future series id")` on repeats.
  - `resolved_origin(origins: Mapping[str, pd.Timestamp]) -> pd.Timestamp` — the latest value;
    `ValueError("window has no items")` when empty.
  - `Predictor.forecast_per_item_origins(panel: Panel, origins: Mapping[str, pd.Timestamp], batch_size: int = 512) -> tuple[list[str], np.ndarray, dict]`
    — every panel id needs an origin (`ValueError("missing origin for item <id>")`); items are
    grouped by origin in first-appearance order; each group is forecast with
    `self.forecast_with_diagnostics(Panel([...], panel.roles, panel.freq, panel.seasonal_period, panel.target_support), origin, batch_size)`
    (through `self`, so test monkeypatches apply); results scattered back to panel order;
    diagnostics = sums of `fallback_no_context`, `fallback_undefined_scale`, `n_items`;
    `inference_time_s` measured around the whole call; `device` from the groups; extra key
    `n_origins`. Docstring carries the information-set caveat above.
  - `train_panel(..., earliest_origin: pd.Timestamp | None = None)` — defaults to `origin`;
    on-grid and `<= origin` (`ValueError("earliest_origin after origin")`); stored as the bundle's
    `earliest_origin` and as `report["earliest_origin"]` (ISO string). On completed-bundle reuse
    (the early return), when `earliest_origin` is given and differs from
    `load_bundle(bundle_dir).panel_metadata["earliest_origin"]`, raise
    `ValueError("bundle earliest_origin mismatch")`.
  - `run_fev`: per window `origins = window_origins(future, spec.id_column, spec.timestamp_column)`,
    `origin = resolved_origin(origins)`; window 0 trains with
    `train_panel(..., earliest_origin=min(origins.values()))`; identity unchanged (`origin`);
    inference via `forecast_per_item_origins`; INFO log
    `origins: task=%s window=%d distinct=%d earliest=%s latest=%s`; `extra_info` gains
    `"max_distinct_origins": int` (max over the task's windows). Docstring carries the caveat.
    `window_origin` is removed.
  - `write_fev_fixture(..., lengths: Sequence[int] | None = None)` — per-series lengths, each
    `<= length`, series `i` uses `timestamps[:lengths[i]]`; wrong size or a value above `length`
    raises `ValueError`; `None` keeps today's behaviour.
  - Test helper in `tests/test_bench/test_fev_run.py`: `file_states(directory) -> dict[Path, tuple[bytes, int]]`
    (bytes and `st_mtime_ns` of every file, as the existing e2e test builds inline).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bench/test_fev_run.py (replaces test_window_origin_requires_one_shared_origin_and_accepts_custom_column)
def test_window_origins_are_per_item_and_resolved_origin_is_latest():
    rows = [
        {"item_id": "a", "timestamp": ["2023-01-01", "2023-01-08"]},
        {"item_id": "b", "timestamp": ["2022-12-25", "2023-01-01"]},
    ]
    origins = window_origins(rows, "item_id")
    assert origins == {"a": pd.Timestamp("2023-01-01"), "b": pd.Timestamp("2022-12-25")}
    assert resolved_origin(origins) == pd.Timestamp("2023-01-01")
    assert window_origins([{"id": "a", "date": ["2023-01-01"]}], "id", "date") == {"a": pd.Timestamp("2023-01-01")}
    with pytest.raises(ValueError, match="empty future rows for item"):
        window_origins([{"item_id": "a", "timestamp": []}], "item_id")
    with pytest.raises(ValueError, match="window has no items"):
        resolved_origin({})


RAGGED = (40, 38, 40, 28, 38, 40)   # 3 origins; "1"/"4" end inside the validation block; "3" ends before the cutoff


def test_ragged_ends_forecast_each_item_at_its_own_origin(tmp_path, capsys):
    spec = fixture_spec(write_fev_fixture(tmp_path, lengths=RAGGED), num_windows=1)
    settings = tiny_settings(base_steps=20, d_model=16)
    out = tmp_path / "run"
    summaries = run_fev([spec], settings, out)
    task = spec.to_task()
    past, future = task.get_window(0).get_input_data()
    panel = panel_from_window(past, future, spec, task.freq)
    origins = window_origins(future, spec.id_column, spec.timestamp_column)
    latest, earliest = max(origins.values()), min(origins.values())
    assert len(set(origins.values())) == 3
    plan = plan_inner_validation(panel, latest, spec.horizon, settings.training.min_train_history)
    assert plan.val_items.tolist() == [0, 1, 2, 4, 5] and earliest < plan.train_upto   # positions, not ids
    report = json.loads((out / spec.name / "run.json").read_text())
    assert pd.Timestamp(report["origin"]) == latest and pd.Timestamp(report["earliest_origin"]) == earliest
    assert report["n_val_items"] == 5 and report["fit_result"]["stop_reason"] == "max_steps"
    with np.load(out / spec.name / "window_0.npz", allow_pickle=False) as saved:
        ids, arrays = saved["ids"].tolist(), saved["arrays"]
    assert ids == item_ids(panel)
    predictor = Predictor.from_bundle(out / spec.name / "bundle")
    for origin in sorted(set(origins.values())):                       # exact: identical batching per group
        members = [key for key in ids if origins[key] == origin]
        group = Panel([panel.series[panel.index_of(k)] for k in members], panel.roles, panel.freq, panel.seasonal_period, panel.target_support)
        gids, expected, _ = predictor.forecast_with_diagnostics(group, origin)
        assert gids == members
        np.testing.assert_array_equal(arrays[[ids.index(k) for k in members]], expected)
    single = Panel([panel.series[panel.index_of("3")]], panel.roles, panel.freq, panel.seasonal_period, panel.target_support)
    _, alone, _ = predictor.forecast_with_diagnostics(single, origins["3"])   # before the training cutoff
    np.testing.assert_allclose(arrays[ids.index("3")], alone[0], rtol=0, atol=1e-5)
    assert summaries.loc[0, "num_forecasts"] == len(RAGGED) and summaries.loc[0, "max_distinct_origins"] == 3
    before = file_states(out)
    run_fev([spec], settings, out, resume=True)
    assert f"Skipping completed task {spec.name}" in capsys.readouterr().out
    assert file_states(out) == before


def test_ragged_run_passes_fev_report(tmp_path):
    # Build task YAML + profiles YAML as tests/test_bench/test_fev_report.py does, run run_baseline([spec], out)
    # for the seasonal_naive row, then run_fev, then scripts/fev_report.py --run-dir out --tasks <yaml>
    # --profiles <yaml> --baseline seasonal_naive --out report.md: returncode 0 and "Complete" in report.md.
    ...


def test_resume_accepts_legacy_run_directory_without_new_keys(tmp_path, capsys):
    # Produce a completed aligned run, then strip the new keys to the d683bf7 shape:
    # delete "earliest_origin" from run.json and from bundle/manifest.json["run"] (keep the bundle's
    # panel metadata earliest_origin), drop the max_distinct_origins column from summaries.csv (and the
    # key from its stored extra_info if present). This tests missing-metadata handling, not real
    # cross-revision compatibility (the worker's one-off d683bf7 check covers that).
    # (1) run_fev(resume=True) on that directory alone: skip line, every file byte-identical.
    # (2) mixed: add a second fixture task (new) to the same out dir with resume=True: the legacy task's
    #     files stay byte-identical, its summaries row keeps its values with NaN in max_distinct_origins,
    #     the new task's row has the column filled.
    ...
```

```python
# tests/test_core/test_run.py
def test_train_panel_records_earliest_origin_and_checks_it_on_reuse(tmp_path, tiny_panel_and_settings):
    panel, origin, settings = tiny_panel_and_settings
    step = to_offset(panel.freq)
    bundle = train_panel(panel, origin, settings, tmp_path / "a", dataset_fingerprint="fp")
    assert load_bundle(bundle).panel_metadata["earliest_origin"] == origin.isoformat()
    earlier = origin - 3 * step
    bundle = train_panel(panel, origin, settings, tmp_path / "b", dataset_fingerprint="fp", earliest_origin=earlier)
    assert load_bundle(bundle).panel_metadata["earliest_origin"] == earlier.isoformat()
    assert json.loads((tmp_path / "b" / "run.json").read_text())["earliest_origin"] == earlier.isoformat()
    with pytest.raises(ValueError, match="earliest_origin after origin"):
        train_panel(panel, origin, settings, tmp_path / "c", dataset_fingerprint="fp", earliest_origin=origin + step)
    with pytest.raises(ValueError, match="bundle earliest_origin mismatch"):
        train_panel(panel, origin, settings, tmp_path / "b", dataset_fingerprint="fp", resume=True, earliest_origin=origin)
    assert train_panel(panel, origin, settings, tmp_path / "b", dataset_fingerprint="fp", resume=True, earliest_origin=earlier) == bundle
```

```python
# Predictor test module
def test_forecast_per_item_origins_matches_single_origin_when_shared(bundle_and_panel):
    predictor, panel, origin = bundle_and_panel
    ids, arrays, diag = predictor.forecast_with_diagnostics(panel, origin)
    ids2, arrays2, diag2 = predictor.forecast_per_item_origins(panel, {i: origin for i in ids})
    assert ids2 == ids and np.array_equal(arrays2, arrays)          # same items, same order, same batching
    assert diag2["n_items"] == diag["n_items"] and diag2["n_origins"] == 1 and diag2["device"] == diag["device"]


def test_forecast_per_item_origins_rejects_missing_origin(bundle_and_panel):
    predictor, panel, origin = bundle_and_panel
    with pytest.raises(ValueError, match="missing origin for item"):
        predictor.forecast_per_item_origins(panel, {})
```

```python
# tests/test_data/test_sampler.py — build the ragged panel directly with build_series (no fev import)
def test_ragged_histories_never_produce_labels_past_their_own_end():
    # six weekly series with history lengths RAGGED, H = 4, latest origin at position 40;
    plan = plan_inner_validation(panel, latest, 4, min_train_history)
    origins = build_origins(panel, 4, 1, upto=plan.train_upto)
    for series_index, origin_pos in origins:
        assert origin_pos <= panel.series[series_index].history_length
    batch = make_batch(panel, [tuple(x) for x in origins], 4, 32, encoder, label_upto=plan.train_upto)
    # no label mask is set at a position after the item's own history_end
```

Fixture test module: `write_fev_fixture(tmp_path, lengths=RAGGED)` round-trips through fev with
per-row `len(timestamp) == lengths[i]`; wrong size or a value above `length` raises `ValueError`.
The existing capture wrappers that monkeypatch `Predictor.forecast_with_diagnostics`
(test_fev_run.py around lines 54 and 328) gain `batch_size=512` and forward it.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_bench/test_fev_run.py tests/test_core/test_run.py tests/test_data/test_sampler.py -k "origin or ragged or legacy" -v`
Expected: FAIL (`window_origins`/`forecast_per_item_origins` undefined; `lengths` unexpected).

- [ ] **Step 3: Implement** the interfaces above. `run_fev` keeps the completed-task skip,
  identity checks, diagnostics and canonical timings unchanged; the only new per-window work is
  the origins dict, the INFO line and the grouped forecast. In `train_panel` validate
  `earliest_origin` with `validate_on_grid` next to the origin check and add the reuse comparison
  inside the existing early-return branch.

- [ ] **Step 4: Run the suite**

Run: `pytest -q -p no:cacheprovider --no-cov` in the Colab-stack venv, then `ruff check . && ruff format --check .`
Expected: all pass; the pre-existing shared-origin e2e tests pass unchanged.

- [ ] **Step 5: Commit**

```bash
git add demand_forecast tests
git commit -m "fix: forecast fev items at their own origins; latest origin trains and identifies the window"
```

### Task 2: Profile column and protocol text for item-specific origins

**Files:**
- Modify: `scripts/profile_fev.py` (`PROFILE_COLUMNS`, `profile_task`)
- Modify: `docs/protocol.md` (frozen policy paragraph, profile CSV table and resume note)
- Modify: `docs/superpowers/specs/2026-10-30-fev-bench-v2-design.md` (§4.3; forecast-command sentence)
- Test: `tests/test_bench/test_profile.py`

**Interfaces:**
- Produces: 13th profile column `distinct_origins_first_window` (int) = number of distinct first
  future timestamps across items of fev's first window, from the `future` dataset already fetched
  by `task.get_window(0).get_input_data()`; `0` when fev rejects an empty window. Existing
  twelve-column CSVs are rejected by `--resume` with the existing column-mismatch error and left
  unchanged; the documented remedy is a fresh output path (no migration, no invented counts).

- [ ] **Step 1: Write the failing tests**

```python
def test_profile_counts_distinct_first_window_origins(tmp_path):
    aligned = profile_task(fixture_spec(write_fev_fixture(tmp_path / "a")))
    ragged = profile_task(fixture_spec(write_fev_fixture(tmp_path / "b", lengths=(40, 38, 40, 28, 38, 40)), num_windows=1))
    assert aligned["distinct_origins_first_window"] == 1 and ragged["distinct_origins_first_window"] == 3
    assert PROFILE_COLUMNS[-1] == "distinct_origins_first_window"


def test_resume_rejects_twelve_column_profile_and_leaves_it_unchanged(tmp_path):
    # write a CSV with the previous 12 columns, run the CLI with --resume: non-zero exit, message names the
    # column mismatch, file bytes unchanged
    ...
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_bench/test_profile.py -v`.

- [ ] **Step 3: Implement** the column, then the docs:
  - `docs/protocol.md`, section "Selection, short histories and fallbacks" after the
    `first_window` paragraph, add:
    > **Item-specific origins.** fev integer cutoffs are positions: every item is split at its own
    > position, so items that end on different timestamps have different first future timestamps
    > (walmart: 11 origins, 37 of 2,936 items ending early). The window origin `O` for the run
    > identity, the inner-validation block and training is the latest item origin: the validation
    > block is `[O − H·Δ, O)` and the inclusive training cutoff is `O − (H + 1)·Δ` (except under
    > the short-history exception); items ending inside the block contribute a partial validation
    > horizon and items ending before it contribute none. Every item is forecast at its own first
    > future timestamp with its own past as context. The bundle's earliest permissible origin is the
    > earliest item origin, a global floor. Early-ending items are scored on periods that other
    > items' available past covers; this equals the information set fev gives every model (each
    > item's past is truncated at its own cutoff) and is not an out-of-time test for those items.
    > Shared-origin windows are the special case with one origin. Prompt 00's profile reports
    > `distinct_origins_first_window` per task.
  - Profile CSV table: add the row `distinct_origins_first_window | Number of distinct first future timestamps across the items of fev's first window; 0 for an empty first window.`
    and, in the paragraph above the table, "A profile written before this column exists cannot be
    resumed; use a fresh output path."
  - Design §4.3: replace "For fev, `origin = cutoff + step`" with two definitions: a timestamp
    cutoff gives one origin `cutoff + Δ` for all items; an integer cutoff is a per-item position,
    so origins may differ per item (see the protocol's item-specific origins rule).
  - Design forecast-command sentence (near line 244): "refuses origins earlier than the training
    cutoff" → "refuses origins earlier than the bundle's earliest permissible origin (the earliest
    item origin of the training window)".

- [ ] **Step 4: Run** `pytest tests/test_bench/test_profile.py -q && ruff check scripts tests` — expected PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/profile_fev.py tests/test_bench/test_profile.py docs/protocol.md docs/superpowers/specs/2026-10-30-fev-bench-v2-design.md
git commit -m "feat: profile distinct first-window origins; document item-specific origins"
```

### Task 3: Colab template restart rule and prompt 01 rev 2

**Files:**
- Modify: `docs/colab.md` §2 (install/restart text; keep the restart-check cell) and §2's
  provenance copy line (copy into `$FEV_RUN_DIR/provenance-$FEV_SHA/` instead of the run root)
- Modify: `docs/colab/prompts/01-fev-run-v2.md` → rev 2

No code; acceptance is textual review plus `ruff format --check .`.

- [ ] **Step 1: `docs/colab.md` §2** — replace the paragraph "If the install log shows numpy,
  pandas, pyarrow or torch being replaced, do ONE Runtime > Restart session and run all …" with:
  > On a **fresh VM** always do ONE **Runtime > Restart session and run all** after the first
  > editable install: the kernel processes the editable install's `.pth`/finder only at
  > interpreter start, so in-kernel `import demand_forecast` fails until then. On the rerun the
  > second install must leave every version in place (with a lock, an empty `pip freeze` diff).
  > Keep the restart-check cell immediately after the install cell; it only raises on a
  > numpy/torch version mismatch, so it never loops after the restart.
  Change the `cp docs/protocol.md … "$FEV_RUN_DIR/"` line and the `git rev-parse HEAD >
  "$FEV_RUN_DIR/commit.txt"` line to write into `"$FEV_RUN_DIR/provenance-$FEV_SHA/"` (mkdir -p
  first) so a later SHA never overwrites an earlier snapshot or `commit.txt`; update the §5
  copy-back checklist paths accordingly.

- [ ] **Step 2: Prompt 01 rev 2** — edits, in place:
  1. Title: append " — rev 2"; under it a "Revision note": rev 1 at `d683bf7` stopped on walmart
     (`windows with item-specific origins are not supported`, exit 2) after hierarchical_sales_1W,
     rossmann_1W and favorita_stores_1W completed; rev 2 runs at the fix SHA, adds prerequisite D
     and the rerun policy below.
  2. §0: every "A/B/C" or "§3–§5" reference to the prerequisites becomes "A–D" / "§3–§5b",
     including the score-viewing rule.
  3. §2 step 1: replace "If numpy/pandas/pyarrow/torch were replaced, do ONE …" with the
     unconditional fresh-VM restart rule (one sentence); keep the freeze-diff requirement; the
     provenance copy and `commit.txt` go to `$FEV_RUN_DIR/provenance-$FEV_SHA/`.
  4. §5: replace the entire extrapolation passage (from "A step processes a fixed batch of 128
     windows" to "label these \"extrapolated\"") with one rule: "Measured per-step time differs by
     task (rev 1 on a T4: 0.35 s hierarchical, 0.53 s hermes, 0.64 s m5), most plausibly with the
     per-sample feature assembly (covariate and static counts). For unprobed tasks use the
     **slowest measured** `train_s_per_step`, scale `eval_s` by `ceil(n_val_items / 512)` relative
     to the slowest probe, and label them \"extrapolated\"." Budgets come from `n_fit_ids` in
     `progress.json`/`run.json`, not from prompt 00 series counts (rohlik keeps 3,942 of 5,243).
  5. New "## 5b. Prerequisite D — walmart item-specific origins (trial directory)":
     `demand-forecast fev run --tasks walmart --config configs/fev-v2-standard.yaml --policy first_window --out-dir "$TRIAL" --stop-after 50`
     with the exit-1 pattern. Expected in the log, in order: `panel: task=walmart window=0 …`,
     `origins: task=walmart window=0 distinct=<n> earliest=<ts> latest=<ts>` with `n > 1`
     (diagnostic: 11, 2011-10-14, 2012-02-03), `identity: …`, `plan: …`, `fit: …`,
     `Training interrupted at step 50`; `progress.json["step"] == 50` and
     `progress.json["max_steps"]` (4,404 only if `n_fit_ids == 2936`); no `run.json`/bundle/NPZ.
     State that D covers startup only (origins, plan, first steps); inference correctness for
     ragged origins is covered by the repository's ragged-fixture tests at this SHA.
  6. §6 heading "only after §3–§5b passed"; first paragraph adds the rerun policy: "Rev 2 default
     (a): a **new** `FEV_RUN_DIR` under the rev-2 SHA with §3–§5b rerun. Alternative (b), only if
     the user chose it in writing: `--resume` into the rev-1 run directory. (b) is valid because
     the identity payload is unchanged for shared-origin windows and the skip path verifies every
     artifact; conditions: keep the rev-1 `environment-lock.txt` and dataset manifest, state
     whether A–C evidence is reused (allowed) or rerun (then use a fresh `$TRIAL-rev2` path), the
     three completed tasks must print `Skipping completed task`, their `run.json` lacks
     `earliest_origin` and their summaries rows show NaN in `max_distinct_origins` (expected;
     `summaries.csv` is rewritten as later tasks complete, existing values preserved), and §7
     records the producing SHA per task."
  7. §7 report template: status line "Prerequisites A–D"; per task "producing SHA;
     `max_distinct_origins` (n/a for rev-1 tasks)"; prerequisite D line "distinct origins,
     earliest/latest origin, interrupt evidence"; the caveat sentence that walmart's early-ending
     items are scored on periods other items' available past covers.
  8. Replace every statement that is now false (single-origin assumptions, `resolved_budget`
     nested in `progress.json`, prompt-00 series counts as budgets); keep `<SHA>` placeholders.

- [ ] **Step 3: Commit**

```bash
git add docs/colab.md docs/colab/prompts/01-fev-run-v2.md
git commit -m "docs: fresh-VM restart rule; prompt 01 rev 2 with the walmart origin prerequisite"
```

## Self-review notes

- Spec coverage: incident fixed (Task 1), diagnostic that would have caught it (Task 2), execution
  template defects found by prompt 01 (Task 3). Identity stability is asserted by the unchanged
  shared-origin e2e tests, the legacy-shape resume test and the worker's one-off d683bf7
  compatibility check (handoff).
- Type consistency: `window_origins`/`resolved_origin` (bench/run.py), `forecast_per_item_origins`
  (predictor), `earliest_origin` keyword (core/run.py, bundle metadata, run.json), `lengths`
  (fixture), `file_states` (test helper) are the only new names; Task 3 references the `origins:`
  log text produced by Task 1.
