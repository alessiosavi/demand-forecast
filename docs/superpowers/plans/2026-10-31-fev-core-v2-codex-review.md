# Codex (gpt-6-astra, xhigh, read-only) review of 2026-10-31-fev-core-v2.md

## Round 1 (on rev 1)

**Verdict: changes needed.**

Reviewed [plan](/opt/SP/Workspace/Python/demand-forecast/docs/superpowers/plans/2026-10-31-fev-core-v2.md) SHA `7f906db8190ce`, design rev 3, prior reviews, and repository `f60778a`. Below, `P:` identifies plan lines.

**Findings**

1. **Blocking — Tasks 13, 17, 23, 25: inference always takes the fallback.**  
   Task 13 makes `Batch.eligible` require an observed label; Task 17 uses `eligible == False` to select empirical-context quantiles (P:499,633). Real forecast batches have no labels, including every fev forecast. Thus even a successfully trained model produces the fallback for every item. Finite-output and resume tests would miss this.

   **Correction:** Separate positive-finite-scale eligibility from loss eligibility. Prediction fallbacks depend only on observed context and unfloored scale; loss eligibility additionally requires labels. Test that removing labels preserves neural predictions for a variable-history item, and changing model weights changes those predictions.

2. **Major — Tasks 13, 21, 22: the short-history exception is incomplete and can produce an empty training loader.**  
   Validation eligibility checks history length and labels but omits positive finite seasonal error (P:759). A sufficiently long constant-history panel therefore enters validation with SQL `NaN`, contrary to design §6.

   Separately, `origin_pos + horizon − 1 <= train_upto` excludes partial-label windows. With eight observations, horizon eight and one required context observation, **zero origins qualify**, even under the exception. Task 21 tests only the exception flag, not successful fitting.

   **Correction:** Include SQL eligibility in validation planning; trigger the exception when no contribution remains. Support partial training horizons with labels explicitly masked beyond the training cutoff. Define empty/no-eligible-training behavior and add complete `8/H8`, constant-history and mixed-panel fit→forecast tests.

3. **Major — Tasks 2, 3, 11: pinned fev APIs contradict the tests.**
   
   - Task 2 calls `SeasonalNaiveBaseline().evaluation_summary(task)`, but `ForecastingModel` provides `fit_predict`, not `evaluation_summary`. Call `task.evaluation_summary(model.fit_predict(task), model_name=..., training_time_s=..., inference_time_s=...)`. [Pinned model API](https://raw.githubusercontent.com/autogluon/fev/v0.10.0/src/fev/model.py).
   - Task 11 references nonexistent `task.metrics`. Construct metric objects with `get_metric` from `eval_metric` and `extra_metrics`; pass cleaned `DatasetDict` predictions to `window.compute_metrics`. [Pinned task API](https://raw.githubusercontent.com/autogluon/fev/v0.10.0/src/fev/task.py).
   - Task 3 assumes `board.model_name` is a column. Upstream returns model names in the index. Use `.loc["seasonal_naive", "skill_score"]`, or explicitly define the wrapper to reset the index. `n_resamples=1000` **is correct**. [Pinned leaderboard implementation](https://raw.githubusercontent.com/autogluon/fev/v0.10.0/src/fev/analysis.py).

4. **Major — Tasks 11, 22, 24, 26: the migration does not preserve green PR boundaries.**  
   The actual import audit gives:

   | Boundary | Concrete consequence |
   |---|---|
   | PR 0 | No existing legacy-import break identified. |
   | PR 1 | Removing `init_metrics` breaks `ForecastPipeline.validate` and patches in `test_pipeline.py:109,171`. Replacing `wape` breaks `test_forecast_contract.py:224`, which requires `None` for zero demand. |
   | PR 2 | Inherits those unresolved failures. |
   | PR 3 | Task 24 removes classes still eagerly imported by `config/__init__.py:3`. That initializer is absent from Task 26’s edits, so configuration imports and pytest collection fail. |

   Task 24 must also replace the **imports** in `tests/conftest.py:17`, not merely rebuild `sample_config`. Task 22’s instruction to delete classes importing `Trainer`/`Evaluator` from `test_training_flow.py` selects **nothing**: neither name occurs there. Replacing the trainer temporarily breaks `pipeline.py` and its dependent tests until Task 26 deletes them.

   **Correction:** Retain legacy metric entry points through PR 2; migrate consumers atomically in PR 3. Explicitly update `config/__init__.py` and conftest imports. Preserve the training-flow classes. Replace Task 26’s broad deletion grep: `data.loader` also matches the new `data.loader_long` and `data.loader_fev` tests.

5. **Major — Tasks 1, 3, 14, 24, 25: task metadata is insufficient to run the seven datasets correctly.**  
   The upstream YAML has no `categorical_statics`; Task 14 treats every unspecified static as numeric. Consequently a verbatim copy cannot supply the required typing for `StoreType`, `family`, `country`, numeric-coded IDs, etc. `FevTaskSpec` also lacks explicit defaults and mappings for omitted targets, cutoffs, metric configuration and renamed role fields (P:144).

   Task 25 never specifies how each task overrides the settings’ horizon, target and roles. One unchanged configuration cannot correctly cover horizons 8/13/39/52. Task 14 additionally defaults `target_support="nonnegative"`, reopening the explicitly rejected implicit-support policy, and cannot receive the already-resolved task frequency.

   **Correction:** Add the seven explicit role/support profiles and a deterministic task→settings projection. Pass task frequency, seasonality and declared support into the loader. Define all `FevTaskSpec` defaults and fev keyword mappings. Task 1’s `fixture_task` must also receive or retain `known`, `past`, and `statics`; they are currently undefined in its signature.

6. **Major — Tasks 4, 20, 22, 24: the frozen configuration and training budget have no executable owner.**  
   Task 4 freezes `d_model=64`, two layers and dropout 0.1. Task 24 merely adds fields to the existing `ModelConfig`, whose defaults are **256, four layers, dropout 0.3**, with patch embedding enabled. It does not explicitly reject the unavailable patch/decomposition options.

   The budget multiplier has no implementation task or rounding/override rules. Literally applying it to hierarchical sales gives `3000 × 118/2000 = 177` steps, while Task 22 evaluates only every 200 steps. Short runs—including the CLI fixtures—can finish without any validation or selected checkpoint.

   **Correction:** Provide a standard V2 configuration consumed by both commands, reject unsupported options, and define one budget-resolution function. Specify AdamW/cosine/warmup behavior in Task 22 and require terminal validation for ordinary runs shorter than `eval_every`.

7. **Major — Tasks 22, 25: exact interruption/resume is not established.**  
   Checkpoint contents omit history, best checkpoint/score, patience state and the within-epoch sampler position (P:791), although the test requires complete identical histories. Changing `max_steps` from 40 to 20 is not a faithful interruption test for a schedule parameterized by total steps. With default warmup 100, it also avoids exercising cosine decay.

   Task 25 only tests deleting a completed prediction file. It does not establish resuming interrupted training, loading the original first-window bundle when window zero is skipped, or rejecting incompatible cached artifacts. Appending another summary after recomputation can create duplicate task/model rows.

   **Correction:** Interrupt a run with an unchanged planned budget; restore all training and selection state. Define bundle/checkpoint recovery, seed initialization before model construction, artifact identity checks, and replacement of existing summaries. Test an interruption before the first bundle and one between windows.

8. **Major — Tasks 7, 9, 14, 22–25: temporal boundaries remain ambiguous.**  
   `Series.target` ends at the last non-NaN target (P:309), while fev’s historical span includes trailing missing observations. Task 14 appends future covariates without defining how that original boundary survives trimming. This can shift known-future values or discard historical covariates.

   No orchestration interface explicitly binds encoder training IDs and the signed fallback median to the permitted training split. Bundle saving requires a training cutoff but receives no explicit cutoff argument or defined manifest key.

   **Correction:** Preserve the historical grid boundary independently of the last observed target; align future inputs by timestamp. Define the training view, artifact-fitting IDs and forecast origin explicitly. Persist a precisely defined earliest permissible forecast origin accounting for labels used in checkpoint selection. Test trailing/all-missing history and validation-only IDs.

9. **Moderate — Tasks 13, 22: sampling differs from the design and has an unresolved argument.**  
   Design §4.5 requires uniform-series sampling. Shuffling the flat origin index instead weights a series by its number of origins. Task 22 calls `build_origins(..., min_context, ...)`, but neither its constructor nor `fit` receives `min_context`.

   **Correction:** Specify the series/origin sampling distribution and epoch length; add an unequal-history sampling test. Pass `ForecastConfig` or an explicit `min_context` through the trainer. Also define the finite origin range when `require_label=False`.

10. **Moderate — Tasks 6–8: grid and identity rules leave concrete counterexamples.**  
    The weekly grid rule accepts an entire series at 13:00: `Week.is_on_offset` is true, and all timestamps are integral steps from the first timestamp. The current test only rejects mixed midnight/13:00 observations.

    Joining IDs with `|` also merges `("a|b","c")` and `("a","b|c")`. Finally, duplicate/resample sums need an explicit all-NaN policy; ordinary pandas sums convert an entirely unknown target group to zero.

    **Correction:** Define and test the canonical time-of-day rule, reject or escape ambiguous composite keys, and preserve explicit missing targets during aggregation.

11. **Moderate — Task 11: metric definitions and parity coverage remain insufficient.**  
    WAPE/WQL names plus `eps=1` do not uniquely specify fev’s formula: fev floors the **mean absolute target**, not the target sum. The existing baseline fixture lacks the missing-label, undefined-scale and empty-eligible cases required by design §5.3. `rel=1e-9` also needs consistent float64 arithmetic. [Pinned metrics](https://raw.githubusercontent.com/autogluon/fev/v0.10.0/src/fev/metrics.py).

    **Correction:** State reductions and epsilon placement explicitly; add adversarial official-parity fixtures, including unequal observed-label counts and low target magnitudes. Define coverage behavior when configurable knots omit 0.1/0.9.

12. **Moderate — Task 15: the test calls the wrong API and targets an already-existing capability.**  
    The test uses `n_products`, `n_stores`, `days`; the retained signature uses `num_products`, `num_stores`, `num_days` ([generator.py:228](/opt/SP/Workspace/Python/demand-forecast/demand_forecast/synthetic/generator.py:228)). It cannot pass by merely extending that signature.

    `generate_sales_data` already emits zero rows. The zero-dropping implementation is `generate_product_sales`, which remains untouched; its existing test explicitly requires positive quantities.

    **Correction:** Use the real parameter names. Specify which retained generator/command gets the new zero-row contract, update its assertions deliberately, and retain missingness injection as a separate operation.

13. **Moderate — Tasks 3, 17, 18, 20, 22, 23: acceptance tests need deterministic, executable contracts.**
    
    - **Task 3:** CSV reload changes list-valued summary fields to strings and `None` to missing values; `second.equals(first)` needs canonicalization.
    - **Task 17:** An untrained signed model is not guaranteed to emit a negative value. Force a known head output.
    - **Task 18:** “Synthetic panel” does not establish a learnable relationship supporting the required 30% loss reduction. Define the generator and seed.
    - **Task 20:** The final invariance test has no defined `model`, does not require evaluation mode, and never changes future past-only covariates. The promised future-NaN model test is absent.
    - **Task 22:** Allowing either stop reason does not verify early stopping or restoration. Separate absolute best-checkpoint tracking from tolerance-based patience.
    - **Task 23:** Default JSON serialization cannot verify bit-identical reloads. A read-only probe changed float32 to float64 and `origin` timestamps to integers; `.equals` returned false. Compare losslessly serialized arrays and explicit timestamp metadata in the subprocess.

14. **Moderate — Tasks 1, 3, 11, 26: optional-bench behavior needs an explicit test boundary.**  
    Task 11 correctly registers `bench`, but a marker does not skip imports when extras are absent. Earlier bench tests and shared helpers can import fev during collection; CLI registration can introduce the same dependency.

    **Correction:** Require lazy bench imports, actionable missing-extra errors, and collection-time skips for bench-only tests. Verify base-package CLI/help without `bench`. Include the packaged weekly YAML in a wheel-install smoke test.

**Confirmed**

- **No material issue in the core mathematical contracts of Tasks 10, 12 and 16:** full-past seasonal error, unfloored eligibility, doubled pinball and monotone quantiles are consistent. The hand values `4.0` and `0.25` are correct.
- **No material issue in Task 9’s static encoding contract or Task 19’s attention-mask interface.** Task 20 correctly requires both encoder and cross-attention masks and bidirectional known-future decoding. Padding-length invariance is feasible with the copied V2 structure.
- **No material issue with benchmark-selection separation or the no-Timestral-weights rule.** Preserve full license text, upstream notices and modification notices when copying code.
- **Deferrals are acceptable:** `refit: full` and `per_window` are unnecessary for a frozen `first_window`, `refit: none` run. Move their explicit rejection behavior from the closing self-review into Tasks 24/25 and test it.
- **Local parquet loading is supported:** use `dataset_path=str(parquet)` and `dataset_config=None`; fev selects the parquet loader internally. `trained_on_this_dataset=False` is consistent with the wrapper’s pretraining-overlap convention.
- **The retained legacy families, wrapper, tuning and clustering have no direct imports of the deleted data modules.** Their preservation is viable after finding 4; deleting their tests is unnecessary.
- **Dependency declarations are compatible on Python 3.12.** fev 0.10.0 already requires `datasets>=2.15,<5`; StatsForecast’s requirements overlap the project’s NumPy range, and Python 3.12 wheels exist for StatsForecast and torch. Replacing `get_seasonality` removes the wrapper’s AutoGluon requirement; its Nixtla adapter adds no hidden AutoGluon dependency. Actual installation/resolution was **not executed**. [fev metadata](https://pypi.org/pypi/fev/0.10.0/json), [StatsForecast requirements](https://raw.githubusercontent.com/Nixtla/statsforecast/v2.0.1/settings.ini), [StatsForecast wheels](https://pypi.org/project/statsforecast/2.0.1/), [torch wheels](https://pypi.org/project/torch/2.6.0/), [adapter](https://raw.githubusercontent.com/autogluon/fev/v0.10.0/src/fev/adapters.py).
- **CPU index size is reasonable:** an in-memory vectorized proxy generated 7,592,010 origins in **0.25 seconds**, occupying **60,736,080 bytes**. This is not a measurement of the planned implementation or full suite. The few-minute suite claim remains unverified: Task 22 specifies **400 steps**, retained tests use widths above 16, and baseline multiprocessing/fev preprocessing need CPU-test limits.
- Read-only review completed; no files changed. Full tests and benchmark runs were skipped. Claude’s review of these findings is pending.

**Open questions**

- What exact bounded budget function and standard V2 configuration should be frozen, including short-run validation?
- Which generator API should receive the revised zero-row contract?
- Confirm phase 5 owns the actual seven-task profiles/baselines, dependency lock, interruption trial, memory measurements and frozen-run report. Those remain prerequisites before viewing frozen V2 benchmark scores.
## Round 2 (on rev 2)

**Verdict: changes needed.**

Reviewed [rev 2](/opt/SP/Workspace/Python/demand-forecast/docs/superpowers/plans/2026-10-31-fev-core-v2.md), SHA `d8e1aa6d2bd9`. `P:` below denotes plan lines.

**Findings**

1. **Major — Tasks 25–26: the fev training call has no matching interface.**  
   Task 26 calls `train_from_settings` on a prepared window-zero panel with an explicit origin; Task 25 accepts only settings, output directory and resume, and loads its own panel (P:728,756). Additionally, fev cutoffs can be negative integer positions, so literal `cutoff + step` does not produce a timestamp. [Pinned cutoff contract](https://raw.githubusercontent.com/autogluon/fev/v0.10.0/src/fev/task.py).

   **Correction:** Introduce a shared `train_panel(panel, origin, settings, ...)` entry point. Have both commands use it. Derive fev origins from each item’s first future timestamp; explicitly verify a common origin before using the scalar-origin API.

2. **Major — Task 2: the corrected baseline call still uses nonexistent attributes.**  
   P:86 passes `model.training_time_s` and `model.inference_time_s`. The pinned base class and wrapper expose `training_time` and `inference_time`; the `_s` suffix belongs to `evaluation_summary`’s arguments. The baseline test therefore raises `AttributeError`. [Pinned model source](https://raw.githubusercontent.com/autogluon/fev/v0.10.0/src/fev/model.py).

   **Correction:** Pass `training_time_s=model.training_time` and `inference_time_s=model.inference_time`.

3. **Major — Tasks 13, 21–22: empty training remains undefined.**  
   Partial horizons fix 8/H8, but singleton-only panels still have no origins: the sole observation cannot provide both preceding context and a label. All-missing panels also have empty `fit_ids`. Task 22 nevertheless requires a positive fixed budget and sets the fallback to an undefined empty median (P:394,623,643). There is no specified terminating behavior for an empty sampler.

   **Correction:** Define an explicit no-training result with zero optimizer steps and a usable fallback bundle; use fallback level zero when the permitted training split has no observations. Add singleton-only and all-missing fit→forecast tests, plus the previously requested constant-history fit→forecast test.

4. **Major — Tasks 22, 25: seeding still happens after model initialization.**  
   `Trainer.fit` seeds RNGs, but receives an already constructed model. Task 25 constructs that model without a preceding seed operation (P:643,728). Thus identical configuration seeds do not establish identical initialization, and the interruption comparison’s separate `fresh()` calls are unspecified. `Settings.seed` and `TrainConfig.seed` also have no declared precedence.

   **Correction:** Resolve one seed and apply it before model construction. Make `fresh()` reproduce the same initial state. On resume, restore checkpoint RNG states after construction/loading, without subsequently reseeding.

5. **Major — Task 26: resume identity and interrupted-training recovery remain incomplete.**  
   Matching predictions against the existing bundle and task does not detect a changed requested model/training configuration or seed. An internally consistent old bundle can therefore satisfy the cache check. The “before first bundle” test only creates an empty directory; it never creates or resumes a training checkpoint (P:756,772–774).

   **Correction:** Validate an expected run identity against checkpoints, bundles and predictions, including effective settings, seed and dataset fingerprint. Test interruption with a real checkpoint before bundle creation, unchanged-budget recovery, and rejection of changed configuration. Keep the summary replacement behavior.

6. **Moderate — Task 22: `best_step == 20` contradicts absolute-best tracking.**  
   With `rel_tol=1.0`, ordinary nonnegative SQL improvements cannot reset patience, so stopping at step 40 is consistent. But SQL can improve from step 20 to 40 without clearing that tolerance; absolute-best tracking must then select step 40 (P:643,648–652).

   **Correction:** Stub evaluation scores. For `[1.0, 0.9]`, assert stopping at 40 and restoring step 40; for `[1.0, 1.1]`, assert restoring step 20.

7. **Moderate — Tasks 4, 22: small budgets invert the clipping bounds and violate the test cap.**  
   For every `base_steps < 125`, the declared lower bound 500 exceeds the upper bound `4 * base_steps`. A NumPy probe gives `resolve_budget(60, 3, 200) == (240, 48)`. Consequently the exception test requires 240 steps, exceeding the explicit 150-step limit (P:30,32,655–656). The resume test can likewise run 240 steps.

   **Correction:** Define valid bounds for small budgets—for example, a lower bound of `min(500, base_steps)`—or provide an explicit absolute-step override with documented precedence. Preserve the intended production-budget formula and terminal evaluation.

8. **Moderate — Task 11: the parity test cannot call the declared API.**  
   `arrays_from_window` returns four values, including `y_pred`; `summarize` takes only three array arguments before `m`. P:360 therefore supplies `m` both positionally and by keyword. A signature-binding probe confirms `TypeError: multiple values for argument 'm'`.

   **Correction:** Unpack the helper result and pass the declared arguments explicitly. Also retain explicit undefined-scale and entirely empty-eligible parity cases; the revised parameter list does not identify either case.

9. **Moderate — Tasks 13, 17: two state-based tests do not establish their intended guarantees.**  
   The sampler test saves state after exhausting the iterator, then expects the entire exhausted epoch to replay (P:426). That does not verify restoration of the saved cursor. The label-free prediction test clears labels and their mask but leaves `loss_eligible` unchanged (P:524); an implementation incorrectly branching on that stale flag could pass.

   **Correction:** Save sampler state midway through an epoch and compare the remaining suffix. In the prediction test, also clear `loss_eligible`, assert the row remains scale/context eligible, and verify weight-sensitive predictions.

10. **Major — Tasks 15, 24: the generator correction still breaks retained tests.**  
    P:465 calls `generate_product_sales(PRODUCT, DATES, rng=...)`. The actual signature is `(products, start, end, seed=42, promo_share=0.10)`; adding `emit_zero_rows` does not make that call valid. Moreover, [the retained CLI test](/opt/SP/Workspace/Python/demand-forecast/tests/test_cli.py:113) still requires strictly positive quantities.

    **Correction:** Use the existing `products` fixture and call `(products, START, END, seed=0)`, asserting `len(products) * len(DATES)` rows. Update the CLI zero-row assertions in Task 15. Task 24’s deletion regex currently also selects `tests/test_cli.py`; trim the obsolete tests rather than deleting that file wholesale.

11. **Major — Task 1: the bench conftest is not a reliable collection boundary.**  
    Import-time `pytest.importorskip` in conftest can raise an uncaught `Skipped` during initial conftest loading, depending on invocation; this is documented upstream. Also, `pytestmark` in conftest does not mark descendant test modules. [Pytest issue](https://github.com/pytest-dev/pytest/issues/15142), [marker documentation](https://docs.pytest.org/en/stable/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module).

    **Correction:** Put import guards and markers in bench test modules before bench-dependent helper imports, or implement a supported collection hook. Verify full-suite collection without bench; checking only `test_package.py` and `test_config.py` avoids exercising this boundary.

12. **Moderate — Task 27: the wheel test lacks its build dependency and installation check.**  
    P:795 invokes `python -m build`, but `build` is absent from `.[dev]`; CI installs it only in the separate build job, which depends on tests passing. The test inspects a ZIP rather than installing the wheel. The proposed second-step install also does not itself remove previously installed bench dependencies.

    **Correction:** Install the build frontend in the test environment or move this check into the build job. Use the fresh venv promised by Global Constraints, install the wheel, and test CLI/help and packaged task/profile loading outside the checkout.

**Resolved**

- **Round 1 finding 1:** The `eligible` / `has_context` / `loss_eligible` contracts and label-free fallback selection are correct. Test strengthening remains in finding 9 above.
- **Finding 2:** Partial horizons and seasonal-error validation eligibility are corrected. A read-only 8/H8 probe produces origins **1–7**, with loss-eligible origins **2–7** for variable, fully observed history. Empty-training handling remains open.
- **Finding 3:** `fit_predict`, metric construction and leaderboard indexing are corrected; timing attributes and the parity-test call remain open.
- **Finding 4:** The legacy **import graph** is repaired. Tasks 19–22 preserve legacy consumers; Task 23 retains `Predictor`; Task 24 removes obsolete modules and updates exports; Task 25 rewrites settings, config exports and conftest together. No additional dangling legacy import was found at Tasks 26–27. Full trees are nevertheless not green because of the test failures above, including the inherited Task 15 CLI failure.
- **Finding 5:** Explicit profiles, required support, task defaults and task→settings projection resolve the metadata problem. The training-interface gap is finding 1 above.
- **Finding 6:** Standard V2 defaults, unsupported-option rejection, optimizer/schedule ownership and terminal validation are corrected. Budget and selection tests remain open.
- **Finding 7:** The checkpoint state list is substantially complete, the interruption test preserves its planned budget, and summary replacement is specified. Initialization and recovery verification remain open.
- **Finding 8:** Explicit fev `history_end`, `history_length`, `fit_ids`, `training_cutoff` and `earliest_origin` resolve the core temporal contracts. Origin wiring and empty-history execution remain open.
- **Finding 9:** Uniform-series sampling, finite origin bounds and `min_context` ownership are corrected; the cursor test remains open.
- **Finding 10:** Closed. Midnight `W-SUN`, `MS` and business-day timestamps satisfy the rule. Implement calendar stepping without assuming these non-fixed offsets expose `.nanos`. Pipe rejection and `min_count=1` resolve the stated counterexamples.
- **Finding 11:** Metric reductions and epsilon placement are corrected; parity execution/coverage remains open.
- **Finding 12:** The correct zero-dropping generator is now targeted; its call and retained CLI assertion remain open.
- **Finding 13:** CSV comparison, evaluation-mode invariance and lossless array reload are improved; deterministic selection/resume assertions remain open.
- **Finding 14:** Lazy imports and packaged YAML are specified, but collection and wheel verification remain open.

No material issue remains in the corrected eligibility mathematics, midnight grid rule, task role profiles, or legacy import ordering.

**Open questions**

No additional scope decision is needed. Phase 5 explicitly owns real-data profiles, baselines, dependency locking, interruption trials, memory measurements and the frozen-run report. Those are appropriate prerequisites, not missing implementation tasks here.

Read-only probes and import/source audits completed; no files edited. Full tests and benchmark runs were skipped. A PyTorch probe was blocked by read-only temporary-file requirements. Findings await Claude’s review.
## Round 3 (on rev 3)

**Verdict: changes needed.**

Reviewed [rev 3](/opt/SP/Workspace/Python/demand-forecast/docs/superpowers/plans/2026-10-31-fev-core-v2.md), SHA-256 `0fa18019b895b`, against all 12 round-2 findings. `P:` denotes plan lines.

**Findings**

1. **Major — Task 26: the new fingerprint access names a nonexistent attribute.**  
   P:771 calls `task.dataset_fingerprint`. Pinned fev stores `task._dataset_fingerprint` after loading; `dataset_fingerprint` is an evaluation-summary key, not a public Task attribute. Both end-to-end `run_fev` tests would fail at this access. [Pinned source](https://raw.githubusercontent.com/autogluon/fev/v0.10.0/src/fev/task.py).

   **Correction:** Define a pinned adapter that retrieves the initialized fingerprint after loading, or computes it explicitly. Also spell the task output directory `out_dir / spec.name`, matching the tests, rather than `out_dir / task`.

2. **Major — Global Constraints, Tasks 25–26: resume identity omits the window definition and has conflicting mismatch behavior.**  
   P:20,743,771 hash settings, seed, dataset fingerprint and package version. The settings projection omits `initial_cutoff` and `num_windows`; `origin` is also absent from the hash. For the same 40-step dataset with horizon four, changing two windows to one changes the first future position from 32 to 36 while leaving every declared hash input unchanged. Cached `window_0.npz` can therefore refer to the wrong dates. The upstream fingerprint describes dataset contents, independently of evaluation cutoffs. [Fingerprint implementation](https://raw.githubusercontent.com/autogluon/fev/v0.10.0/src/fev/utils.py).

   Separately, P:20 requires identity mismatches to raise, while P:771 says mismatched bundles/predictions are recomputed. This makes the changed-configuration rejection test depend on which artifacts remain.

   **Correction:** Include the resolved training origin and canonical task/window definition in identity construction, consistently across orchestration and training. Require mismatched checkpoints, bundles and predictions to raise; recompute only missing artifacts. Test changed cutoffs and mismatches after checkpoint removal.

3. **Moderate — Tasks 22, 25: sampler seed wiring remains unspecified.**  
   Seeding before model construction is corrected. However, P:648 still uses `WindowSampler(..., seed=<the run seed>)`, while neither `Trainer.__init__` nor `fit` receives that seed, and `TrainConfig.seed` has been removed. The identity string does not supply it.

   **Correction:** Pass `settings.seed` explicitly into the trainer/sampler, or explicitly document its retrieval mechanism. Preserve RNG restoration without subsequent reseeding. Exercise a nonzero seed.

4. **Moderate — Task 22: the singleton test applies the empty-split fallback invariant to a nonempty split.**  
   P:648 requires the observed training-split median, using zero only when no observations exist. P:674 nevertheless requires `fallback_level == 0.0` for singleton-only data. Two singletons valued 5 and 7 have no training origins but require fallback level 6. The assertion works only with an unstated zero-median fixture.

   The requested **all-missing fit→forecast test is still absent**; neither this test nor the constant-history test exercises empty `fit_ids`.

   **Correction:** Define nonzero singleton values and assert their observed median. Add a separate all-missing case asserting zero steps, fallback level zero, and finite forecasts after bundle save/reload, including `target_support="real"`.

5. **Moderate — Tasks 22, 25–26: `InterruptedRun` is named, but its control flow and recovery evidence remain incomplete.**  
   P:823 establishes the exception’s name and module. However, P:648 returns an interrupted `FitResult`, while P:743 proceeds to save a bundle without specifying an interrupted-result guard. That conflicts with P:790’s requirement that no bundle exist.

   Moreover, P:790 checks only that the checkpoint **directory** exists. An empty directory followed by deterministic training from scratch can satisfy the remaining comparison; it does not prove checkpoint recovery.

   **Correction:** Specify checkpoint saving when interruption occurs; inspect the result in `train_panel` and raise `InterruptedRun` before bundle creation, propagating through `run_fev`. Assert a checkpoint file containing the interrupted step and verify that resume actually loads it.

6. **Low — Tasks 13, 22: helper name drift remains.**  
   P:427,430 use `encoder_long`; P:658,678–682 use `enc_long`, without declaring an alias.

   **Correction:** Standardize the fixture name. Update P:829’s name inventory to include `train_panel`, `run_identity`, `window_origin` and `InterruptedRun`.

**Resolved**

| Round-2 finding | Round-3 assessment |
|---|---|
| **1 — Training interface/origin** | Resolved: both callers use `train_panel`; `window_origin` uses future timestamps and rejects unequal item origins. |
| **2 — Timing attributes** | Resolved: `training_time` and `inference_time` match the [pinned model](https://raw.githubusercontent.com/autogluon/fev/v0.10.0/src/fev/model.py). |
| **3 — Empty training** | Partially resolved: zero-step result and empty-split fallback are defined; finding 4 remains. Constant-history fit→forecast coverage is added. |
| **4 — Seeding** | Initialization and resume ordering are corrected; sampler wiring remains in finding 3. |
| **5 — Resume identity/recovery** | Identity persistence and interruption assertions are added; findings 1, 2 and 5 remain. |
| **6 — Early stopping** | Resolved: both sequences stop at 40; `[1.0, 0.9]` selects 40 and `[1.0, 1.1]` selects 20. Saved-state assertions match absolute-best selection. |
| **7 — Budget clipping** | Resolved: all three requested examples match the revised formula. |
| **8 — Metric parity call** | Resolved: explicit unpacking binds correctly; all-constant and no-label cases are present. The comment should clarify that all-constant history makes **SQL/MASE**, not necessarily WAPE/WQL, NaN. |
| **9 — State-based tests** | Resolved: sampler state is saved after seven draws and compares the remaining suffix; label-free prediction also clears `loss_eligible` and checks weight sensitivity. |
| **10 — Generator/CLI** | Resolved: the call matches the existing signature and fixture; both positive-quantity assertions are targeted; `test_cli.py` is trimmed. |
| **11 — Optional bench boundary** | Resolved: module-level guards precede dependent imports, markers belong to test modules, and CI runs the full suite in a fresh base-only environment. This follows [pytest’s supported pattern](https://docs.pytest.org/en/stable/how-to/skipping.html#skipping-on-a-missing-import-dependency). |
| **12 — Wheel check** | Resolved: the build job already installs `build`; the revised check installs the wheel into a fresh venv and exercises CLI/help and packaged YAML outside the checkout. |

**Open questions**

No owner scope decision is needed; the remaining points are interface and acceptance-test corrections.

Passed: read-only budget, early-stop and signature-binding probes; repository and pinned-source audits. Skipped: implementation tests, CI, wheel installation and benchmarks. No files edited. Findings await Claude’s review.