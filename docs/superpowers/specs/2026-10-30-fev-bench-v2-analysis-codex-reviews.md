# Codex (gpt-6-astra, xhigh, read-only) review of 2026-10-30-fev-bench-v2-analysis.md

## Round 1 (on rev 1)

**Verdict: changes needed — the V2 direction is sound, but the proposed validation and acceptance process would not yet support a credible fev-bench result.**

Reviewed the 332-line [analysis document](/private/tmp/claude-501/-opt-SP-Workspace-Python-demand-forecast/e635ab62-5822-4c43-bb2e-f0f44880d3f0/scratchpad/analysis-frequency-quantiles-timestral.md), SHA-256 prefix `113839d9c543`, against demand-forecast `f60778a` and Timestral `a534bdb`. No files were edited.

Below, `analysis` refers to that document; `fev/` refers to the supplied scratchpad sources; repository paths are relative to their respective repositories.

**Findings**

1. **Blocking — The acceptance criteria use benchmark test results for model selection.**

   **Challenged claim:** Keep changes only if they improve the seven tasks, and require Rossmann/Favorita improvement after adding covariates (`analysis:303–312`).

   **Evidence:** These are the same rolling test windows used for the reported benchmark score. Repeatedly selecting architecture, preprocessing, and hyperparameters against them turns the benchmark into development data. The proposed loop also lacks an explicit boundary between preprocessing fitted for internal validation and preprocessing fitted for final forecasting (`analysis:257–259`).

   **Correction:** Define chronological inner validation using only data available before the current forecast origin. Fit learned preprocessing within that split, select configurations/checkpoints there, and then refit under a declared policy. Freeze choices before the reported benchmark run. Earlier evaluation targets may become training observations at later origins; that is legitimate chronological retraining. Selecting changes using the complete seven-task test scoreboard is different and must be disclosed as benchmark tuning.

   The shuffle test at `analysis:290` is also invalid: shuffled covariates can improve a weak model without indicating leakage. Replace it with deterministic invariance tests: changing withheld targets or future **past-only** covariates must not change predictions. Separately, use a synthetic forecasting problem to verify that informative known-future covariates are actually used.

2. **Major — The fev contract needs substantially more precision than “emit predictions and call evaluation_summary.”**

   **Challenged claim:** The adapter/evaluator specification is sufficient (`analysis:35–54,253–260`).

   **Evidence:**

   - `fev/task.py:145–179` consumes forecasts **positionally**. IDs do not rescue predictions reordered by grouping, sorting, or batching.
   - `fev/utils.py:416–432` treats integer cutoffs as series-relative indices; timestamp cutoffs include observations **at or before** the cutoff.
   - `fev/utils.py:435–453` excludes items lacking the configured context length or full horizon. `min_context_length` counts positions, not finite observations.
   - Default minimum context is **1**; Rohlik overrides it to **8**. Published Rohlik summaries contain **3,942 forecasts**, although the source dataset contains 5,243 series.
   - `fev/metrics.py:461–490` separately excludes undefined/zero-scale contributions from SQL. This does **not** authorize omitting those items from submitted predictions.
   - `fev/task.py:1001–1013` averages window scores. Pooling all forecast cells across windows can produce a different result.

   **Correction:** Preserve task-provided item order and timestamps; use `iter_windows()` and its input views; assert exact prediction coverage per window. Do not increase `min_context_length`, drop sparse items, or shorten the official horizon to accommodate V2.

   Emit finite `predictions` and all nine string quantile keys. SQL reads the quantile keys; MASE/WAPE read `predictions`. Preserve raw numerical forecasts without the legacy rounding contract.

   Pin fev, task definitions, dataset fingerprints and model revisions. The supplied comparison files span fev **0.7.0, 0.8.0, 0.9.0 and 0.11.0.dev0**. Their scores are useful references, but matching fingerprints, task settings and forecast counts must precede claims of exact reproducibility. Configure failure handling and bootstrap resampling explicitly; confidence intervals are not enabled merely by calling `leaderboard()` (`fev/analysis.py:332–358`).

3. **Major — Pinball in asinh-normalized space is not SQL, and should not be the sole selection metric.**

   **Challenged claim:** Normalized pinball “rewards SQL directly,” with early stopping on that loss (`analysis:233,273–275`).

   **Evidence:** SQL divides original-unit pinball loss by each item’s mean absolute seasonal difference (`fev/metrics.py:484–490`). Standard deviation is a different denominator; asinh additionally changes the error geometry.

   A numerical check demonstrated a ranking reversal. For targets `[0,100]`, forecasts A=`[0,80]`, B=`[8,100]`:

   | Median pinball | A | B | Preferred |
   |---|---:|---:|---|
   | Original units | 5 | 2 | B |
   | After asinh | 0.0558 | 0.6941 | A |

   Monotone transformations preserve population quantiles under ideal fitting, but they do not preserve finite-model empirical loss rankings.

   **Correction:** Evaluate and select checkpoints using exact fev SQL in original units, with original target missingness and the complete available past used for its denominator. A normalized training loss remains a reasonable surrogate, subject to validation.

   With affine normalization of scale `s`, normalized pinball needs weight `s / seasonal_error` to reproduce SQL’s weighting. Asinh has no equivalent constant correction. Declare the training treatment of zero/undefined scales separately from the official metric. Prefer uniform-series sampling initially; length-proportional sampling introduces another weighting mismatch.

4. **Major — Input masks alone do not fix missing-data handling; target masks and destructive preprocessing are missing from the plan.**

   **Challenged claim:** Adding `[values, missing]` channels and left-padding largely completes missingness support (`analysis:279–292`).

   **Evidence:**

   - `core/pipeline.py:157–166` applies sparse-item filtering.
   - `data/preprocessor.py:173–188` rejects short/zero-heavy series and missing categorical values.
   - `core/pipeline.py:191–197,222–235` clips targets and applies the transformation to the whole grid, including validation rows.
   - [V2’s encoder](/opt/SP/Workspace/Python/demand-forecast/demand_forecast/models/transformer_v2.py:495) receives no padding mask; patching and decomposition also lack missingness-aware behavior (`models/components.py:331–339,363–378`).
   - The proposed loss never specifies an observed-target mask.

   **Correction:** The fev path must preserve benchmark items, zeros, missing targets and raw evaluation labels. Disable inherited item filtering and evaluation-target clipping.

   Specify separate observation, padding and forecast-label masks. Fill nonfinite numerical inputs safely before arithmetic; multiplying NaNs by zero is insufficient. Define behavior for all-missing, constant and very short contexts. Propagate validity through attention, patching and decomposition, or disable those optional components initially.

   Category missingness and unseen-category handling need explicit policies, including unseen SKU IDs when reusing a first-window model. Do not reinterpret stockout-related zeros as latent demand: these tasks score their supplied targets.

5. **Major — Short histories, index-based access and retraining policy must precede the first meaningful V2 experiment.**

   **Challenged claim:** These can arrive in Phases 2–3, while date generalization can wait (`analysis:294–310`).

   **Evidence:** The newly added data section correctly establishes Rossmann’s first context as **29 weeks**. Holding back another 13 weeks for internal validation leaves only **16 weeks** before that validation origin. A complete `window + horizon` sampler cannot support the proposed default windows. The existing sampler requires complete windows (`data/dataset.py:106`).

   The analysis also correctly identifies Wednesday/Sunday anchors, then contradicts itself at `analysis:295–296`. Walmart adds a Friday anchor. Full daily-frequency support can wait; honoring these existing timestamps cannot.

   **Correction and cheapest sequence:**

   1. Freeze the protocol, profile all seven datasets, and reproduce official baselines.
   2. Build a thin V2 adapter with indexed arrays, short-context padding/masks, explicit horizon, nine trained quantiles, simple mask-aware normalization and exact SQL validation.
   3. Smoke-test hierarchical sales, **the earliest Rossmann window**, and Rohlik’s shortest eligible contexts before wider training. Add correctly typed dynamic inputs and full-horizon access.
   4. Tune a small configuration set on inner validation, then perform the frozen seven-task run. Evaluate optional architectural additions afterward using the same development boundary.

   Keep ragged arrays and static tables rather than repeatedly expanding static strings into long frames. Indexed storage should not become millions of Python `(series, offset)` objects unnecessarily.

   The analysis’s M5 memory estimate is illustrative, not measured: valid offsets depend on actual lengths, lookback, horizon and validation policy. Measure peak RAM/VRAM and batches per second on actual M5 data. Per-window fitting means **32 task-window fits per candidate**, before internal validation, refitting or additional seeds.

6. **Major — Merely concatenating future covariates leaves an avoidable information restriction in V2.**

   **Challenged claim:** Extending `future_time` is sufficient, and rows-with-roles has value only for zero-shot variable schemas (`analysis:238–249`).

   **Evidence:** [V2:527–534](/opt/SP/Workspace/Python/demand-forecast/demand_forecast/models/transformer_v2.py:527) applies causal decoder self-attention. Consequently, an early forecast step cannot see a promotion scheduled later in the known horizon.

   I verified this on a tiny V2 in evaluation mode: perturbing the final future-input token changed its corresponding output, while the first forecast changed by exactly **0**.

   **Correction:** Preserve V2 initially, but let every forecast step access the complete known-future schedule—through bidirectional decoder attention when decoder inputs contain only safe known information, or a separate known-future encoder.

   Rows-with-roles also offers shared processing across covariates, permutation-equivariant interactions and explicit availability boundaries. Those can matter within one task; fixed schema does not make them irrelevant. Conversely, the evidence does not establish that they outperform a properly implemented V2 on these tasks.

   **Recommendation:** Keep V2’s encoder–decoder and static embeddings/FiLM for the first controlled result. Fix future-information access now. Treat variate attention as a measured alternative. Defer a DHF rewrite: with Timestral’s patch size 32, these horizons require only one or two forecast patches, while full DHF training introduces a substantially different sampling/masking scheme.

7. **Major — Welford/asinh and clampsym are hypotheses to test, not an automatically transferable package.**

   **Challenged claim:** Port causal normalization and clampsym early; nonnegative clipping is always correct for sales (`analysis:235–242,283–285`).

   **Evidence:** Timestral uses different normalization rules for target/past-only rows and known-future rows (`timestral/model.py:122–151`). Its public forecaster also advances forecast normalization frames (`forecaster.py:458–483`). This is more specific than denormalizing everything with one window statistic.

   V2 currently encodes one observed context per training example. Using that entire context for mask-aware affine normalization is already leakage-safe; causal per-patch Welford is not required for that setup.

   Clampsym negates numerical covariates as well as targets and reverses quantile indices before averaging (`forecaster.py:214–230`). For a retail-only V2, negative prices and negated promotion flags can be outside its training distribution. It also approximately doubles model inference work.

   **Correction:** Start with a simple, explicitly defined normalization baseline; ablate asinh and per-patch normalization separately. Preserve Timestral’s exact normalization/refinement behavior when loading its weights.

   Apply nonnegative clipping only where target support is established. Nonnegative historical context does not prove future support. Keep sign-symmetrization and range caps optional, with inner-validation evidence and measured cost.

8. **Major — Timestral release facts need updating, and the gradient result supports a narrower conclusion.**

   **Challenged claim:** The technical report is “coming soon,” training details are wholly unavailable, and the gradient check establishes a straightforward fine-tuning path (`analysis:213–226`).

   **Evidence:** The [Timestral technical report](https://arxiv.org/html/2610.37585v1) was released **October 27, 2026**. It describes exclusively synthetic pretraining, subsequent real-plus-synthetic training, forecast-block training, role assignment and masked normalized-space pinball. It does not provide a complete reproducible training implementation and recipe.

   Additional code corrections:

   - Known-future rows cannot read **past-only rows either**, not just targets (`timestral/masks.py:104–118`). This prevents bidirectional covariate representations from carrying unavailable information backward.
   - Known-future normalization can use the supplied known horizon; binary known inputs receive special treatment (`model.py:135–151`).
   - Categorical target encoding is conditional: the forecaster uses it for the univariate-target case; multivariate-target handling differs (`forecaster.py:246–276`).
   - Broadcasting a static category and target-encoding it separately within each series collapses its identity into that series’s target statistics. It does not replace learned cross-series static embeddings. The [official fev Timestral wrapper](https://raw.githubusercontent.com/autogluon/fev/main/models/timestral/model.py) ignores statics.

   **Gradient verdict:** The narrow conclusion is sound. I reproduced finite, nonzero gradients on **47/47 parameter tensors** for both paths and sorted outputs. I disabled an optional `dill` import that otherwise probes unavailable writable temporary storage; the model computation was unchanged.

   The spike tests a randomly initialized tiny model’s `predict_block`, not checkpoint loading, an optimizer update, adapter training, public inference, advancing-frame parity or benchmark quality. The public `forecast` method uses inference mode (`forecaster.py:160`). Record these boundaries and require a loaded-checkpoint fine-tuning/parity spike before calling a pretrained adapter implementation-ready.

9. **Major — Retraining policy and `trained_on_this_dataset` need explicit, correct semantics.**

   **Challenged claim:** Per-window fitting is “the honest choice”; metadata needs no special treatment (`analysis:267–268`).

   **Evidence:** Both first-window fitting and chronological per-window fitting are legitimate. The [official AutoGluon wrapper](https://raw.githubusercontent.com/autogluon/fev/main/models/autogluon/model.py) supports both and defaults its wrapper argument to first-window fitting.

   The leakage flag is especially easy to misuse. Published task-trained TFT/DeepAR rows have `trained_on_this_dataset=False`. The [current model interface](https://raw.githubusercontent.com/autogluon/fev/main/src/fev/model.py) describes training-corpus overlap for **pretrained models**. `fev/analysis.py:332–333` can replace flagged results when leakage imputation is enabled.

   **Correction:** Identify V2 as task-specific/non-zero-shot and record its fitting policy independently of pretraining-overlap metadata. Follow the pinned fev convention; do not automatically set the leakage flag merely because fitting used permitted training history.

   For first-window reuse, keep fitted vocabularies and preprocessing compatible with the existing weights. Updating observed input context is expected; silently refitting its scaler or remapping embedding IDs is not. Report training and inference time under the chosen policy.

10. **Major — The new leaderboard table is useful, but its causal conclusions and proposed performance targets are unsupported.**

   **Challenged claim:** A ratio of 0.65 is a realistic V2 goal, 0.46–0.50 “almost certainly” requires pretraining, and the backbone decision changes only Phase 4 (`analysis:104–114,329–332`).

   **Evidence:** I independently recomputed the supplied seven-task geometric ratios: TFT **0.6670**, Timestral-5M **0.5019**, Timestral-209M **0.4606**, TimesFM-3 **0.4572**. The arithmetic is correct. It does not isolate architecture, training corpus, optimization budget or fallback behavior.

   The [fev-bench paper, Appendix B](https://arxiv.org/html/2509.26468v4) documents per-window deep-model refitting, preprocessing search for boosted models, and SeasonalNaive fallbacks when training data is insufficient. Rohlik’s nearly identical deep-model and naive scores therefore need fallback-log inspection before attributing them to neural architecture quality.

   **Correction:** Present those ratios as references or aspirations, not established feasibility bounds. A pretrained Timestral adapter changes normalization, representation, checkpoint compatibility, training and licensing; it is a separate design track, not merely an additional Phase 4 layer.

   Reuse existing [StatsForecast](https://raw.githubusercontent.com/autogluon/fev/main/models/statsforecast/model.py) and Timestral wrappers. The official weekly SeasonalNaive reference supplies probabilistic intervals and uses season length **1** here; the repository’s point seasonal-52 baseline is not interchangeable.

   Explicitly budget optimization. Existing defaults—10 epochs, learning rate `1e-5`, early-stop minimum improvement `1.0` (`config/settings.py:145–150`)—are not a justified SQL training recipe. In particular, an absolute improvement threshold of 1.0 is badly matched to many scores in the comparison table.

11. **Major — “Modified MIT” understates a decision-relevant weight restriction.**

   **Challenged claim:** Licensing is adequately captured by Apache attribution and a Modified MIT label (`analysis:195,318`).

   **Evidence:** The [released weight license](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE) denies rights under that license when the company’s—or employer’s—global consolidated monthly revenue exceeded **US$20 million in the preceding month**. Its text extends the restriction to derivatives, modifications and combined works.

   The inference code has a separate [Apache-2.0 license](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/LICENSE), with redistribution, attribution and modification-notice requirements.

   **Correction:** Separate code-only reuse from pretrained-weight use/fine-tuning. Confirm eligibility for the particular checkpoint before choosing that route. Preserve required notices for copied code; the repository’s own license does not erase upstream obligations.

12. **Minor — Two asserted implementation defects are false.**

   **Challenged claims:** AdamW decays inactive clusters; tsfresh metafeatures are diagnostic only (`analysis:187–191`).

   **Evidence:** `trainer.py:205–207` calls `zero_grad()` before backward. Installed PyTorch defaults this to `set_to_none=True`; Adam processes parameters only when their gradients are not `None` (`torch/optim/optimizer.py:1066`, `torch/optim/adam.py:151–154`). Unused cluster parameters therefore do not receive the alleged decay.

   `core/pipeline.py:175–188` explicitly uses extracted metafeatures to fit KMeans when `k > 1`.

   **Correction:** Remove both claims. Optimizers over all wrapper parameters may be unnecessarily broad, but that is not this alleged bug. Starting with `k=1` is sensible scope control; embeddings have not been shown to subsume clustering.

   The optimizer conclusion is source-verified. A runtime optimizer check was prevented by temporary-directory requirements in the read-only environment.

13. **Minor — Tighten the premise wording and several citations.**

   **Challenged claims:** Fixed windows block fev; confidence utilities are exclusively Gaussian/homoscedastic; point forecasts receive an additional SQL penalty.

   **Evidence and corrections:**

   - `analysis:140–141` reintroduces “fixed window,” contradicting the correct configurable-window finding immediately above it.
   - `trainer.py:166`, rather than `:167`, is the relevant `point_forecast` call.
   - The cutoff Monday check is conditional on the catalogue path; later forecasting checks are unconditional. Non-weekly configurations can fail explicitly as well as mis-slice, so “silently” is too categorical.
   - [confidence.py:89–111](/opt/SP/Workspace/Python/demand-forecast/demand_forecast/inference/confidence.py:89) already supplies arbitrary empirical residual quantiles. Its Gaussian helper can estimate different standard deviations by horizon (`:49–50`).
   - Missing quantile keys fail fev’s schema. Repeating one point prediction across the symmetric nine knots gives SQL equal to MASE under matching validity rules; I verified this numerically. There is no separate punitive multiplier for being a point model.

   These corrections do not weaken the requirement to train nine conditional quantiles. They clarify the actual deficiency and identify an existing residual-quantile baseline.

14. **Minor — VSN weights are not feature importance “for free.”**

   **Challenged claim:** Temporal VSN weights directly provide importance, supplemented by ordinary permutation (`analysis:248,288–290`).

   **Evidence:** Selection weights are internal, learned allocations affected by embeddings, scaling and correlated predictors. They do not measure the change in forecasting accuracy when a feature is unavailable. Shuffling individual timestamps also destroys time-series structure and can create implausible covariate combinations.

   **Correction:** Label VSN outputs as selection-weight diagnostics. For predictive importance, measure inner-validation SQL changes using grouped feature ablations or suitable block permutations, keeping a feature’s values, missingness and past/future portions consistent. Report correlated-feature limitations. Defer this work until the forecasting path is correct; an importance table is not a prerequisite for a credible first score.

**Confirmed correct**

- **Core premise:** Weekly arithmetic and Monday assumptions are widespread. Lookback is configurable, default **52**; the fixed **16** guard belongs to the forecast/benchmark contract, not V2’s general capacity.
- **Quantile training:** V2 has independent quantile outputs, but production training selects the point head. An independent backward check left **all four quantile-head parameter tensors without gradients**.
- **Task specification:** No material issue found in the seven-task horizon, window-count, seasonality and covariate table. Price is explicitly available on M5; the listed Rossmann/Favorita future inputs are promotions and related variables, not a common price column across all three.
- **Downloaded-data checks:** I independently read all three available Parquet files. Their series counts, lengths, anchors, target zero rates, fractional/count covariates and Rossmann missing-static rates agree with the added table.
- **Basic Timestral architecture:** No material issue found in the listed model dimensions, patch size, gated attention/RMSNorm/SwiGLU description, 21-knot monotone head or released width factor. The qualifications above concern information flow, normalization, categorical handling and training/release claims.
- **V2 static conditioning:** Learned SKU/category embeddings, static projection, VSN, FiLM and enrichment exist and provide a reasonable foundation to retain.
- **Initial ordering:** Harness and baselines first is correct. No material objection to a separate fev adapter or preserving the existing forecast contract.

Minimum acceptance evidence should include exact forecast coverage/order, horizons **8/13/39/52** without interpolation, finite ordered nine-quantile output, `predictions == q0.5`, missing/padding edge cases, leakage invariance, quantile gradients and tiny-batch learning, baseline metric parity, checkpoint reload parity, and measured memory/runtime. Quality results should report every task and window under a frozen protocol, including failures and fallbacks.

**Open questions**

- **Compute and retraining:** Decide hardware, total search budget, per-task/per-window limits, seeds and first-window versus per-window fitting. Thirty-two fits is the starting count for one per-window candidate.
- **Unprofiled datasets:** M5, Rohlik, Walmart and Hermes still need direct checks of ragged lengths, missing values, negative targets, zero-scale items, unseen categories and actual eligible-item counts. Source cards alone do not settle these.
- **M5 memory:** Measure actual peak RAM/VRAM with indexed loading and the chosen batch/context sizes before committing to an EC2 instance.
- **Pretrained route:** Confirm checkpoint-license eligibility and whether a Timestral baseline is desired. Decide on fine-tuning only after a loaded-checkpoint training/inference-parity experiment.
- **Historical comparisons:** Verify task/fingerprint compatibility and inspect fallback logs before treating the mixed-version published rows as exact reproduction targets.
- **Verification limits:** No full test suite, seven-task V2 training run or pretrained-weight fine-tuning was performed. The analysis’s “194 tests” count was not independently rerun. Claude’s review and disposition of these findings remain pending.
## Round 2 (on rev 2)

**Verdict: changes needed — rev 2 resolves most findings, but normalization edge cases, short-history validation, and decoder padding remain underspecified.**

References below use `rev2` for the revised analysis and `fev/` for the supplied scratchpad sources. Review-only; no files edited.

**Findings**

1. **Major — The normalization equivalence is conditional, and the fallback does not cover all-missing contexts.**  
   `rev2:273–275` is mathematically correct **when the scale equals fev’s positive, finite denominator**, with matching masks and reduction. For \(z=(y-a)/s\), \(2\rho_\tau(z-\hat z)=2\rho_\tau(y-\hat y)/s\). The location cancels. Ordinary pinball lacks fev’s factor of two (`fev/metrics.py:565–579`; repository `losses.py:103–109`).

   However, the capped lookback in `rev2:259` need not equal fev’s whole available past. Executing the supplied metric on history `[0,10,11,12]`, using only `[11,12]` for normalization, produced SQL **0.25** versus normalized doubled pinball **1.0**. An ε floor also changes weighting for positive scales below ε; it does not repair NaN scales. Missing labels require fev’s observed-cell reduction, not an unweighted average of per-item means.

   **Correction:** Distinguish full-prefix SQL scale from cropped model context. Either use the full-prefix denominator or describe normalized training loss as a surrogate, retaining exact original-unit SQL selection. Explicitly handle zero/undefined scales, empty-label batches, and all-missing inputs. Empirical quantiles are a valid finite fallback when observations exist; they are undefined when none exist. Specify a deterministic finite fallback for that case and log its use.

   **Rossmann clarification:** All seven tasks use **seasonality 1**, not 52. I measured positive finite scales for **all 1,115 items at 29 weeks**. Shortness alone does not invalidate those scales.

2. **Major — Inner validation is technically possible for earliest Rossmann, but its training regime and insufficient-history exception need definition.**  
   `rev2:235–239,304–306` reserves 13 of 29 weeks, leaving 16. With complete 13-step training labels and at least one context observation, training origins have context lengths **1, 2 or 3 only**. Left-padding makes this executable, but does not supply longer observed histories. With SQL-aligned exclusion, every one-step context has undefined scale; I measured seven zero-scale items at both two and three steps.

   Thus Rossmann is **not inherently unworkable**, but the plan must explicitly accept and test this short-context regime. More generally, an eligible Rohlik context of eight weeks leaves **zero** observations after an eight-step holdout. Official prediction eligibility cannot also serve as inner-validation eligibility.

   **Correction:** Specify minimum training/validation history, permissible partial-label sampling, and handling when validation SQL has no eligible contributions. A sound exception is a configuration and training-step budget fixed beforehand, followed by fitting on all available past without local early stopping; use a declared forecast fallback where training remains impossible. Alternatively retain Rossmann’s full holdout with explicit short-context training. Preserve official horizons and item coverage, and report validation exclusions separately.

3. **Major — Padding must also be masked in decoder cross-attention.**  
   `rev2:268` specifies an encoder attention mask only. [Decoder cross-attention](/opt/SP/Workspace/Python/demand-forecast/demand_forecast/models/transformer_v2.py:159) currently receives unmasked encoder memory. Padded query states can remain nonzero through residuals, FiLM and static enrichment.

   In an in-memory check, changing padded encoder inputs while masking their keys changed valid encoder outputs by **0**, but changed decoder outputs by **0.31985** without a memory mask.

   **Correction:** Carry the padding mask through every decoder cross-attention layer. Define behavior when all keys would be masked; distinguish absent positions from present positions with missing targets. Add padding-content and padding-length invariance checks to Phase 1.

4. **Minor — The clipping resolution still conflates observed nonnegativity with established target support.**  
   `rev2:225–227,324` connects clipping to dataset profiling. Finding no negatives in a panel does not establish nonnegative support; using withheld target values to choose clipping would also breach the development boundary.

   **Correction:** Enable clipping from documented task semantics, or select it using permitted inner-validation data. Full-panel profiling may describe data, but must not determine this forecasting policy from evaluation labels.

5. **Minor — Phase 0’s historical parity gate needs qualification; Phases 1–2 need explicit edge-case evidence.**  
   `rev2:298–299` requires matching published SQL, although §1 correctly notes mixed fev versions. Pinning fev alone does not pin the StatsForecast wrapper, dependency versions or historical dataset revision.

   **Correction:** Require exact historical parity only after compatible task definitions, fingerprints, forecast counts and baseline implementation are established; specify a numerical tolerance. Otherwise document the discrepancy and establish a reproducible pinned baseline.

   The remaining acceptance items are measurable, but insufficient until they include findings 1–3: metric fixtures with missing/zero/undefined scales, finite all-missing fallback, padding invariance, and successful execution of the short-history validation exception. Phase 2’s synthetic test should specifically demonstrate that a **later known-future input can affect an earlier forecast**. Beating SeasonalNaive is useful quality evidence, not a correctness prerequisite.

**Resolved**

- **#6 — Closed.** Removing causal decoder self-attention creates no future-target leakage path in the inspected V2: decoder inputs are known-future features; FiLM uses statics; cross-attention reads observed history. Subsequent decoder layers preserve that boundary. An in-memory mask-removal check changed the first-step response to a final-token perturbation from **0** to **0.05848**. No material issue remains with bidirectional decoding itself.
- **#8, #11 — Closed.** The revised Timestral report, information-flow qualifications, gradient-check limits and separate code/weight licensing treatment resolve the findings. The [technical report](https://arxiv.org/html/2610.37585v1) supports the revised training description; the supplied weight LICENSE contains the stated revenue restriction.
- **#9 — Closed.** `first_window` development and a frozen `per_window` run are legitimate. Published deep-model experiments used per-window fitting, despite the current wrapper’s first-window default. Report the actual policy and total fitting cost; freeze the final policy before viewing test scores. No inherent comparability problem remains. [fev-bench Appendix B](https://arxiv.org/html/2509.26468v4#A2), [AutoGluon wrapper](https://raw.githubusercontent.com/autogluon/fev/main/models/autogluon/model.py).
- **#10, #12, #13, #14 — Closed.** Reference-score interpretation, separate pretrained track, withdrawn optimizer/metafeature claims, corrected premise wording and diagnostic-only VSN interpretation are satisfactory.
- **#1 and #2:** Test/development separation, leakage tests and the positional fev contract are resolved; the execution and acceptance qualifications above remain.
- **#3, #4, #5 and #7:** Partially resolved; outstanding corrections are identified above.

**Open questions**

- Freeze the short-history exception, finite fallback, lookback and final retraining policy before implementation.
- Hardware, search budget and seeds remain execution decisions. The four unprofiled panels and actual M5 memory/runtime measurements remain Phase 0–1 evidence requirements.
- No additional design blocker was identified within this round’s scope. No training run or seven-task benchmark was performed; implementation readiness remains pending these corrections and Claude’s review.