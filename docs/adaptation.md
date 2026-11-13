# Task-local adaptation inputs

Export one task before running a trainer. This module loads no model and performs
no fitting. It does not change the yardstick runner or the scoring adapters.
The native fev loader handles local data paths relative to the task YAML and
remote datasets according to the task definition.

```bash
python -m demand_forecast.bench.adaptation \
  --tasks /path/to/cohort.yaml --task fred_md_2025/macro \
  --phase development --output /path/to/adaptation-bundle
```

The output directory must be new. Validation, serialization and hashing finish
in a temporary sibling directory before publication. A failed build publishes
no bundle. Task names are identity strings, never output paths; `/` is supported.
The CLI prints the manifest path and fingerprint. Offline operation requires
local or already cached data; the exporter never downloads model weights.

Python entry points:

```python
from demand_forecast.bench.adaptation import build_manifest, build_task_manifest

manifest = build_task_manifest(task_yaml, task_name, output, phase="development", max_context=None)
# Lower-level seam for an already selected fev 0.10.0 Task:
manifest = build_manifest(
    task, task_yaml=task_yaml, output=output, phase="development", max_context=None
)
```

The lower-level caller is responsible for matching `task` to `task_yaml`.
`phase` is `development` (default), `confirmation` or `official`; it records the
caller's provenance declaration. It does not certify cohort membership or
permit selection on confirmation/official outer labels. That enforcement belongs
to the later selection workflow.

## Cut-offs and coverage

H is the task horizon. Each item's T is its first outer **forecast observation**.
Integer fev cut-offs are Python slice indices (negative indices resolve against
that item's length). A fev timestamp cut-off includes that timestamp in history;
T is the following observation. Date requests must exactly match an observed
item timestamp. Missing/off-grid dates fail rather than silently snapping.

The inner forecast indices are T−2H and T−H. All indices are zero-based and all
end indices are exclusive. Splits use each item's validated timestamp grid;
calendar frequencies such as months use observed steps, not fixed timedeltas.
Null, duplicate, decreasing or irregular timestamps fail. Target and declared
known-covariate lengths must equal the timestamp length.

An item needs the task's minimum context before T and T−2H and a full outer
horizon. Pooled training additionally stops strictly before the earliest
T−2H timestamp among these feasible items. This prevents one long item's
training labels crossing another item's inner boundary. Each training end is
`min(own T−2H index, first index at/after the shared boundary)`.
An empty/too-short shared prefix excludes that item as well.

Excluded items have no payload rows, retain their cut-off metadata and receive
`fallback: frozen_model`. Reasons are `insufficient_outer_history`,
`insufficient_inner_history` and `insufficient_shared_training_history`.
An entirely infeasible task is a valid `frozen_fallback` manifest with empty
training and inner rows. These are structural checks, not sampler eligibility:
the adaptation trainer must still enforce its registered minimum observed
history (32 points), valid gradient windows and any model context limits.

## Files and arrays (schema version 2)

Parity with the unchanged yardstick runner and `SimonsAdapter` takes precedence
over preserving source representations in model inputs. Version 2 decodes raw
Arrow cells for timestamp validation, full-precision scoring fields and source
hashes, then applies the scorer's formatting to bounded training/model slices.
History and horizon are formatted separately, before the optional model cap,
matching fev and the runner. Neither shared datasets nor scoring code change.
Version 2 distinguishes these bundles from version 1. Regenerate any intermediate
version 2 bundles built with the experimental runner change; source hashes
distinguish them from the final implementation.

Offline synthetic tasks using real fev objects establish the default NumPy path:

| Source case | Unchanged scorer presents to Simons | Native training material |
|---|---|---|
| Null in text known covariate | `None`; literal `"None"` remains a distinct string | NaN for the null; a category code for literal `"None"` |
| Null in numeric known covariate | NaN; finite values rounded to float32 | NaN; float32 numeric values |
| Boolean known covariate | Boolean values in an object array | Numeric float32 0/1, matching Simons' numeric conversion |
| Float64 target | Float32 target block | The same float32 values; scoring fields and source hashes retain float64 precision |

These preserve the same missingness and model numeric representation. Numeric
conversion also retains the scorer's float64 intermediate before float32,
which matters for large integer values. The text case reflects the installed
formatter, rather than assuming all NumPy/library
versions behave identically. Actual formatted slices determine both native
training encoding and inner inputs; a caller that changes fev's dataset format
also changes its scorer inputs. Source hashes remain based on raw cells.

```text
bundle/
  manifest.json
  univariate/train-00000.parquet
  covariates/train-00000.parquet   # present only with eligible known-covariate rows
  inner-0.json                    # origin T−2H
  inner-1.json                    # origin T−H
```

Let D be the number of task targets, V a contiguous block of at most 32 targets,
K an item's bounded training length, C the number of known covariates, H the
horizon, and L the model's history length at an inner origin. Item order is
lexicographic string ID order. Target/covariate order follows the loaded task.
No targets from different items share a model group.

Both Parquet files follow the existing Simons native readers. The main rows
have exactly `item_id: string`, `source: string` (task name), `freq: string`,
`target: list<list<float32>> [V,K]`. IDs are compact JSON strings `[original_id,
block_number]`, avoiding collisions with arbitrary original IDs.
`manifest.groups` maps each native row to `item_id`, `native_item_id`,
`target_indices` and `target_columns`. Channels stay together in contiguous
blocks, including a smaller final block when D exceeds 32.

Covariate rows have `item_id: string` (JSON `[original_id, target_index]`),
`source`, `freq`, `target: list<float32> [K]` and
`covariates: list<list<float32>> [C,K]`. There is one native covariate row per
target channel because Simons' existing real-covariate reader takes one target.
This secondary view does not change the main grouped rows. Use `univariate/`
for the main reader and `covariates/` for the covariate reader; do not treat the
whole bundle as an interchangeable native training folder.

Numeric/boolean training covariates become float32. String covariates use sorted
categories fitted **only on that item's scorer-formatted bounded training prefix**.
Null/unknown values become NaN; a formatter-produced string would be treated as
a category, just as Simons would. `covariates.training_encodings` records each
item's column `kind` (`numeric` or `categorical`), categorical `categories` and
`unknown: null`.
The later trainer uses this mapping for native rows. Inner inference preserves
scorer-formatted covariate values so Simons can apply its own per-window conversion
exactly as the scoring adapter does. Do not replace those values with native codes.

Each inner JSON contains `schema_version`, `origin_number` (0 or 1), `scoring`
and `items`. Each item has:

- `item_id`, `origin: {index, timestamp}`;
- `history_start_index`, `score_history_start_index`, `history_timestamps [L]`,
  `horizon_timestamps [H]`;
- `model_input`: an ordered list of blocks, each with `target_indices [V]`,
  `target [V,L]` (float32 values) and `known_covariates` (null without covariates,
  otherwise a column-name mapping to scorer-formatted arrays `[L+H]`);
- `score_past [D,S]`: float64 history with the fev task's context cap;
- `score_target [D,H]`: float64 held-out observations.

The fev `max_context_length` caps score history and model input. The optional
CLI/Python `max_context` applies a further model-input cap, matching yardstick
`Options.max_context`; it does not shorten scoring history. Model defaults
inside a checkpoint remain the checkpoint's responsibility.
JSON is canonical UTF-8 with sorted keys, compact separators and a final newline.
JSON null represents missing observations; convert numeric arrays with
`np.asarray(values, dtype=np.float64)` (float32 for model targets) to restore NaN.
NaN is retained in native Parquet. Infinite permitted values fail.

## Manifest fields and identity

`manifest.json` has these top-level fields:

| Field | Meaning |
|---|---|
| `schema_version`, `phase`, `status` | Schema 2, declared provenance, `ready` or `frozen_fallback` |
| `task` | `name`, exact source `yaml_sha256`, loaded fev `definition`, `freq`, `horizon`, ordered `target_columns` |
| `versions` | fev package version; reference fev/Simons commits; SHA256 of installed fev task/metrics, scoring adapter, runner and builder source; PyArrow writer version |
| `cutoffs` | `fev_initial_cutoff`, index `convention`, `inner_offsets [-2H,-H]`, `shared_training_boundary` (timestamp or null) |
| `scoring` | `eval_metric` (name or full configuration), `extra_metrics`, ordered `quantile_levels`, `seasonality`, `target_columns`, `aggregation` |
| `model_input` | `dtype`, `max_variates`, `max_context`, `fev_max_context_length`, role/settings assertions `covariates`, `past_covariates`, `shared_categories` |
| `items` | Every ID, `outer_origin`, two `inner_origins`, own `training_boundary`, pooled `training_end_index`, `training_end_timestamp`, precise `training_input_sha256`, `eligible` |
| `excluded_items` | ID, reason and frozen fallback for every excluded item |
| `groups` | Native main-row mapping to original items and ordered channels |
| `covariates` | `known_future`, `past_only_dropped`, `static_dropped`, `dropped_value_counts`, bounded `training_encodings` |
| `counts` | Item/channel/group counts and training/inner value counts |
| `files` | Relative payload filename → `{sha256, bytes}`; excludes the manifest itself |
| `fingerprint` | SHA256 of canonical manifest bytes **with this field omitted**, including all payload hashes |

Each origin/boundary is `{index, timestamp}`. An excluded item's out-of-range
index is retained and its timestamp is null. The outer-origin timestamp is
boundary metadata; no observation values from T onward are exported.
`training_end_timestamp` is the first excluded timestamp, not the last label.
Counts include `items`, `eligible_items`, `excluded_items`, `channels`,
`eligible_channels`, `training_groups`, `training_covariate_rows`,
`training_target_values`, `training_observed_target_values`,
`training_known_covariate_values`, and `inner_target_values_per_origin`.
Known-covariate training counts count unique item/column/time cells, before
replication across native per-target rows. Dropped past-only counts count
historical cells strictly before each item's T (including excluded items);
static counts count item/column cells. Their **values are never extracted**.

Identity covers permitted payloads and configuration, not fev's full dataset
fingerprint, cache paths, runtime timestamps or the output directory. A change
to YAML bytes, source versions/settings, eligible metadata or an exported value
changes identity. Each eligible item also hashes its permitted training target
and raw known-covariate prefix before float32 conversion, so rounding cannot
hide a source-value change; excluded items have a null training hash. Changes to dropped values or data values at/after T do not.
Byte identity is guaranteed for repeated builds in the same software environment;
writer/source changes are explicitly part of the fingerprint.

## Reproduce fev scores

Restore each `score_target` and transpose it to `[H,D]`; stack items into
`y_true [N,H,D]`. Transpose each `score_past` and concatenate into
`y_past [sum(S),D]`; retain `y_past_lengths [N]`. Reassemble predicted blocks
using `target_indices`, with `y_pred [N,H,D]` and
`q_pred [N,H,D,Q]` in the manifest's quantile order. Pass these arrays,
`seasonality` and `quantile_levels` to
`fev.metrics.get_metric(scoring['eval_metric']).compute_scores(...)` using its
default item/time reduction. This preserves SQL's historical seasonal scale,
missing values, channel aggregation, and configured metric parameters. Never
average independently computed item scores to reproduce dataset-wide metrics.
The parity test compares this calculation with a real synthetic fev window.

## Leakage guarantees and verification

Training targets and training covariates stop before both the item's T−2H and
the shared calendar boundary. Training encoders use that same prefix. Inner
inputs stop before their origin; known covariates stop at origin+H. Held-out
labels appear only in scoring fields and, causally, later inner history.
Nothing uses or hashes observation values at/after the item's T.

Perturbing data from the training boundary onward leaves native training files
byte-identical. Perturbing data at/after T leaves **every bundle file** identical.
Perturbing the second inner target changes only `inner-1.json` and the manifest's
hashes/fingerprint. Perturbing the first inner target also changes `inner-1.json`:
those labels are legitimate later history. Isolating that perturbation to the
first file would violate exact scoring-adapter input parity. The fingerprint
therefore describes permitted effective inputs, not every value in the source.

Offline tests in `tests/test_bench/test_adaptation_data.py` cover native shapes,
short/ragged items, calendar boundaries, slash names, integer/inclusive date
cut-offs, covariate roles, causal perturbations, byte determinism, adapter calls,
and real fev SQL parity. No weights, cloud services or benchmark downloads are
needed. Real cohort feasibility and model-training behavior require separate
verification by the subsequent tasks.

## Per-task Simons trainer (patch 0004)

Apply the ordered patch series in `third_party/simons/` to the pinned upstream
checkout. Run the trainer inside that prepared Simons tree:

```bash
python -m simons.training.adapt \
  --bundle /path/to/adaptation-bundle \
  --base /path/to/immutable-base-export \
  --output /path/to/task-arm-seed \
  --params head --lr-multiplier 0.02 --max-updates 500 --seed 11 \
  --device cuda
```

The base must be a **local** directory with `config.json` and
`model.safetensors`. The trainer never downloads weights or datasets. The recipe
is specified by CLI flags; `--help` lists every flag. Registered arms are
`head|all`, multipliers `0.02|0.10`, seeds `11|29`. Each new output starts from the
same base with fresh NorMuon/AdamW state. Reusing an output resumes its own
identity-checked state; use a separate output for every task/arm/seed. No job
generation, outer-window scoring, seed selection or cross-task weight carryover
is implemented here.

`head` trains exactly `norm.weight`, `head.weight`, `head.bias`; all other
parameters have gradients disabled. For the native stage-4 25M config these are
384, 110,592 and 288 scalars: **111,264 trainable / 25,010,304 frozen**.
`all` enables all 25,121,568 scalars, including learned covariate modules. Counts,
names, trainable shapes and individual tensor sizes are computed from the actual
base and saved in the receipt. A different architecture has different counts.
Frozen tensors are passed to the upstream optimizer with no gradients, so its
weight decay cannot alter them.

The trainer accepts **bundle schema 2 only**, checking the manifest and both
inner-file versions separately. Other versions fail with an explicit
`schema_version must be 2` error; regenerate older bundles. It verifies the
manifest fingerprint and every payload's hash/size before consuming data, and
checks the native Parquet field names and float32 list types before sampling.
The trainer receipt remains `adaptation.json` schema 1; that is a separate format.
Source/version hashes and precise `training_input_sha256` values remain covered
by bundle identity even when a source change rounds to identical float32 rows.

Training reads only hash-verified native Parquet rows inside the bundle. Each
update samples 32 effective grouped windows: uniformly choose an eligible item,
then one of its eligible contiguous target blocks, then a valid origin.
History is capped at 2,048 steps and the base's context, rounded down to its
patch/far-pool layout. Every target channel needs at least 32 finite history
points; the group needs at least one finite horizon label. No valid origins
means a frozen fallback. Targets retain their groups of at most 32; known
covariates use the bundle's native numeric/history-fitted categorical codes,
with the upstream varying-column filter and 16-column cap. Past-only/static
features never enter training. There is no synthetic mixing or augmentation.

Windows are left-padded with missing history and right-padded to the base's
native future length (512 in the registered checkpoint). The exact H future
steps are supervised; every future target is hidden from the forward pass, and
all covariates after H are missing. The loss reuses
`training.network.covariate_forward` with native causal normalization, asinh
units and nine quantiles 0.1–0.9. It averages pinball over **observed horizon
cells and quantiles**, with history weight zero. The pointwise mask also handles
H that is not divisible by the patch width; upstream's auxiliary loss has a
patch-level mask. Up to four compatible grouped windows form a microbatch;
gradients accumulate with their observed-cell weights to the effective batch.
`--batch-size`, `--microbatch-size` and `--context` allow a bounded qualification
recipe. The baseline's inference context/config remains intact.

NorMuon/AdamW uses stage-4 base rates 0.002/0.0005, weight decay 0.0697 on hidden
matrices, muP base width 704 and global gradient clip 1. The multiplier scales
both base rates before upstream width/head scaling. Rewarm rises from 0.1 to 1
for 50 updates, then decays linearly to zero at the last planned update. A run
shorter than 50 updates consists only of the shortened rewarm. There is no
pretraining warm-start rate doubling. GPU mixed precision is not enabled by
this trainer; task-batch memory and update speed must be qualified on the GPU.

Inner evaluation defaults to updates 0/50/100/250/500. `--eval-updates 0,2,4`
changes the list; zero and the planned final update are always included, and
checks beyond `--max-updates` are omitted. Both inner files use the scoring
adapter's public `forecast_series` path, six raw-mode switches disabled,
`shared_categories=False`, quantile sorting and interpolation to the bundle's
requested levels. Actual scoring uses float64 `score_past` and `score_target`,
SQL's historical seasonal absolute-error scale, finite-loss reduction over
items/time per channel/quantile, mean over quantiles, then mean over channels.
Per-group scores are diagnostics; they are never averaged to form task SQL.
A wholly undefined channel invalidates the aggregate, matching fev. SQL is the
registered metric; `SQL` and `{"name":"SQL","epsilon":...}` are supported.
Other metrics/configurations are rejected explicitly before training.

Model targets are restored as float32. Inner known covariates are restored as
object arrays in the manifest's `known_future` column order before Simons applies
its column cap and per-window conversion. This preserves booleans, numeric nulls,
text nulls and literal strings such as `"None"`; it never substitutes the native
training category codes. No-covariate blocks pass `None`. Scoring fields retain
their float64 precision independently of model inputs and training material.
Offline parity checks exercise both the default NumPy formatter and unformatted
data, text nulls versus literal categories, NumPy boolean training values,
float64 seasonal scales erased by float32, and reloaded selected exports.

The registration overrides A5's earlier combined-gain draft: an adapted
candidate needs **at least 2% gain on each inner window** relative to update 0.
Choose the qualifying candidate with the smallest arithmetic mean of the two
scores; ties keep the earliest update. Early stopping counts two consecutive
checks without at least 0.5% mean-SQL gain over the last significant best mean.
Smaller gains accumulate against that best until the threshold is reached.
A zero baseline cannot qualify an adapted candidate by percentage gain.
Any non-finite forecast, non-finite overall score or non-finite loss/gradient
selects the frozen base, even if an earlier candidate qualified. Selection is
**per task**: items too short for the two inner windows contribute neither
training material nor inner scores, and every item uses the task's single set
of selected weights. Excluded IDs and their exact reasons remain in the receipt;
the input bundle's `fallback` metadata does not override the trainer's task-wide
selection. Some excluded items do not prevent adaptation. If all items are
excluded, the trainer keeps the frozen base with reason `all_items_excluded`.
Empty/unsampleable training also keeps the base, with reason `empty_training_set`.

Outputs are `config.json`, `model.safetensors`, `adaptation.json` and
`checkpoints/checkpoint-{slot-0,slot-1,final}.pt`. The config bytes are preserved
from the base. A frozen decision copies both base files byte for byte; an
adapted decision exports the best candidate, rather than the final iterate.
Load either with `Simons.from_pretrained(output)`; for raw scoring apply the
same six switches as `SimonsAdapter`. Published processing remains available
in the preserved config for a separate system experiment.

`adaptation.json` schema version 1 records:

- `identity`: bundle fingerprint, base file hashes, full effective recipe, seed,
  hashes of patched Simons Python sources and Torch/NumPy/device/thread runtime;
- `inputs`: manifest hash, every payload hash/size, base hashes, task and phase;
- `recipe`, `seed`, `parameters`, `sampling`, `objective`, `optimization`:
  actual settings, parameter names/counts, sampling eligibility, peak group rates;
- `evaluations`: ordered `{update, wall_seconds, windows}`; `wall_seconds` measures
  each two-window evaluation separately. Each of the two windows has its
  overall `score`, per-channel scores, per-item/block `groups` and any failure
  `reason`; non-finite numeric results are JSON null;
- `decision`: selected update, `adapted|frozen_base`, reason, stop/patience state;
- `training_wall_seconds`: cumulative time inside training updates, including a
  failed non-finite attempt, excluding evaluation and checkpoint/export overhead;
  divide by `updates_run` for mean update cost when all attempts succeeded;
- `peak_cuda_memory_bytes`: peak Torch allocated memory on the selected CUDA
  device, reset before loading the base and retained as a maximum across resumed
  segments; JSON null on CPU. This is allocated tensor memory, not reserved
  allocator memory or total device usage. CUDA synchronization brackets training
  updates and evaluations so their wall times include completed device work;
- `updates_run`, `wall_seconds`, `last_update`, `exposure`: successful committed
  updates, measured loop wall time, last finite loss/unclipped gradient norm and
  update `wall_seconds`, and observed history/horizon/known-covariate scalar exposures plus grouped-window
  count. Horizon cells count once, not nine times for quantiles. Counters resume
  transactionally and exclude replayed work after the last checkpoint; physical
  replay work lost in a crash is not measurable from this receipt. Training and
  evaluation timings follow the same committed-work accounting;
- `excluded_items`, `ineligible_training_groups`, `export`: coverage and exact
  selected-file hashes. Unique-corpus observed values describe the eligible
  sampling corpus, not the distinct labels actually drawn.

Every evaluation, every 50 updates or ten minutes, and every new runner nonce
publishes a hashed/fsynced alternating checkpoint. Intervals are configurable
with `--checkpoint-updates` and `--checkpoint-seconds`. Checkpoints include the
optimizer, current model, best candidate, selection history, counters and RNG
state. Sampling is stateless per (seed, update), independent of inner labels;
changing inner targets can affect selection, never the gradients at a given
update. Resume requires the same verified bundle/base/recipe/software/runtime.
Replayed updates and evaluation scores match in the same runtime. Export bytes
and deterministic receipt fields match uninterrupted execution; measured times
and peak memory naturally differ. CUDA determinism is requested explicitly and
must be qualified on the actual hardware/software stack.

If `$RUN_DIR/checkpoint-request` contains a new nonempty nonce, the next update
boundary saves state before atomically writing `checkpoint-done` with that
nonce. Without `RUN_DIR`, the output directory is used. A restart repairs a
missing acknowledgment even when the save already committed. Completion is
checkpointed before export publication, so an interrupted publication is
repairable by rerunning the same command. Corrupt input hashes and identity
mismatches fail closed; they are configuration/integrity errors, not coverage
fallbacks.

## Adaptation pilot guest job (AD5)

Generate the job locally; generation reads the patch bundle and makes no cloud
calls. Supply an immutable repository revision containing the bundle builder and
yardstick runner, the pin in `third_party/simons/PINNED`, and the SHA-256 of the
operator's **shifted development task YAML**. There is no default task file or
implicit cohort. The job passes exactly one requested name to each bundle and
scoring command, including names such as `fred_md_2025/macro`.

```bash
python scripts/gcp/jobs/make_adapt_job.py \
  --name adapt-pilot-01 --repo-sha "$REPO_SHA" --upstream-pin "$SIMONS_PIN" \
  --tasks-uri gs://BUCKET/pilot/shifted.yaml --tasks-sha256 "$TASKS_SHA256" \
  --names epf_de ercot_1H fred_md_2025/macro rohlik_sales_1W hospital_admissions_1D \
  --hub-weights abel-lab/Simons-25M --hub-revision "$HUB_REVISION_40_HEX" \
  --params head all --lr-multipliers 0.02 0.10 --seeds 11 29 \
  --max-updates 500 --processing both --output /tmp/adapt-pilot-01.sh
```

Alternatively use `--checkpoint-uri gs://BUCKET/export-prefix`, containing
`config.json` and `model.safetensors`; it is mutually exclusive with the Hub
source. Hub revisions must be full 40-character lowercase commit IDs. Defaults
are the two parameter arms, rates 0.02/0.10, seeds 11/29, 500 updates and both
outer processing modes. `--processing raw` or `published` runs only that mode.
The registered inner selection and evaluation schedule remain trainer defaults;
seeds are reported individually and are never selected by outer performance.

The generated Bash job implements the `RUN_DIR` / `STATE_URI` guest contract in
`docs/gcp.md`. One Python 3.12 environment, provisioned through uv when needed,
uses stage-1 torch 2.8.0+cu128, CUDA 12.8 and `requirements-yardsticks.txt`. The
guest verifies the repository revision, clones Simons at its upstream pin, checks
and applies every embedded patch through `prepare.py`, and installs that tree
editable alongside this repository. It verifies the dependency pins, CUDA, and
both the imported Simons and trainer paths, then records pip freeze and runtime
facts before downloading inputs or loading datasets.

Each task builds one bundle, scores the frozen base once per requested mode,
then runs the Cartesian product of arms, multipliers and seeds from that base.
The trainer automatically resumes its own checkpoint slots. Only a selected
adapted export is scored; a frozen decision writes a reuse receipt pointing to
the already measured frozen score. Failures are recorded per fit/mode, remaining
fits and tasks are attempted, and any failed bundle, fit or score makes the final
exit nonzero. Failed or incomplete work is retried by rerunning the same job.

Small evidence lives under `$RUN_DIR/state/adapt/<name>/`:

```text
inputs.json, tasks.yaml, checkpoint-sha256.json, environment.txt
results.csv, completed.json
tasks/<UTF-8-task-name-as-hex>/
  bundle-manifest.json, bundle.log
  frozen/<mode>/{status.json,run.json,summaries.csv,score.log}
  fits/<params>-lr<multiplier>-seed<seed>/
    fit.json, adaptation.json, train.log, checkpoints.json
    <mode>/{status.json,run.json,summaries.csv,score.log}  # adapted selection
    <mode>-reuse.json                                # frozen selection
```

`results.csv` has one row per task/arm/seed, including failed fits, with columns:
`task`, `params`, `lr_multiplier`, `seed`, `status`, `selected`, `selected_update`,
`decision_reason`, `inner_0_window_0`, `inner_0_window_1`,
`inner_selected_window_0`, `inner_selected_window_1`, `outer_sql_raw`,
`outer_sql_published`, `frozen_sql_raw`, `frozen_sql_published`,
`trainable_parameters`, `training_seconds`, `inner_evaluation_seconds`,
`outer_evaluation_seconds`, `peak_gpu_memory_bytes`, `error`.
The outer columns describe the selected model, so frozen decisions equal the
frozen control. Missing/failed/unrequested scores are empty. Training and inner
evaluation seconds come from the trainer receipt; outer seconds sum the adapted
scorer subprocess durations. Frozen reuse adds zero outer seconds. Each frozen
scorer's duration is retained separately in its status receipt. Peak GPU memory
is the trainer's peak allocated CUDA bytes, with the same limitations described
above; it does not include the separate outer scorer process.

Bundles, exports, trainer checkpoints, prediction arrays and dataset caches stay
outside `state/`. Disposable work uses a fixed `/tmp/demand-forecast-adapt-<hash>`
path determined by inputs and the artifact URI, preserving scorer checkpoint
identities across replacement guest labels. Durable checkpoint generations and
selected exports are uploaded as immutable hash-addressed objects under
`${STATE_URI%/}-adapt-artifacts/<name>/blobs/<sha256>`, a sibling prefix in the
same bucket, excluded from the runner's state restore. `checkpoints.json` and
`fit.json` publish references only after uploads succeed. Restore downloads and
verifies those referenced objects; lost bundles are rebuilt and their complete
manifest hashes must match. Object lifecycle cleanup is an operator task.

The supervisor forwards guest checkpoint-request nonces to the active trainer's
private control directory. It acknowledges a matching nonce only after checkpoint
uploads and atomic reference publication; normal trainer checkpoints are uploaded
while training continues. This provides the last successfully uploaded generation
on preemption, without assuming a final shutdown upload will finish.
`completed.json` records the full inputs, payload hashes, base hashes, bundle
versions/fingerprints, per-fit and per-score status, and hashes of the small state
artifacts. A successful, fully verified completion exits before environment
installation or downloads. A changed input identity or changed base/bundle content
under the same name is refused. A damaged completion is rebuilt from independently
verified evidence; finished fits and scores are reused.
If both base identity records are absent while task evidence survives, the job
refuses reuse. A missing bundle manifest is recoverable only when a surviving
completion hash or verified trainer receipt establishes its original exact hash;
frozen scores alone cannot establish that bundle identity. Restore the original
identity evidence or use a new name instead of rebinding surviving results.

The job requires bundle `schema_version` to be the integer **2** and preserves
the exact manifest bytes. It checks the manifest's task/phase and payload hashes
before frozen scoring or fitting. The embedded trainer patch 0004 independently
checks schema 2, the manifest fingerprint, both inner-file versions and the native
training representation. The job records bundle schema 2 separately from
`adaptation.json` schema 1 and its own completion schema 1. Unsupported bundles
fail the task before any scoring or fitting; remaining tasks are attempted.
The caller must provide the
registered shifted development windows: a supplied YAML hash authenticates its
bytes, not its cohort declaration. The job never substitutes official task YAML
or opens another task's dataset to establish that declaration. Offline tests
cannot establish CUDA determinism, real dataset coverage, GPU cost/throughput,
package installation on the guest image, or live preemption durability.

## Known scoring question

Should the scorer's handling of nullable text covariates change? This remains
open and is not fixed here. Any such scoring change belongs in a separate PR
with before/after scores on real tasks, because it can move every model's
reference rows and break leaderboard comparability. Adaptation follows the
scorer accepted for those comparisons.
