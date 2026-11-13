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
that item's length), clipped to `[0, item_length]` as in fev. For a date cut-off,
T is the number of observations with timestamp **at or before** the requested
date. An off-grid date is valid: a Friday cut-off in a Sunday weekly series
leaves the preceding Sunday in history and forecasts from the following Sunday.
The builder uses pinned fev 0.10.0's Arrow cut-off calculation, including its
source timestamp precision and timezone checks. Only dates fev cannot cast or
resolve raise `unresolvable date cut-off`.

The first cut-off is `task.cutoffs[0]`, unchanged. Integer steps (including the
default horizon) and frequency-string steps do not shift this first cut-off,
even for anchored monthly offsets. fev rejects frequency-string steps paired
with integer cut-offs. Date cut-offs support either step form.

The inner forecast indices are T−2H and T−H. All indices are zero-based and all
end indices are exclusive. Splits use each item's validated timestamp grid;
calendar frequencies such as months use observed steps, not fixed timedeltas.
Null, duplicate, decreasing or irregular timestamps fail. Native fev validates
each item's frequency when loading and rejects an irregular dataset before any
window or bundle is built. Target and declared
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
An item ending at/before a date cut-off has T equal to its length; one starting
after it has T=0. Both are excluded as `insufficient_outer_history`, matching
fev's first-window context/horizon filter. The same reason covers too little
history or an incomplete outer horizon in other items.
An entirely infeasible task is a valid `frozen_fallback` manifest with empty
training and inner rows; fev's window raises when all items are filtered out.
These are structural checks, not sampler eligibility:
the adaptation trainer must still enforce its registered minimum observed
history (32 points), valid gradient windows and any model context limits.

The 96-task shifted pilot cohort was inspected from YAML only:

| Tasks | Count | Cut-off form | Step form | Supported |
|---|---:|---|---|---|
| All remaining tasks | 92 | Negative integer | Omitted; defaults to integer H | Yes |
| `rohlik_sales_1D` | 1 | Date `2023-10-20` | Frequency string `14D` | Yes |
| `rohlik_orders_1D` | 1 | Date `2022-06-30` | Frequency string `61D` | Yes |
| `rohlik_sales_1W` | 1 | Date `2023-05-05` | Frequency string `56D` | Yes |
| `rohlik_orders_1W` | 1 | Date `2022-07-25` | Frequency string `56D` | Yes |

No unsupported cut-off/step forms were found. This checks input forms;
dataset feasibility requires loading each task and remains unverified by this
YAML-only inventory.

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
The date cut-off fix keeps schema 2 and preserves in-range integer cut-off
origins, payload bytes and input hashes. Rebuilding an old integer bundle still
changes its fingerprint because `versions.builder_sha256` records the changed
builder source. Preserving that old fingerprint would discard existing source
provenance. Out-of-range integer origins now record fev's clipped index;
those items remain excluded.

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
short/ragged items, calendar boundaries, slash names, integer/inclusive off-grid
date cut-offs and step forms against real fev first windows, rejected irregular
timestamps, covariate roles, causal perturbations, byte determinism, adapter calls,
and real fev SQL parity. No weights, cloud services or benchmark downloads are
needed. Real cohort feasibility and model-training behavior require separate
verification by the subsequent tasks.

## Known scoring question

Should the scorer's handling of nullable text covariates change? This remains
open and is not fixed here. Any such scoring change belongs in a separate PR
with before/after scores on real tasks, because it can move every model's
reference rows and break leaderboard comparability. Adaptation follows the
scorer accepted for those comparisons.
