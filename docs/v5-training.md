# Native v5 training jobs

Every cloud run must be registered in the stage-2 plan before launch, with its
code revision, configuration or checkpoint digest, state prefix, job URI and
cost ceiling. Generate scripts offline and leave launches to the operator.
Q and S1 publish experiment artifacts only.

Generate a training job from a local YAML configuration:

```bash
python scripts/gcp/jobs/make_v5_job.py \
  --name leg-1-job --action train --repo-sha <40-character-revision> \
  --config leg-1.yaml --leg leg-1 --output /tmp/leg-1-job.sh
```

The guest supplies absolute `RUN_DIR` and the bucket `STATE_URI`. It alone
restores, copies and marks the job. Use one state prefix per train, qualify or
export unit, and one prefix per T1 arm. Allow only one writer per prefix. A fresh job
name and job URI are required for every evaluation extension, because the guest
refuses a job URI already recorded as DONE.

The script checks a valid `unit.json` against its own SHA-256 before installing
anything. It then fetches and verifies the exact repository revision, creates a
seeded Python 3.12 environment with uv, installs torch 2.8.0 from the cu128 index,
installs `requirements-v5.txt` and the editable corpus extra, and runs `pip check`.
The fingerprint of sorted `pip freeze --exclude-editable` lines is part of the
training identity. A changed dependency environment cannot resume an old run.
`requirements-v5.txt` is authoritative for the guest environment. Its pins were
resolved for Linux x86_64 and Python 3.12; the command, torch constraint and
nvidia-pin filtering are recorded in its header. Regenerate only deliberately
and record the environment change in the stage plan: existing states become
unresumable under the changed environment by design.
The v5 export adds `safetensors==0.6.2`, which gives jobs generated from the
merged export revision onward a new environment hash. The registered S1 and Q
units still fetch older revisions with their own requirements files.

For a source with `kind: upstream`, add `--upstream` to job generation. For
train/qualify the generator parses YAML locally and requires the upstream source
and `--upstream` together, alongside the existing corpus checks; T1 refuses
`--upstream`. Before anything is installed, the guest checks that the fetched
training repository's `third_party/simons/PINNED` holds the pin embedded during
generation, clones Simons (without blobs it does not need) into volatile scratch
storage at that pin and verifies HEAD (exit 2 on either mismatch). After the
environment fingerprint has been taken it exports `<upstream>/src` on
`PYTHONPATH` for the job and its DataLoader workers. It installs no upstream package and does not run `prepare.py`,
so dependency installation, `pip check` and the environment fingerprint are
unchanged by this option.
The source lazily imports its standalone synthetic APIs and yields one undated
item per panel, using local random streams seeded by one caller draw. It
generates 2,048 history steps plus 288 future steps. The upstream-only
`plain_share` option is the probability of returning the base target series
alone. For the other panels, the source applies numeric covariate effects
and assigns each covariate a known or past role. The default `plain_share: 0.0`
keeps the covariate effects on every panel and preserves the draw sequence.
Base sampling uses `families=("kernel", "sarima")`, `noise=0.0`, `missing=0.1`,
`min_observed=96`, `weights=None` and `pad=0.0`; leading padding is off because it
can leave the history entirely missing before covariate-effect normalization.
The sampler applies `control: missing_as_zero`. To run the upstream integration
tests locally, set `SIMONS_UPSTREAM` to a clean clone at the pin.

For corpus jobs using the [4.2 training view](corpus-v4.md#version-42-a-training-view-with-bounded-row-groups),
add `--corpus-uri gs://example-bucket/corpus-v4.2 --corpus-digest <hex>`.
Use the `corpus_digest` in the regroup job's
`$STATE_URI/regroup/<name>/completed.json`. It covers the published `univariate/`
and `covariates/` trees. For a 4.1 run, use its own URI and listing digest.
Corpus sources in the YAML must have `root: null`. Only `univariate/` and
`covariates/` are copied. The script validates the relative name, size and CRC32C
listing digest, clears the volatile corpus cache before copying, and checks the
listing again, including object generations. It writes `corpus/manifest.json`
only after the second listing agrees. Replacing an object during the copy is a
failure. Copying the corpus again after every preemption costs time and money;
Q records that copy time so the stage-2 plan can budget it.

An initial export checkpoint applies only to `train`:
`--init-uri gs://bucket/initial.pt --init-sha256 <hex>`. Its bytes are verified
before training, and its digest participates in the training identity.

## Scoring export and parity

Export one checkpoint in the training environment and check the resulting
directory in the scoring environment.

```bash
python -m demand_forecast.models.v5.export write --checkpoint checkpoint-final.pt --out v5-export
python -m demand_forecast.models.v5.export check v5-export
```

The destination must be absent or empty. The export is published as one
directory after verification. It contains the checkpoint's selected raw or EMA
state dict in `model.safetensors`, provenance and file digests in `config.json`,
and eight replayable cases in `parity.npz`. Load it with
`demand_forecast.models.v5.export.load_export` and use `require_parity` before
scoring. Each case stores its panel and requests, every collated batch field,
and the model's scaled CPU float32 quantiles and gates. The archive opens with
`numpy.load(..., allow_pickle=False)`.

The check reports exact comparisons for discrete fields, names, shapes and
dtypes. It shows the largest fraction of the allowed error for each case's
floating batch and output fields. Ratios above 1 or exact mismatches give exit
status 1. Batch tolerances are `atol=rtol=1e-6`. Output tolerances are
`atol=rtol=1e-4` on CPU and `atol=rtol=2e-3` on another device
(`check v5-export --device DEVICE`). NaN positions must match.

Run `bash scripts/ci/v5_parity.sh` to run the Linux CI gate locally. It uses
Python 3.12, CPU torch 2.8.0 and the relevant training and scoring pins. It
creates seeded, untrained tiny exports with and without `increment_spread` in
the training environment and checks them in the scoring environment. If the
gate fails on merged code, scoring work pauses while the environment choice
is revisited.

**Optional input presentations.** Set `sampler.derive` to `[increment_spread]` to
replace a dynamic category's second value statistic with the root mean square
of its one-step target increments divided by `sqrt(2)`. A pair counts when both
adjacent cells have the active label and finite targets and the later cell is
strictly before the context cell or the forecast origin for a known horizon.
With zero pairs, the statistic is zero. For independent noise around a constant
level it estimates the noise spread. For AR(1) noise it estimates
`sigma * sqrt(1 - rho)` at constant level, with half the squared baseline step
added to the squared statistic when the level moves.

Enable `lag_rows` to give a retained past-only numeric column a known row
named `<column>~lag` when its recent history supports a delayed effect. The
scan uses up to 512 cells before the statistics cutoff and considers delays
up to 64 cells. A lag row takes a free feature slot and uses the known-row
gate floor. At each origin its horizon shows shifted values only when their
source cells are finite and strictly before that origin. Forecast evidence
reports the lag row separately as `derived:<column>~lag`.

Enable `onset_rows` to give a retained dynamic category a numeric row named
`<column>~onset` when `value_stats` are enabled. The row uses the category's
role and takes a free feature slot. Lag and onset rows compete for those slots
by descending score and then by name. The onset row is built only when its
profile has an observed value among the context cells before the statistics
cutoff. Otherwise the category gets no onset row, takes no slot and has no
`derived:<column>~onset` evidence entry.

The onset row starts at an observed change of label and covers up to 32 cells
of the run. A missing label ends the run. Each cell averages earlier completed
responses at the same label and age, subtracting the mean of the finite targets
in the preceding `min(season, 8)` cells at each earlier onset. The mean is
shrunk by `n / (n + 2)`, and at least one response is required for an observed
value. The row uses scaled target units. A known category's row has the known
gate floor and can use horizon labels to place onsets. Its response history
freezes at each origin. A past category's row has the past gate and appears in
the context only. Forecast evidence uses `derived:<column>~onset`.

Enable `naive_rows` to present classical forecasts of the kept target as
known numeric rows. This applies to a designated target or the sole target
of an item. For a joint request on a multi-target item, the presentation is
inactive. For a season of length `S`, let `W = S * ceil(8 / S)`. The
`<target>~last` row uses the last finite value within `4W` cells.
`<target>~season` uses the last finite value at the served cell's phase within
four seasons when `S > 1`. `<target>~season2` uses the same rule with the long
lag when it exceeds `S` and the item has at least that many cells before the
first origin. `<target>~drift` extends the seasonal value (the last value when
`S = 1`) by the time since its source times the mean per-step seasonal
change. The mean uses pair endpoints in the last `W_d = S * ceil(64 / S)`
cells and requires at least `W / 2` finite pairs.

Each context cell uses sources strictly before itself, and each horizon
uses sources strictly before its own origin. Sources can precede the context
window and use the target scale fitted at the statistics cutoff. Cells with
insufficient sources are missing. The rows use the known-row gate floor and
appear as `derived:<row name>` in forecast evidence. Each row reserves a
feature slot, skipping a name that collides with an existing column. Base
features keep their usual ranking in the remaining slots, and lag and onset
rows can use slots left free. Turning off the presentation restores the
displaced base features.

The switch defaults to an empty list. The export records enabled names and
the adapter uses them for scoring. Use the `no_derived` scoring variant to
disable them on the same weights.

**Calibrated score for static categories.** Set `sampler.screen_fair_spread` to
`true` to score a static category attribute against 32 seeded permutations of
its labels. The default score is the share of spread between label groups, which
is biased upward when a label has two or three items, so an unrelated attribute
can pass the screening level. With the option, the attribute scores above zero
only when its largest component exceeds the same component under every
permutation. Permutations act within the items that are valid for each summary
entry, and a multi-label attribute is scored by membership. The option defaults
to `false`. The export records it and the adapter uses it for scoring; an export
written without the key replays with the default score. The static diagnostics
always use the calibrated score.

### Export as a resumable job

Generate a job from a checkpoint's storage URI and SHA-256:

```bash
python scripts/gcp/jobs/make_v5_job.py \
  --name scoring-export --action export --repo-sha <40-character-revision> \
  --member-uri <checkpoint-gs-uri> --member-sha256 <checkpoint-hash> \
  --output /tmp/scoring-export.sh
```

The generator requires both member flags. Export accepts no configuration,
corpus, upstream, init, leg or T1 arguments. After the unit preflight and
repository verification, the script downloads the checkpoint to scratch and
compares its SHA-256 before creating the Python environment. A digest mismatch
exits 2.

Run the same export action with a local checkpoint:

```bash
python -m demand_forecast.pretrain.v5.job export \
  --run-dir /absolute/run --code <revision> --job-sha256 <script-hash> \
  --env-sha256 <freeze-hash> --name scoring-export \
  --checkpoint-file checkpoint-final.pt --checkpoint-sha256 <checkpoint-hash>
```

The action checks the checkpoint digest before touching the run directory.
It writes and checks the export under `staging/` with `load_export` and
`require_parity`, then copies the three verified files into
`state/export/NAME/` and publishes `unit.json` last. The unit identity hashes
the canonical JSON of the action, name, code revision, environment hash and
checkpoint hash. A matching valid unit is reused. An invalid manifest is
removed before rebuilding.

## Running a local training unit

The module also runs directly under the guest contract:

```bash
python -m demand_forecast.pretrain.v5.job train \
  --run-dir /absolute/run --code <revision> --job-sha256 <script-hash> \
  --env-sha256 <freeze-hash> --config leg-1.yaml --name leg-1
```

`--stop-after N` on train and T1 is available for small tests. Exit 0 means a
validated `unit.json` has been published. Exit 75 means stopped unfinished,
including training's TERM and stop-after outcomes. Unit failures exit 1,
remove the completion marker, and record the reason and traceback in
`failure.json` and the job log. A local export input digest mismatch exits 1
with a stderr diagnostic before publication begins. It leaves an existing valid unit
untouched and writes no `failure.json`. Script revision or corpus
validation failures exit 2.

The guest (`scripts/gcp/guest-run.sh`) publishes these bucket outcomes after
the job exits, assuming the final state copy and marker upload succeed:

| Job exit status | Guest marker | Marker body |
| --- | --- | --- |
| 0 | `DONE` | `exit_status=0` |
| 75 (unfinished unit) | `FAILED` | `exit_status=75` |
| 1 | `FAILED` | `exit_status=1` |
| 2 (script validation failure) | `FAILED` | `exit_status=2` |
| 143 or 130 (guest TERM or INT trap) | `FAILED` | `exit_status=143` or `exit_status=130` |
| 124 or 137 | `TIMEOUT` | `exit_status=124` or `exit_status=137` |

A failed final copy prevents `DONE`: a successful job is then recorded as
`FAILED` with `exit_status=74`. Marker or log upload failures also make the
guest exit 74, so inspect its log if publication is incomplete.

On preemption, the metadata watcher first publishes `PREEMPTED` with
`exit_status=interrupted` and `source=watcher`, then requests a checkpoint and
attempts a bounded state copy while leaving the job running. The platform hook
is a marker-only fallback (`source=hook`, also `exit_status=interrupted`). If the
job finishes during the watcher's copy and the copied state inventory is
unchanged, the watcher can archive `PREEMPTED` and reconcile it to the final
`DONE`, `FAILED` or `TIMEOUT` using the same exit-status mapping above.

Operator relaunch rule: first check the unit directory in the bucket for
`failure.json`. If it is absent, relaunch the **same job URI on the same state
prefix** after `PREEMPTED`, after `TIMEOUT`, or after `FAILED` whose body says
`exit_status=74`, `75`, `130` or `143`. This resumes unfinished work or retries
publication of an already completed unit after a failed final copy.

`FAILED` with `exit_status=1` or `2`, or **any outcome with a unit
`failure.json`**, is a real failure: read that file (if present) and the log;
do not relaunch unchanged. In particular, exit 2 for a wrong revision or corpus
listing mismatch is a real failure even before a unit `failure.json` exists.
For other `FAILED` statuses, inspect the log and diagnose before deciding.

At startup the guest refuses a `DONE` for the same job URI, and also refuses a
legacy `DONE` without a job URI. A `DONE` for a different URI is archived before
admitting the new job. `FAILED`, `TIMEOUT` and `PREEMPTED` are read and archived
as well, including outcomes for the same URI; the guest does not block those
URIs or automatically apply the operator's retry rule. It restores state only
after this admission and archival step. Inspect the current marker before
deciding to launch.

State layout:

```text
RUN_DIR/
  state/
    train/LEG/{identity.json,checkpoint-slot-*.pt,history.jsonl,telemetry.jsonl,
               checkpoint-final.pt,train.json,unit.json}
    qualify/NAME/{identity.json,qualify.json,unit.json}
    export/NAME/{config.json,model.safetensors,parity.npz,unit.json}
    t1/PATH-ARM-sSEED/
      train/{identity.json,checkpoint-slot-*.pt,checkpoint-final.pt,train.json,...}
      eval-N-M/{checkpoint-final.pt,train.json,result.json,unit.json}
  staging/          temporary checkpoint writes, qualification trials and export artifacts
  v5-cache/         HOME, uv/Python/pip/Hugging Face caches, repository and venv
  corpus/           volatile corpus cache, rebuilt on every active invocation
  checkpoint-done   guest request acknowledgement
```

Everything needed to survive a machine is under `state/`. The corpus,
repository, environment and configuration copies are disposable caches outside
it. Training checkpoint temporary files, qualification trials and export verification
use `staging/`. State receives the completed qualification report and verified
export files. Each unit manifest is written last with atomic rename and fsync
and records artifact sizes and SHA-256
hashes. Evaluation directories include copies of their training checkpoint and
manifest, so they remain valid when copied locally; this uses additional disk
and bucket space per evaluation. There are no parent-relative paths or symlinks
in a manifest. Checkpoint bytes also carry the training loop's SHA-256 footer.

**Training loss.** The step projection, quantile head and training loss use
float32 under mixed precision so small quantile gaps survive at large scaled
levels. Set `optim.label_cap` to a finite positive value to exclude supervised
cells whose absolute scaled labels exceed it. The pinball loss averages the
remaining cells. When every cell is excluded, the sparsity and balance penalties
still drive the update. With a cap set, each `history.jsonl` record reports the
total `capped` cells across all batches in its logging window. Qualification
applies the cap in its training trials and worst-batch measurement.

**Corpus sampling.** The default `folder_power: null` draws uniformly across
admitted rows. With a power in `[0, 1]`, the source draws a `<kind>/<dataset>`
folder with weight `values ** folder_power`, using the number of numeric entries
in its admitted groups. It then draws a row uniformly within that folder.
Power 0 gives every admitted folder equal proposal probability. Power 1 makes
that probability proportional to its entry count. Power 1 is therefore not
the default draw: a folder with longer series or more columns is proposed more
often than its row count alone would give. The sampler counts refused
rows in `skips`, so the realised folder shares can differ from the proposal
shares. Source weights apply to panel draws. A typed panel can supply several
episodes, while corpus and upstream panels supply at most one.

**Calendar effect of generated panels.** On daily typed panels with the calendar
mechanism, each target's calendar effect adds day-of-week, day-of-month and month table entries. The generator
draws day-of-month after the two existing tables, preserving their values for
the same seed.

For Q, use `--action qualify --config q.yaml --name q-typed`. Run a separate
unit for typed 0.5 + corpus 0.5 with the corpus options. Both the generator and
`job.py qualify` accept `--models full|t1|both`, `--workers N[,N...]`,
`--updates N` and `--no-inference`. The defaults run both models with 8, 16,
32 and 46 training loader workers, 200 batch attempts per trial and a 30,490-series
inference probe per model. `--no-inference` omits the probe and retains parameter
counts. `--inference-workers N` overrides the default of 16 on Linux hosts with
at least 18 CPUs and 0 elsewhere. The inference result records the count in
`workers` and any serial fallback in `worker_reason`, including when the
file-descriptor limit cannot accommodate the request.

`qualify.json` records cold index/first-batch/restore timings, excluded corpus
row share, admitted corpus rows per unparsed label (`unparsed_frequency_rows`),
trials at one and the configured number of origins, worst-batch forward and
backward, checkpoint write time and size, and inference. Each measurement has
`status` set to `ok`, `oom` or `failed`, with a `message` on error. The parameter
count is the exception: it carries a status only when counting fails. The job
publishes the report once at least one training trial succeeds. If all training
trials fail, the job writes `failure.json` and can be retried under the same name.
Restore timing uses the first successful worker trial's checkpoint. Checkpoint
write timing retries with later successful worker trials' checkpoints until
one works, and reports the last error if every attempt fails.
For training trials, `supervised_cells_per_second`
multiplies supervised cells per successful update by updates per second after
warmup. It is null when the timed tail has no successful updates.
`peak_gpu_bytes` includes the optimiser and averaged weights.

For training trials and inference, `host_memory_bytes` is the peak sampled
sum of process and descendant RSS in bytes. Shared pages can contribute to
more than one process's RSS. The value is null when RSS cannot be sampled,
including on hosts without `/proc`.
`host_memory_pss_bytes` uses the same sampling points and apportions shared
pages across the processes that map them. It is null without a complete PSS
sample. `host_memory_rusage_bytes` adds the lifetime RSS high-water figures
reported by `getrusage` for the process and its waited-for children. This
figure is in bytes on Linux and macOS and can include earlier work in the unit.
The operator uses Q to register the bases, batch shapes, workers, checkpoint
cadence, recovery bound and revised cost ceilings before S1.

Every batch report includes `exposure`, keyed by source kind (two sources of
the same kind are counted together). Its `panels`
counts returned panels, including those that yield no retained episode.
`episodes` counts retained episodes and `cells` counts their supervised
labels. At `exposure.corpus.folders`, each `<kind>/<dataset>` entry records
the same three counts. If every draw from a source or folder is refused, those
draws contribute only to the reason counts under `skips`. Each qualification
trial in `qualify.json` sums these counts across all observed batches.

On Linux, PyTorch loader workers pass batch tensors through file descriptors,
and prefetched batches can keep many descriptors open. Before starting workers,
the v5 loader tries to raise the process's soft open-file limit within its hard
limit and logs any increase. An `RLIMIT_NOFILE` error at startup means the soft
limit is still below the budget for the configured workers. Raise `ulimit -n`
or use fewer workers. During batch transfer, `received 0 items of ancdata` can
indicate that PyTorch ran out of file descriptors.

For S1, register a base with fixed episodes per batch, `max_cells: null`, and
`designate_target: true`. Generate one job per registered arm:

```bash
python scripts/gcp/jobs/make_v5_job.py \
  --name static-true-s7 --action t1 --repo-sha <revision> --base s1-base.yaml \
  --path static_category --arm true --seed 7 --episodes 200 --max-panels 1000 \
  --output /tmp/static-true-s7.sh
```

Repeat for `null` under its own prefix. Path-specific arms come from `t1.PATHS`;
selection has `true`, `null`, and `gates_one`. Its true-arm result includes forced
interventions. Evaluation uses the registered held-out seed 10,000, independent
of the training seed, and all arms share the same frozen evaluation digest.
The unit records path, arm, training seed, base-file hash, requested episodes and
max panels, and the actual replay digest in addition to its training identity.

**Cold-start episodes.** `sampler.cold_share` is the probability per attempt in
the support loop of trying a cold frame for a `static_category` or `multi_label`
path lead. It defaults to zero. The support rule rejects cold frames more often
than ordinary frames because peers need enough observations before the earlier
cutoff. Use the attempt and plan counts in `report["cold"]` to gauge realised
cold-lead exposure. Below `1.0`, the realised share can differ from `cold_share`.
`sampler.cold_steps` is the inclusive range of grid positions from the item's
start to its single origin and defaults to `(3, 4)`. The origin is also the
statistics cutoff. Only the lead can be cold, so a panel that supplies eight
episodes can contribute at most one cold episode. Both static T1 paths use one
episode per panel and a cold share of one in every arm, giving every path
episode a cold lead and drawing more panels per training batch.

The `several_targets` path sets `sampler.horizon` to `(1, 5)` for both training
arms and their shared held-out evaluation: requested horizons of 1 to 4 steps,
because the sampler truncates a log-uniform draw and the range's upper bound is
exclusive.

Both `event_response` arms and their shared held-out evaluation enable
`sampler.first_event_only`. A path episode qualifies only if the complete
event flag has no event from that item's start up to the forecast origin,
including positions with missing targets or flags in the exposed data. An
event at the origin is part of the horizon. The existing visible horizon
event and same-label peer requirements still apply. The option defaults to
`false` and applies only to `event_response` path episodes.

After a stop between training and evaluation, rerun the same job: the completed
training is validated and reused. For an inconclusive result, double both
`--episodes` and `--max-panels`, give the script a new name and job URI, and use
the same arm's state prefix. Training parameters and identity are unchanged;
only the new `eval-N-M/` is evaluated and published. One extension is allowed by
S1; unresolved results go back to the owner. Copy each arm's evaluation directory
locally for `t1-compare`: provide the **byte-exact base file used to generate
the arms**, code, path, seed and evaluation parameters and one `--arm NAME=DIR`
per required arm. The base file's SHA-256 is compared, so even changes only to
`workers`, `device` or YAML formatting are refused, although `workers` and
`device` are excluded from the training identity. Each arm must name an existing
real directory, not a symlink. For `selection`, the top-level `compare.json`
outcome combines the deciding comparison and `selection_report.outcome`: either
failure means fail; otherwise either inconclusive means inconclusive; only two
passes mean pass. This job generator does not perform that local comparison.
S1 also requires plan 4's fitted-table checks inside its registered ceiling.

On a preemption, training acknowledges the guest's request only when its last
checkpoint write plus the longest recent update fits within six seconds. If the
acknowledgement arrives within the guest's eight-second wait and its bounded
copy completes, recovery uses that requested checkpoint. Otherwise it uses the
newest valid slot from a completed periodic copy, falling back to the older
valid slot if needed. With successful copies within their time limit, the bound
is `C + P + 2*D` plus one update: C is checkpoint cadence, P is the guest copy
interval (`guard-copy-interval`), and D is the periodic copy limit (540 seconds
unless `GCP_PERIODIC_COPY_TIMEOUT` changes it). Convert all terms to the same time
unit. Failed copies permit greater loss; inspect guest logs. Q estimates copy
lag by comparing checkpoint save times in `telemetry.jsonl` with bucket object
upload times. The guest does not measure copy durations, so that lag is an
estimate and must be recorded with its conditions before S1.
