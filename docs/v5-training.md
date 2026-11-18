# Native v5 training jobs

Every cloud run must be registered in the stage-2 plan before launch, with its
code revision, configuration, state prefix, job URI and cost ceiling. Generating
a script below is offline; launching it is the operator's work. Q and S1 publish
experiment artifacts only, not a model release.

Generate a training job from a local YAML configuration:

```bash
python scripts/gcp/jobs/make_v5_job.py \
  --name leg-1-job --action train --repo-sha <40-character-revision> \
  --config leg-1.yaml --leg leg-1 --output /tmp/leg-1-job.sh
```

The guest supplies absolute `RUN_DIR` and the bucket `STATE_URI`. It alone
restores, copies and marks the job. Use one state prefix per train or qualify
unit, and one prefix per T1 arm; allow only one writer per prefix. A fresh job
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
item with a generated target and numeric covariate effects per panel, using
local random streams seeded by one caller draw. It generates 2,048 history
steps plus 288 future steps and assigns each covariate a known or past role.
Base sampling uses `families=("kernel", "sarima")`, `noise=0.0`, `missing=0.1`,
`min_observed=96`, `weights=None` and `pad=0.0`; leading padding is off because it
can leave the history entirely missing before covariate-effect normalization.
Generator options remain typed-only; `control: missing_as_zero` is applied by
the sampler. Local tests use `SIMONS_UPSTREAM` pointing at a clone at the pin (the clean clone the
other upstream tests use; no preparation is needed).

For corpus jobs, add `--corpus-uri gs://bucket/corpus-v4.1 --corpus-digest <hex>`.
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

The module also runs directly under the guest contract:

```bash
python -m demand_forecast.pretrain.v5.job train \
  --run-dir /absolute/run --code <revision> --job-sha256 <script-hash> \
  --env-sha256 <freeze-hash> --config leg-1.yaml --name leg-1
```

`--stop-after N` on train and T1 is available for small tests. Exit 0 means a
validated `unit.json` has been published. Exit 75 means stopped unfinished,
including training's TERM and stop-after outcomes. Exit 1 records the reason in
the unit's `failure.json`; nonfinite training is a failure. Script revision or
corpus validation failures exit 2. A failure removes the attempted unit's old
completion marker; it does not declare another configuration complete.

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
    t1/PATH-ARM-sSEED/
      train/{identity.json,checkpoint-slot-*.pt,checkpoint-final.pt,train.json,...}
      eval-N-M/{checkpoint-final.pt,train.json,result.json,unit.json}
  staging/          temporary training checkpoint writes and qualification trials
  v5-cache/         HOME, uv/Python/pip/Hugging Face caches, repository and venv
  corpus/           volatile corpus cache, rebuilt on every active invocation
  checkpoint-done   guest request acknowledgement
```

Everything needed to survive a machine is under `state/`. The corpus,
repository, environment and configuration copies are disposable caches outside
it. Training checkpoint temporary files and qualification trials use `staging/`;
only the completed qualification report is copied into state. Each unit manifest is
written last with atomic rename and fsync and records artifact sizes and SHA-256
hashes. Evaluation directories include copies of their training checkpoint and
manifest, so they remain valid when copied locally; this uses additional disk
and bucket space per evaluation. There are no parent-relative paths or symlinks
in a manifest. Checkpoint bytes also carry the training loop's SHA-256 footer.

For Q, use `--action qualify --config q.yaml --name q-typed`; run a separate
unit for typed 0.5 + corpus 0.5 with the corpus options. `qualify.json` measures
full and T1 models, throughput and supervised/padded cells for worker counts,
summed host and peak GPU memory, one and configured origins, worst-batch forward
and backward, cold index/first-batch/restore times, excluded corpus row share,
checkpoint write time and size, parameter counts, and full-panel inference for
30,490 series. Q selects nothing automatically. The operator uses these
measurements to register the bases, batch shapes, workers, checkpoint cadence,
recovery bound and revised cost ceilings before S1.

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
