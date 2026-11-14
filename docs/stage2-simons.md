# Stage 2: Simons training on interruptible Google machines

Experiment 1 trains the released Simons-25M recipe from random initialization
on our corpus. The implementation uses upstream commit
`8e338e915d26c8960713889ef12ab1cf5c08d591`, with the complete patches under
`third_party/simons/patches/`. The `12m` variant copies the same five upstream
25M configurations and changes only `model.width` to 264. Seeds, micro-batches
and machine-local data/weight paths are explicit job inputs. Effective batch,
contexts, objectives, synthetic families/pools/banks, optimizer and curriculum
values remain upstream's values with the default `--recipe full`. In particular,
checkpoint-block counts are not increased to make a card fit.

| Job | Input weights | Data | Absolute updates executed | Durable product |
| --- | --- | --- | --- | --- |
| `stage1` | Random initialization | Synthetic only | 0–65,000 | Stage 1 checkpoint |
| `stage2` | Stage 1 final; fresh optimizer/read initialization | Synthetic only | 0–12,000 | Stage 2 checkpoint |
| `stage3` | Stage 2 final; fresh optimizer | Corpus + synthetic | 26,392–57,600 | Stage 3 checkpoint |
| `stage4` | Stage 3 final; fresh optimizer | Corpus + synthetic + long covariate pass | 53,600–57,600 | Stage 4 checkpoint |
| `stage4-long` | **Stage 3 final**; fresh optimizer | Same corpus/passes | 49,600–57,600 | Long-leg checkpoint |
| `export` | Stage 4 and long-leg finals | None | None | Uniform soup, then inference files |
| `qualify` | Named leg's normal starting weights | Named leg's normal data | Warm-up + 500 measured updates | Named JSON report |

The table above describes `full` for 25M/12M. The third leg intentionally skips context phase 0. Its first absolute update
is 26,392, not zero. The fourth and fifth legs branch independently from the
third. Run the first three in order, then each branch, then export. Each
predecessor must have both its completion marker and its referenced checkpoint;
a missing predecessor fails before training. A resumed leg uses its own model,
optimizer and absolute update position, without requiring the predecessor again.

Generate a job offline from the repository root:

```bash
python scripts/gcp/jobs/make_simons_job.py \
  --size 25m --leg stage1 --seed 11 --micro-batches 2 \
  --checkpoint-minutes 10 \
  --repo-sha "$(git rev-parse HEAD)" \
  --upstream-pin "$(cat third_party/simons/PINNED)" \
  --output /tmp/simons-stage1/job.sh
bash -n /tmp/simons-stage1/job.sh
```

Repeat with `stage2`, `stage3`, `stage4`, `stage4-long`, and `export`. For legs
3–5 add `--corpus-uri gs://BUCKET/CORPUS_PREFIX`. That prefix must contain
`univariate/<dataset>/` (Simons main `train-*.parquet` corpus) and
`covariates/<dataset>/` (Simons covariate-window format), matching the public
release and corpus job. The release handshake described below is required as well. Both directories are checked before any training,
including resumed legs and qualification of legs 3–5. A missing directory fails
with its full local path. Subfolders supported by upstream remain supported.
The job copies the prefix to `$RUN_DIR/corpus/` on local disk. No corpus is
needed for the first two legs or export. Long-pass usable lengths/observation
requirements remain as upstream specifies; missing usable rows are errors.

Upload each leg's generated `job.sh` under its own Cloud Storage object name,
for example `gs://BUCKET/jobs/simons-25m-seed11-stage1.sh`, then
`gs://BUCKET/jobs/simons-25m-seed11-stage2.sh`, and so on through export.
**Every leg's job URI must be distinct.** Do not overwrite a shared job object
between legs: the runner associates terminal markers with the job URI.

Use one state prefix per run for all five training legs and export, with the
same size and seed. Start each leg with `launch` and a new session label,
passing `--state-uri <run prefix>` and that leg's `--job-uri`. For example,
with `guard` configured as described in the separately maintained `docs/gcp.md`:

```bash
gcloud storage cp /tmp/simons-stage1/job.sh \
  gs://BUCKET/jobs/simons-25m-seed11-stage1.sh
guard launch simons-stage1-01 \
  --machine-type "$GCP_MACHINE" --zone "$GCP_ZONE" \
  --provisioning "$GCP_PROVISIONING" --hours 2 --boot-disk-size 100 \
  --image-family "$GCP_IMAGE_FAMILY" --image-project "$GCP_IMAGE_PROJECT" \
  --state-uri gs://BUCKET/runs/simons-25m-seed11/state \
  --job-uri gs://BUCKET/jobs/simons-25m-seed11-stage1.sh
# Continue this leg after preemption, once its old VM is absent:
guard resume simons-stage1-01 simons-stage1-02
```

After a leg completes, close its session before starting the next leg. For
stage2, upload that leg's generator output under its own object name and use
`launch simons-stage2-01` with the same machine options and
`--state-uri gs://BUCKET/runs/simons-25m-seed11/state`, changing `--job-uri` to
`gs://BUCKET/jobs/simons-25m-seed11-stage2.sh`. Repeat for stage3, stage4,
stage4-long and export, waiting for each leg to complete. Use `resume` only
to continue an interrupted leg; it retains that leg's job URI and state prefix
and takes a new session label, rather than selecting the next leg.

A terminal marker belongs to a job URI, not to the whole state prefix. Launching
a different job URI on the run prefix archives the previous job's terminal
markers and runs the new leg, preserving the training state for its predecessor
weights. Launch or resume of the same job URI is refused once it is `DONE`.

The guest runner's implementation is maintained separately. On the guest, the
actual job invocation is:

```bash
# RUN_DIR and STATE_URI are already exported by the guest runner.
# Its restore has already populated RUN_DIR/state, and periodic upload is active.
bash /path/to/job.sh
```

Use one GPU per job and one active writer per `STATE_URI`. All jobs in one
chain share the run prefix, size, seed, recipe and screen budget; use a separate prefix for another
chain/size/seed. Qualification runs use their own state prefix, separate from
the training run. The guest image needs NVIDIA drivers, Linux, Git,
Python >= 3.10 with pip and an already installed CUDA-matched torch >= 2.5,
plus authenticated `gcloud` for corpus copying. The 2026-11-13 A100 canary used
image family `pytorch-2-9-cu129-ubuntu-2204-nvidia-580`: the job runs as root,
system `python3` is 3.10, and torch `2.9.1+cu129` imports with CUDA available.
That image lacks `ensurepip`; the job does not require it or use `apt`.

Every unfinished start, including resume on the same or a replacement machine,
creates fresh scratch and an environment with `--system-site-packages`, reusing
the image's torch. It first tries `python3 -m venv`; if that fails, it installs
uv with system pip into scratch, then uses `uv venv --clear --seed
--system-site-packages` with that same system interpreter and Python downloads
disabled. uv supplies pip without `ensurepip` and clears the failed venv.
The environment's pip installs the patched Simons tree in editable mode and
missing upstream dependencies. An exact torch constraint, including the CUDA
version suffix, prevents resolution to a different torch build; installation
does not upgrade the image's satisfied packages or download another torch.

Before corpus copying or training, the environment interpreter must import
torch at the image's original path/version, satisfy the installed Simons torch
requirement, report CUDA available, and import both Simons and the patched job
module from this attempt's source tree. A failed check exits nonzero with a
one-line reason. `environment.txt` retains the resolved package list and adds
a runtime JSON comment with Python, torch, CUDA runtime and NVIDIA driver
versions and the torch/Simons paths; training copies it into the attempt record.
Jobs fetch upstream source, never model weights.
Hub and datasets offline modes prevent implicit model/dataset downloads. The
patch bundle is embedded, so the job does not need credentials for our repository.
The supplied repository SHA records provenance; the embedded bundle SHA records
the SHA-256 of the uncompressed, sorted-key JSON of the bundle members, including
changes not yet committed when generated. Gzip is transport only: the guest
decompresses the payload and verifies that digest before extracting members.
Identical sources have the same digest across Python/zlib versions and compression
levels. This changes the digest value from jobs generated before this change,
which hashed compressed bytes; old digest values are not accepted by the new
bundle verifier. This does not make old training state incompatible: the existing
resume checks tolerate changed bundle provenance, as described below.

The guest restores `<STATE_URI>/` to `$RUN_DIR/state/`, copies it back every ten
minutes, and once more on exit. The job itself neither creates machines nor
performs state uploads. A nonzero job status remains a failure/interruption;
only a complete leg returns zero. If invoked directly with restored completed
state, a training leg or export returns zero before source/dependency
installation or corpus copying; the guest runner refuses a job URI already
marked `DONE` before invoking it.

State is organized as follows:

```text
state/<leg>/identity.json
state/<leg>/checkpoint-slot-0.pt          # alternating snapshot, overwritten in place
state/<leg>/checkpoint-slot-1.pt          # alternating snapshot, overwritten in place
state/<leg>/checkpoint-final.pt           # single stable final for a completed leg
state/<leg>/resume-log.jsonl              # absolute step, UTC time, host per resume
state/<leg>/completed.json                # stable final name, step, bytes, SHA-256; published last
state/<leg>/attempts/run-*/                # recipe, environment and upstream logs
state/qualify/<name>.json
state/qualify/<name>-attempts.json
state/export/config.json
state/export/model.safetensors
state/export/completed.json
```

Training uses a new output directory automatically for every attempt. Checkpoints
are written directly into the durable leg directory. At a completed optimizer
update, either the upstream step cadence or the wall-clock cadence triggers a
save; wall-clock cadence defaults to ten minutes. Writes use a temporary file,
flush/fsync, atomic rename and directory fsync. Every snapshot alternates between
the same two names; the next save replaces the older slot, including after a
restart. Each file contains the absolute step, training seed, optimizer and
experiment identity, plus an embedded SHA-256 footer covering all serialized
content. Resume verifies each candidate's hash and checkpoint schema, then
selects the largest embedded step; file names and modification times never
establish recency. A stale or partially copied slot cannot outrank a valid newer
one. Temporary files are ignored on restore. A process killed before rename
leaves the previous committed snapshot usable. Embedded identity repairs an
identity file missing from an interrupted upload.

Completion publishes `checkpoint-final.pt` and records its name, step, byte count
and content hash in `completed.json`; local snapshot slots are then removed.
The guest's recursive copy does not delete remote objects: up to two old slots
can remain remotely alongside the one stable final (and fixed temporary names
if copied mid-write). Repeated saves add no new checkpoint object names. This
bounds new chain checkpoint storage without relying on bucket deletion; it does
not clean up old step-numbered objects from runs made with earlier job bundles.
A training completion marker is accepted only when the referenced file's size
and content hash validate; export markers check their artifact sizes.
A marker copied ahead of a missing checkpoint cannot skip the leg. A final
checkpoint published before its completion marker is recognized on the
next run and the marker is repaired. Existing checkpoints with no readable
committed snapshot cause an error, rather than silently starting over.

Resume preserves the original learning-rate/re-warm origin, absolute context,
synthetic-share and role/effect phase clocks. Every rank derives fresh global
Python, NumPy, Torch CPU/CUDA and private/worker streams from `(run seed, rank,
absolute restart step)`; synthetic pools and mechanistic banks are rebuilt.
The real sampler remains upstream's absolute-step/rank sampler. Two restarts
with the same seed/step/configuration/worker count produce the same next batches.
They do **not** reproduce the uninterrupted run's random draws. Numerical
bit identity across GPU types, library versions or kernels is not promised.
Curriculum boundaries and remaining phase lengths are preserved. No claim of
convergence equivalence is implied by the CPU tests.

Only committed state that reached Cloud Storage survives machine loss. Save and
upload clocks are independent, so rollback can exceed one save interval (roughly
two intervals plus update/save/upload time with the default clocks). Cadence is
checked at update boundaries; an update taking longer than ten minutes delays
the save. The guest's final copy helps on a normal timeout but cannot be assumed
on a hard Spot interruption. There is no concurrent-writer lock in the job.

Generate qualification before committing to a long run:

```bash
python scripts/gcp/jobs/make_simons_job.py \
  --size 25m --leg qualify --qualify-leg stage1 --name a100-stage1 \
  --seed 11 --micro-batches 2 --max-micro-batches 128 --warmup 20 \
  --repo-sha "$(git rev-parse HEAD)" \
  --upstream-pin "$(cat third_party/simons/PINNED)" \
  --output /tmp/simons-qualify/job.sh
```

Qualification uses the named leg's configuration and starting weights; it needs
that leg's predecessor, and corpus for legs 3–5. Before qualifying a later leg,
copy the completed predecessor state into the qualification run's own state
prefix. Upload the qualification job under its own object name and launch it
with a new session label, that `--job-uri`, and the qualification `--state-uri`.
It never advances or completes that training leg. It warms up for 20 updates by
default, then measures exactly 500 updates with synchronized CUDA boundaries.
CPU execution fails clearly.
Every OOM attempt runs in a separate process, doubles only `micro_batches`, and
is recorded; the count stops at `--max-micro-batches`. An exhausted limit returns
nonzero with an explicit report. Other failures do not trigger OOM retries.

Qualification reports always carry `qualification_only: true`. Their checkpoints
are explicitly rejected as predecessors, including when renamed into a leg directory.

Reports record peak Torch GPU allocation and reservation, 50 ms sampled process-tree
host RSS, updates
per second, supervised scalar values per second (main observed targets plus
unweighted auxiliary observed targets, excluding the quantile dimension), and
loader/generator wait share. Wait includes main loading/generation/transfer and
auxiliary queue waits; it is not a separate CPU-utilization measurement. Timings
exclude logging/checkpoint I/O and include synchronization overhead. The probe
defaults to the first 500 post-warm-up updates of the leg. Use `--qualify-step` to
probe a later absolute phase without changing its LR/role/context origin. For
example, `--qualify-leg stage3 --qualify-step 41000 --name a100-stage3-far`
measures the 32,768-step context, whereas the default stage3 start measures the
7,680-step context and synthetic-share ramp. Probe late role/effect phases of
stage2 separately as well. A later probe uses the normal predecessor weights and
a fresh optimizer, not learned weights from that future step; it is a memory and
speed measurement, not a convergence experiment. The selected phase must leave
room for the warm-up and all 500 updates. One probe does not establish worst-case
fit over every packed shape/curriculum phase.

Upstream micro-batching splits packed **main** trunk forward/backward passes
with the same global observed-value loss normalization. It does not split raw
loading, synthetic pools/banks, preprocessing/packing or auxiliary covariate
forwards. If those peaks exceed a 40 GB card even at the maximum count, retain
the report and choose a larger card; reducing batch/context/pools would change
the experiment. The live 25M full stage1 qualification fit two micro-batches on an A100
40 GB; later-leg and 60M fit remain unmeasured.

Export uses upstream's uniform arithmetic averaging of `stage4` and
`stage4-long` final weights, then upstream inference export (including its
export context setting). It publishes `config.json` and `model.safetensors`
under `state/export/`, with a completion marker last. Interrupted export is
rebuilt. Its soup checkpoint is intermediate local scratch, not a third chain
member or an additional permanent weight artifact.

Measured locally: shell tests with fake image commands/packages for venv failure
and uv fallback, fresh environments on repeated starts, and rejection of missing
CUDA, old torch or an unpatched Simons import. These tests do not measure actual
image dependency installation or GPU operation; the real image must confirm
environment creation, torch reuse without a large download, and GPU training.
CPU tests cover role schedules across interruption, deterministic
restarts, partial-write/hash rejection, bounded remote snapshot names, embedded-step selection,
cadence/retention, corpus layout checks on fresh/resumed legs, predecessor wiring, wrong
pin/source rejection, width-only configuration differences, main micro-batch
loss/gradient parity, and upstream export averaging. Generated shell syntax is
checked for all seven job types at both sizes. These tests use tiny models and
synthetic inputs; no real data, weights or Google calls are required.

Unmeasured: GPU/host fit at recipe scale, throughput, cold bank creation cost,
CUDA determinism, native multiprocess loader startup, long-history/auxiliary peaks,
Google guest synchronization,
new-bundle Spot/timeout recovery, distributed execution and convergence. The live full 25M stage1 ran on A100 40 GB at 1.434 updates/s with two
micro-batches and resumed correctly after preemption, preserving learning-rate
position and loss level. Those measurements predate this patch; they do not
establish new counter overhead, screen/60M throughput or later-leg fit. The
corpus release handshake must still pass before corpus-reading training.

For integration tests, set `SIMONS_UPSTREAM` to a clean local clone at `PINNED`
and run `python -m pytest tests/test_gcp_jobs -q`. Tests copy only source,
configs and Git metadata into pytest scratch and apply patches there; they do
not modify the supplied clone or install Simons globally. Without that variable
(or without source at the specified path), source-dependent tests skip with a
clear reason and generator/shell tests still run in ordinary repository CI.


## Full and screen recipes; 60M

`--recipe full` is the default. `--screen-updates` is accepted only with
`--recipe screen`; its default is 25,000. Append these flags to every job in a
12M or 25M screen chain, including qualification and export:

```bash
--recipe screen --screen-updates 25000
```

| Screen leg | Updates | Absolute schedule coordinates |
| --- | ---: | --- |
| stage1 | 13,518 | 0–13,518 |
| stage2 | 2,496 | 0–2,496 |
| stage3 | 6,490 | 5,489–11,979; context switches at 8,310 |
| stage4 | 832 | 11,147–11,979; from stage3 |
| stage4-long | 1,664 | 10,315–11,979; independently from stage3 |

The total is exactly N. Integer clocks use nearest-integer rounding of
N/120,208 times the registered full 25M clock. Round the four other leg
lengths first and assign the rounding residual to stage3, then set its origin
from the rounded schedule end and its length. This preserves fractions within
one update for other N. Budgets below 100 are rejected. Tiny budgets can be too
short for the mandatory 500-update qualification probe.
At N=25,000, stage1 warm-up/decay is 270/2,028; stage2 re-warm/decay is
125/374; mixed-leg re-warm is 62 and synthetic ramp is 417. Context phase 0
ends at 5,489 and phase 1 ends at 8,310. Role/effect fractions 0.27/0.53 and
WSD fractions 0.05/0.5418 remain dimensionless. Sampling pools, bank refresh,
workers, batch shapes and loss weights are unchanged. The two soup branches
retain their separate optimizers and common stage3 parent.

`--size 60m` selects upstream `simons-60m-stage*.yaml` for full training,
qualification and export. This is a different native architecture and recipe:
width 704, eight layers, eleven heads, read after block 3; 25M uses 384,
twelve layers, six heads, read after block 5. Mixed-stage checkpoint blocks
are eight rather than twelve. The 60M auxiliary token budget/max batch/workers
is 13,696/428/9 versus 4,096/128/3 (also inherited by the long pass's token
budget). Synthetic stages include calendar with weight 0.35; the other family
weights, weight decay and some LR settings differ. Native stage1 has 55,000
updates, warm-up 1,000 and linear decay 6,600 to 0.15; stage2 has 12,000,
decay 1,440 and re-warm from 0.15. Mixed schedules end at 17,240, start
at 7,899 (stage3), 16,040 (stage4), and 14,840 (long), with ramp 600 and
90-update re-warm in the final branches. Thus the full 60M chain executes
79,941 updates. Job seed/path overrides remain the same as for 25M.

`--size 60m --recipe screen` is refused by the generator and guest program:
a 60M short schedule needs a registered protocol. The registered short schedule
uses the 120,208-update 25M chain; no short protocol is registered for the
native 79,941-update 60M chain. Use `--recipe full` for 60M.

Full synthetic-leg identity retains its existing schema. Legacy e5959bb
checkpoints are full-recipe checkpoints and remain valid predecessors and
resume inputs. Changing the embedded patch bundle/repository provenance alone
does not invalidate an otherwise identical resume; the original identity's
provenance is retained. Each attempt records its actual current repo/bundle
provenance separately in `implementation.json`. Screen identities add `recipe_name` and
`screen_updates`. Mixing recipes, budgets, sizes, seeds or upstream pins fails.
Use a separate state prefix for each recipe and N. Full 25M/12M generated shell
is covered by e5959bb golden hashes, with these explicit byte exceptions:
embedded bundle/digest (new patch and installer), identity checks on the fast
completion path, removal of stale local audit receipts before corpus sync,
HOME/kernel-cache defaults, and bundle verification over uncompressed bytes.
Full training recipe settings are unchanged.
Already-generated jobs and the two existing patches are not edited.

## Corpus audit gate

Every new or resumed corpus-reading training leg and qualification of legs
3–5 requires `attempt-status.json` and `audit.json` at the corpus prefix.
Status must have a nonempty attempt ID, `complete: true`, and integer
`exit_code: 0`. Audit must have the same attempt ID, `valid: true`, integer
exit 0, a nonempty folder report with allowed folders, and `clean` verdicts
for every allowed folder. Missing, stale, malformed or failed receipts abort
before training. Layout validation also runs. Corpus sync removes the two local
receipts first so a missing remote object cannot leave a stale accepted receipt.
The exact audit bytes' SHA-256 and accepted attempt ID enter `identity.json`,
checkpoint identity and `completed.json`. Changing that receipt between mixed
legs or on resume is refused. Completed, validated jobs remain no-ops without
reading a corpus. Legacy mixed state without a receipt acquires one on resume.
Export accepts a legacy full-recipe member alongside an audited member, or two
legacy full-recipe members. Two recorded receipts must match. The export's
`member_corpus_releases` lists each member's receipt (or `null`) in `members`
order; `corpus_release` records the available receipt, or `null` if neither
member recorded one.

For **qualification only**, the explicitly named
`--unsafe-qualification-allow-unaudited-corpus` bypasses the release check for
speed/memory measurements on unaudited data. It does not bypass corpus layout
or predecessor validation. Qualification reports and checkpoint identities
carry `qualification_only: true`; the receipt is marked
`UNSAFE_UNAUDITED_CORPUS`. All qualification checkpoints, with or without this
flag, are rejected as predecessors. Production training/export reject the
flag. Qualification output must use a separate state prefix.

## Scalar exposure and replay records

`exposure` in each training JSONL loss record, training checkpoint,
qualification result and leg completion uses schema `simons-exposure-1`.
Qualification exposure totals include warm-up and measured updates; the existing
500-update timing fields exclude warm-up. Counts accumulate **within that leg**
and restore from its checkpoint; warming
from a predecessor starts new leg counters. Quantile copies, activation
recomputation and micro-batch partitioning do not add exposure.

- `E_real`, `E_synthetic`: finite target scalar positions admitted as inputs or
  eligible labels, unioned once per consumed post-augmentation window. A new
  sampled occurrence counts again. Target and covariate roles remain separate.
- `S_real`, `S_synthetic`: finite eligible scalar labels after role, visibility
  and packing masks, independent of nine quantiles or numerical loss weight.
- Positive-weight auxiliary passes add context/label unions and supervised
  positions, including history labels only when history weight is positive.
  The auxiliary stream named `synthetic` derives its targets from corpus bases;
  it and sibling/real auxiliary targets therefore have **real ancestry**.
  Procedural main-batch effect rows remain synthetic.
- `covariate_values` counts finite consumed covariate scalars separately;
  `channel_windows` counts consumed nonempty target channel windows.
- `updates_attempted` and `optimizer_updates` cover retained trainer work.
  Checkpoints preserve elapsed training seconds and active GPU hours (zero for
  CPU tests); logs record seconds, active GPU hours, the config SHA-256, identity,
  seed, exact recipe and bundle/corpus hashes. These are training clocks,
  excluding checkpoint pauses, not accumulated VM billing or all-attempt time.

`replay.known_updates` counts updates re-executed below the highest update
visible in retained training logs after rollback. Its E/S, covariate and channel
window fields count the corresponding re-executed values; replay counts stay separate
from checkpoint-lineage E/S totals. `lost_telemetry: "unknown"` explicitly
marks work missing from restored files. An upload can lose logs as well as
weights, so the known replay count is a lower bound, never proof of zero waste.
Legacy checkpoints start counters with an explicit missing-history annotation.

Exact at these hooks: consumed target-window E, masked S, real/procedural
ancestry, auxiliary counts, covariate values and micro-batch invariance.
**Missing, listed in every counter snapshot:** original source rows drawn before
packing/replacement, per-source/detailed-role breakdown and original ancestry multiplicity
inside real mixup/stitching. These require sampler instrumentation; consumed
channel windows are not a substitute. Lost/failed work absent from retained
telemetry and all-attempt GPU/VM hours remain unknown. Sum the three shared
stages once and each final branch once when reporting soup ancestry; export
weights do not imply a new exposure pass.

## Score an exported checkpoint on fev

`scripts/gcp/jobs/make_score_job.py` generates one self-contained executable job
for the same guest contract. Generation is offline. Choose a **published repository
commit containing the Simons yardstick adapter**; the chain branch predates it.
The embedded guest SHA-256 covers its uncompressed Python source bytes. The guest
decompresses and verifies those bytes before writing the scoring program; gzip
settings and Python/zlib versions do not affect its identity digest.
The digest differs from jobs generated before this change, which hashed compressed
bytes. The input identity check refuses such scoring state under the same name,
except for the existing compatibility path accepting the `139f3f7` guest digest.
No additional old-digest compatibility is introduced; use a new name/state prefix
for refused state.
For example, with operator-supplied immutable revisions and input hashes:

```bash
python scripts/gcp/jobs/make_score_job.py \
  --name simons25m-stage4 \
  --repo-sha "$SCORING_REPO_SHA" \
  --upstream-pin "$(cat third_party/simons/PINNED)" \
  --checkpoint-uri gs://BUCKET/chain-state/export \
  --tasks-uri gs://BUCKET/scoring/cohort.yaml --tasks-sha256 "$TASKS_SHA256" \
  --names-uri gs://BUCKET/scoring/names.txt --names-sha256 "$NAMES_SHA256" \
  --processing both --output /tmp/simons-score/job.sh
```

Upload and launch through the guard/guest workflow, with a dedicated scoring
state prefix. No launch, cloud access or model download happens in the generator.
Replace `--checkpoint-uri` with `--hub-weights abel-lab/REPO --hub-revision SHA`
for an anonymous Hub download; the full 40-character revision is mandatory.
Only `config.json` and `model.safetensors` are copied. Both files are SHA-256
hashed before scoring, and changed checkpoint contents are refused when resume
downloads them again. A completed marker's early exit checks only recorded state;
it does not contact the checkpoint source or detect changes there. Use a new score
name/state prefix to score a changed checkpoint. The task YAML and optional
one-name-per-line selection require SHA-256 hashes. Omit both names switches to
select the complete YAML. `--max-context N` must be a positive integer.
`--covariates {all,none}` defaults to `all`; for Simons, `none` withholds both
the history and future of known covariates, rather than applying a future-only mask.
The covariates choice is part of the input identity, so changing it requires a
new score name/state prefix.

The job starts with the training environment's system-site-packages venv and uv
fallback for missing ensurepip. The full yardstick requirements cannot run on
the image's Python 3.10: SciPy 1.16.3 requires Python >=3.11. The job explicitly
provisions Python 3.12 using uv and installs stage 1's CUDA 12.8 torch 2.8.0 wheel,
then the fetched repository's yardstick requirements and bench extra, plus the
unmodified Simons source at `PINNED` as described in `yardsticks.md`. The image's
Python 3.10 torch 2.9.1+cu129 cannot be shared across that interpreter ABI. There
is no fev downgrade: fev 0.10.0, datasets 3.6.0 and huggingface-hub 0.36.0 are
asserted, together with CUDA availability, torch 2.8.0+cu128 and the Simons VCS pin.
`environment.txt` records pip freeze, both image and scoring runtimes, and the
NVIDIA driver. Python patch versions and unpinned transitive dependencies can
differ from the stage-1 freeze. A real image must verify successful installation.

Results live under `$RUN_DIR/state/score/<name>/raw/` and `published/`, with labels
`<name>-raw` and `<name>-published`. The default is both, raw first; a failure in
raw still allows published to run. Each CLI invocation uses `--resume --keep-going`.
The guest owns restore, periodic/final state copies, timeouts and terminal markers.
Dataset, Hub and uv caches stay under `$RUN_DIR/score-cache/`, outside `state`.
Checkpoint downloads use a deterministic disposable directory under `/tmp/`, so
the adapter's absolute checkpoint path remains stable across changed guest labels.
No required resume data lives there: both files are obtained and verified again.

The job records its input identity before installation. Different inputs with
the same name fail without overwriting prior results. Each mode records requested,
completed and failed task names (first error line), wall seconds and provenance.
`completed.json` is published atomically after both modes finish; failed tasks or
a nonzero scorer status give a nonzero job exit. Failed runs remain resumable.
A successful marker permits a dependency-free exit only after hashing the recorded
state artifacts and checking task/window coverage. Missing or damaged input
identity can be repaired from the marker before this check. Incomplete evidence
requires the normal resume path; damaged checkpoint hashes are repaired only when
freshly downloaded hashes match the marker's independent copy. Without independent
evidence, damaged identity or checkpoint hashes cause refusal to overwrite state.
Changed inputs, or downloaded checkpoint contents that differ from trusted
recorded hashes, cause refusal. State written by the `139f3f7` guest retains its
input identity and checkpoint path through these recovery fixes only when resumed
with the default `--covariates all`. This is evidence
consistency checking, not a fresh recomputation of fev scores. There is no
concurrent-writer lock: use one guest per score name/state prefix. GPU execution,
numerical parity, memory fit
and real preemption/copy recovery require a GPU canary; offline stubs establish
control flow, input validation and local recovery behavior only.
