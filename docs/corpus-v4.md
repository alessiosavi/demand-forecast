# Corpus v4.0: public Simons native release

**Research-only use.** Original dataset licences apply independently of the
Simons wrapper licence. The allowlist is a contamination policy, not commercial
or legal clearance. Duplicate representations remain allowed as requested; the
97 folders are not 97 independent sources.

The frozen source is `abel-lab/Simons-Pretrain` at
`5f17ab8cfa3fa00a8e821599b631127ba250c342`. The reviewed metadata allows 97
folders and quarantines 24 unresolved energy, macroeconomic and air-quality
lineages. `configs/corpus-v4.yaml` retains each source's licence, frequency,
structure, lineage, reason and uncertainty. Exact frequencies come from native
rows, not the draft's coarse labels.

## Two contamination rules

**Rule F:** any part of a benchmark dataset at the same frequency used in
training is overlap. Benchmark training histories and official test windows both
count. Different resolutions alone do not establish F overlap.

**Rule L (default):** exclude the entire observational lineage, including other
resolutions, mirrors, subsets and derivatives. Content matching can establish
shared observations; failure to match does not establish independent lineage.
The frozen 100 task definitions and their full data at the existing benchmark
revision `d3e7e720f4158eab03ebcb021f1cce779cd92acc` are used, with no evaluation
cutoff or context truncation. Every numeric sequence column is compared,
including observed targets and covariates on both sides.

## Freeze, fetch and measure

Run these modules from an installed repository checkout. No dataset payloads
should be fetched on the development Mac. The following fetch/audit commands
are for the cloud guest:

```bash
python -m scripts.corpus_v4.freeze_manifest SOURCE_DRAFT.yaml configs/corpus-v4.yaml
python -m scripts.corpus_v4.fetch_native "$RUN_DIR/state/corpus" --state "$RUN_DIR/state"
python -m scripts.corpus_v4.audit_fev_overlap "$RUN_DIR/state/corpus" \
  "$RUN_DIR/state/audit" --state "$RUN_DIR/state"
```

The generator is deterministic, sorted by full folder name, and does not use
tier/duplicate annotations to override either contamination rule. Count/rule
regressions fail the tests. Policy changes require review of the frozen file.

Hub access is behind `Hub`; tests substitute `FakeHub`. Production uses anonymous
`huggingface_hub` calls with an immutable revision for both inventory and files.
The resolved revision must agree. Native files absent from the manifest,
unlisted folders, unexpected nested paths and missing folders are refused
before transfer. Every shard must match the Hub's size and LFS SHA256 (or Git
blob SHA1 for a non-LFS file). A digest mismatch is an error, not an implicit
repair or a silently accepted download. The per-folder receipt records these
identities. Existing completed folders are rehashed before skip; measurements
are regenerated to avoid trusting modified stats. Policy/revision mismatches in
an existing corpus require a new root.

Allowed data lands only in `corpus/univariate/<dataset>` and
`corpus/covariates/<dataset>`. Quarantine lands only in
`corpus/quarantine/{univariate,covariates}/<dataset>`. Pass the two explicit
allowed roots to the trainer. Download staging and partial files live under
`$RUN_DIR/state/downloads`, outside either reader root, and remain there after
interruption. A folder is atomically renamed into place only after verification
and measurement. Corpus and download state must be on the same filesystem.
Never pass `corpus/quarantine` to a trainer. Quarantine status never changes
because a restart or an audit finishes.

Schema validation enforces nonempty string `item_id/source/freq`, numeric
`target:[V,T]` for the main tree, and `target:[T]` plus aligned numeric
`covariates:[C,T]` for the covariate tree. Ragged channels, infinities, missing
columns and nonnumeric sequences fail. NaN/null observations remain missing.
`train-*.parquet` and `validation-*.parquet` are the only accepted shard names.
Original segment IDs and channel ordering are preserved; ingestion does not
stitch, resample, flatten, or truncate native files.

Each folder has `stats.json`: rows and split counts, target variates, exact
length histogram/min/max/mean, count of rows at least 16,384 steps, target and
covariate observed counts, target observed share, frequency strings,
`domain/subdomain` presence counts and covariate channel-count histogram.
The first training shard is representative only if it covers every observed
training support value for variates/lengths/frequencies and each marginal total
variation distance from all training shards is at most 0.10. This is a
conservative descriptive check; a false result is reported, never repaired by
reordering shards. The trainer's long covariate pass also requires usable
75%-observed windows; a row-length count alone does not certify that pass.

## Matching and evidence

The new audit reuses `audit_match.Index`, all four exact/affine/run hash families,
`scan_group`, candidate verification, affine fits, `nondegenerate` and `calendar_pattern`, plus
`audit_info`'s description-length scores. It reuses all 40 old planted controls
and their ten rule checks. Existing audit callers and thresholds are unchanged.

Candidate windows have eight points; candidate groups need three seed hits or
24 summed seed spans. Final matches need at least 24 jointly finite points.
The existing copy test requires at least 99% exact points (`atol=0.005`,
`rtol=1e-6`) or an affine maximum residual no larger than
`max(0.01, 0.005 * benchmark range)`. Both series must pass the existing
64-float32-ulp precision gate, four off-modal points and seven information
degrees of freedom. Short exact periodic motifs (periods 1 through 32) are
rejected. Information bits must exceed the run-12 search-opportunity bound plus
20 bits; strict interpolation ramps cannot declare contamination. Opportunities
are conservatively bounded from uncompressed corpus metadata, benchmark points,
4,096 aggregation phases, sum/mean and seed strides. The actual threshold is
recorded in `audit.json`. No threshold is lowered for a control.

For missing data, strides 1–4 seed through regular holes. Original positional
arrays, including their masks, are always used for final verification. This
extends discovery without joining observations across missing timestamps.
Float32 copies, arbitrary affine unit changes and cropped segments can match.
Evidence includes dataset/item/column, corpus shard/row/channel, source,
`<base>@<offset>`, local chunk offsets, fitted offset, half-open bounds,
finite-point count, fit and information diagnostics. Original positions within
a row equal chunk offset plus the local evidence bound for identity searches;
add the ID's segment offset for parent-series positions. For fixed aggregation,
map positions back using `phase + ratio * position` on the aggregated side.
Calendar evidence uses benchmark timestamps and explicitly discarded edge bins.

All benchmark domains are currently searched conservatively because no reviewed
crosswalk proves any pair impossible. Thus every same-frequency pair is searched,
and every cross-frequency pair is considered, including all quarantine pairs.
Fixed-resolution aggregation searches **all** bin phases in both sum and mean,
in the fine-to-coarse direction on either side. Missing inputs make a whole bin
missing, and partial fixed edge bins are discarded. Calendar aggregation of a
benchmark parent preserves the original fiscal/calendar aliases, uses its
timestamps, rejects irregular grids and discards possibly partial first/last
bins. A multi-period calendar offset whose phase cannot be recovered is
inconclusive. Constant, periodic or strict-ramp fine parents cannot establish
identifying ancestry merely because aggregation introduces calendar bin lengths.
Candidate calendar aggregates are also checked against their aligned bin-size
arrays with the old calendar-pattern rejection, including cropped segments.

**Unresolved coverage is reported, never silently clean.** Corpus-to-calendar
aggregation needs an absolute origin missing from native rows. Such pairs are
listed as skipped/inconclusive. Nonintegral ratios, more than 4,096 phases,
unsupported offsets and resource caps also make an allowed folder fail. Searching
all domains means this limitation may prevent a clean release for many folders;
a reviewed domain crosswalk or recovered timestamps is needed to resolve it.
Neither phase guessing nor treating months as thirty days is acceptable.

These heuristics cannot prove absence of overlap, identify ancestry from neutral
IDs, reconstruct missing history, find every nonlinear derivative, guarantee a
seed under arbitrary dense missingness, or detect identifying copies shorter than
24 observations. Constant/zero or short repetitive observations cannot identify
a source. `clean` means the declared search completed without an identifying
match; it is not proof that all observations have independent provenance.

## Results, restart and resource bounds

`audit.json` contains per-folder `clean`, `rule_f_match`, `rule_l_match`, or
`inconclusive`, evidence, coverage counts, and searched/skipped dataset-frequency
pairs with comparison counts and aggregation methods/phases. A match takes
verdict precedence; unresolved reasons remain attached. `valid` means planted
controls and input inventory/provenance checks passed, not that every folder is clean. Exit 2 means a failed control,
an allowed match or an inconclusive allowed folder. Quarantined matches and
inconclusive results remain reported without changing their manifest status.
A clean quarantined folder is only a candidate for release by PR.

Every run executes old controls and benchmark-derived as-is, scaled, cropped and
hourly-parent/aggregated controls. The unrelated seasonal control must remain
negative. If no identifying benchmark series can support the plants, the run is
invalid. Controls use the production matcher and the same information threshold.

State is under `$RUN_DIR/state` (override with `--state`). SQLite spools benchmark
histories to disk. A folder result is atomically checkpointed with the method,
manifest, benchmark content/revision, search budget and its current Parquet
hashes. On restart benchmark identity is rebuilt, all corpus hashes are reread,
completed identical folders resume, and controls run again. Incompletely
restored published folders are moved back into isolated staging after surviving
files are rehashed; missing files are fetched again. Wrong existing hashes
remain errors. No within-folder
progress claim is made. Changing code requires a method-version bump when its
search semantics change; changing inputs automatically invalidates cached results.

Parquet rows are streamed one at a time; numeric rows above two million values
or row groups above 512 MiB / 32 million leaf values are refused before large
row-group decoding. Series chunks use at most 65,536 points with a 512-point
overlap; combined benchmark indices contain at most 262,144 points. Total hash
postings are counted before the old matcher expands them; more than one million
postings for a query is an explicit unresolved resource limit. The old explicit expansion and
candidate caps remain; dropped candidate/offset coverage produces inconclusive.
There is a 10,000-evidence-record guard per folder. Benchmark contents and audit
state require disk in addition to the native corpus. Processing bounded benchmark-index batches
trades repeated corpus reads for bounded memory;
cloud wall time and peak RSS remain unmeasured.

## Cloud guest job and release

```bash
python -m scripts.corpus_v4.make_corpus_job FULL_REPO_SHA \
  gs://YOUR_BUCKET/corpus-v4 job.sh
bash -n job.sh
```

The generated job installs the repository at the full SHA into an editable venv
with the existing `bench` extra. The package requires Python >=3.10, so the
`pytorch-2-9-cu129-ubuntu-2204-nvidia-580` image's Python 3.10 is sufficient.
The venv inherits system site packages. If the image lacks `ensurepip`, the job
bootstraps `uv` with the image's pip into scratch space and uses it to create and
seed the venv without downloading another interpreter. Both paths clear a
half-built or previous venv on restart. A constraint derived from the image's
own `torch.__version__` preserves its CUDA-matched torch, including the local
version suffix; the preflight also checks torch's version and file path.

Before fetching data, the job imports the fetch/audit modules and their lazy
dependencies, verifies editable-install metadata and module paths against the
checkout, and rechecks its HEAD against `REPO_SHA`. It records `pip freeze` in
`RUN_DIR/state/environment.txt`. NumPy/pandas/PyYAML are base dependencies;
PyArrow/datasets are in `bench`, and datasets already depends on huggingface_hub.
The guest provides Python with pip and torch, Git, gcloud, application-default
Google credentials, `RUN_DIR` and `STATE_URI`. No credentials are embedded.
The SHA must contain this implementation and frozen manifest.

If `HOME` is empty or unset, the job sets it to `RUN_DIR/home` for startup-script
execution as root. Default Hugging Face caches are under `RUN_DIR/hf-cache`,
outside the periodically uploaded state tree. The fetcher's explicit staging
cache and the audit's explicit benchmark cache retain their existing resumable
locations under state; these explicit paths override the default cache settings.

The host restores `RUN_DIR/state` before invocation. The job syncs it every ten
minutes and at exit; the host's final copy also covers guest preemption. Each attempt first invalidates the old final audit and publishes
`attempt-status.json` with a unique attempt ID and repository SHA. The final
audit carries the same attempt ID. Consumers must require matching IDs, a
complete status with exit 0, a valid audit and clean allowed folders; an old
`audit.json` at an object-store prefix is insufficient. Its exit
trap uploads the entire corpus (including per-folder stats and receipts) and
`audit.json`, even after an audit failure. A final upload failure exits nonzero.
Do not train from a prefix with a failing/missing audit or incomplete transfer.
No cloud deployment or paid operation is performed by the generator.

Release a quarantined folder only through a reviewed manifest-change PR citing
the exact audit output, corpus/benchmark revisions, complete pair coverage and
lineage evidence. Resolve skipped searches or inconclusive alignment explicitly.
A content-clean result alone cannot settle an unresolved metadata lineage.
The generator must be updated through the reviewed source draft so regeneration
preserves that decision. Never move quarantine bytes into the allowed tree by
hand. The cloud run still must measure native schema validity, actual sizes and
counts, first-shard representativeness, long-row suitability, real overlaps,
control success at real search scale, runtime, memory and transfer/restart safety.
