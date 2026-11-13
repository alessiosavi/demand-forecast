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
python -m scripts.corpus_v4.fetch_native "$RUN_DIR/state/corpus" --state "$RUN_DIR/state" \
  --memory-budget-gib 32
python -m scripts.corpus_v4.audit_fev_overlap "$RUN_DIR/state/corpus" \
  "$RUN_DIR/state/audit" --state "$RUN_DIR/state" --memory-budget-gib 32
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
identities and the statistics digest. Existing completed folders are rehashed
before skip; statistics are reused only when their digest, measurement version
and measured memory requirements agree. Modified or legacy statistics are regenerated. Policy/revision mismatches in
an existing corpus require a new root.

The pinned tree contains 150 Parquet shards in exactly 121 dataset folders
(62 univariate, 59 covariate), matching all 97 allowed and 24 quarantined
manifest entries. Each shard sits directly inside its dataset folder; 115
folders have one shard and six have multiple shards (up to 13). Inventory keeps
all of them. Under the two data roots, only `train-*.parquet`,
`validation-*.parquet`, `LICENSE.md` and `articles.csv` are accepted, at exactly
`<root>/<dataset>/<file>`. Every folder must contain a training shard; side
files cannot satisfy that requirement. Invalid paths are reported together
before transfer. Root documentation, assets and `statistics/` are outside this
folder inventory.

Ten `LICENSE.md` files retain per-source licence records. Three `articles.csv`
files in `nyc_subway_events`, `wikipedia_sports_fixtures` and
`wikipedia_tv_episodes` retain Wikipedia attribution: the release README points
to NOTICE, which describes the article/series mappings and dated revisions.
These side files are fetched, verified, recorded in the same folder receipt,
restored after interruptions and kept beside the data, including in quarantine.
All 13 are non-LFS Git blobs in this revision, verified with the Git blob SHA1
(including the blob header); the 150 shards use LFS SHA256. Side files are never
parsed or included in row/value statistics or the content audit.

Allowed data lands only in `corpus/univariate/<dataset>` and
`corpus/covariates/<dataset>`. Quarantine lands only in
`corpus/quarantine/{univariate,covariates}/<dataset>`. Pass the two explicit
allowed roots to the trainer. Download staging and partial files live under
`$RUN_DIR/state/downloads`, outside either reader root, and remain there after
interruption. Hub downloads use `local_dir` regular files and transfer metadata,
then rename the file into staging; no Hub blob cache or symlinks are created in
state. Obsolete `.hub-cache` directories from earlier failed downloads are removed
before transfer. A folder is atomically renamed into place only after verification
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

The metadata-only regression fixture
`tests/test_corpus_v4/fixtures/simons-5f17ab8-tree.json` contains all 172 release
paths, sizes and hashes, with no dataset payloads. Its paths come from the stored
`simons_api.json`; sizes/Git oids/LFS oids come from `simons_tree.json` alongside
it (the API capture has paths only). The stored `simons_manifest.json` is the
release's `statistics/manifest.json`: its `files[].folder` set matches the
frozen policy exactly. Dataset display names and `source` aliases can differ
from folder names and are not used to infer paths. The breakdown CSVs describe
public, private and synthetic training populations, so their totals are not
acceptance counts for this public subset. They do not specify per-row lengths,
Arrow schemas or exact frequency strings; these remain runtime checks. The
README's segment IDs and optional metadata agree with the native validator;
empty covariate lists are supported, as required by the NOTICE's seven clubs
without schedule pages. Actual longest rows still require the cloud run; no row
is truncated to fit a guard.

Trainer compatibility was checked against upstream Simons commit
`8e338e915d26c8960713889ef12ab1cf5c08d591`:
`src/simons/training/data.py::corpus_files` selects only
`<split>-*.parquet` in the supplied root or one dataset directory below it and
rejects deeper Parquet nesting. The covariate loader in
`src/simons/training/covariates/windows.py` uses the same selector. Licence,
attribution, receipt and statistics files therefore cannot enter either loader.
Supply the explicit `univariate/` and `covariates/` roots described above.

Each folder has `stats.json`: rows and split counts, target variates, exact
length histogram/min/max/mean, count of rows at least 16,384 steps, target and
covariate observed counts, target observed share, frequency strings,
`domain/subdomain` presence counts and covariate channel-count histogram.
It also records `values` (targets plus covariates), `longest_row` (steps),
`largest_row_bytes`, and maxima of row-group bytes, leaf values and rows.
`measurement` records the strategy/version, wall seconds for the validation and
statistics pass, the worker's peak RSS in bytes, configured budget and hard row
byte limit. Each measurement runs in a fresh subprocess: its RSS is specific to
that folder, not a previous folder's process high-water mark. Download/hash time
and process startup are outside `measurement.wall_seconds`.
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

Each input/method/opportunity identity executes old controls and benchmark-derived
as-is, scaled, cropped and hourly-parent/aggregated controls once. Completed
controls are atomically checkpointed and reused across pool restarts. The unrelated seasonal control must remain
negative. If no identifying benchmark series can support the plants, the run is
invalid. Controls use the production matcher and the same information threshold.

State is under `$RUN_DIR/state` (override with `--state`). SQLite stores benchmark
metadata and signed 64-bit offsets into flat numeric/timestamp spool files.
Histories are written in 65,536-point blocks and mapped for iteration; there is
no per-history SQLite BLOB or 1 GB BLOB limit. SQLite's page cache is 8 MiB and
temporary tables stay on disk. A folder result is atomically checkpointed with the method,
manifest, benchmark content/revision, search budget and its current Parquet
hashes. On restart benchmark identity is rebuilt, all corpus hashes are reread,
completed identical folders resume, and completed controls with the same
benchmark/method/opportunity identity resume. A changed opportunity bound reruns
controls at the new threshold. Runtime failures are
retried, including quarantined folders. When missing data is repaired and the
opportunity bound increases, an identical clean result remains valid under the
stricter information threshold and can resume; a reduced bound or a non-clean
result with a changed bound requires a new scan. Matching thresholds are unchanged. Incompletely
restored published folders are moved back into isolated staging after surviving
files are rehashed; missing files are fetched again. Wrong existing hashes remain errors. Within-folder work is checkpointed per
Parquet row group, with original shard row numbers retained. Changing code requires a method-version bump when its
search semantics change; changing inputs automatically invalidates cached results.

The default fetch process RSS budget and total audit worker budget are **32 GiB**
on the 64 GB guest, leaving memory for the parent, benchmark loading, the
operating system and uploads. Audit `--workers` defaults to `os.cpu_count()`
(or one if unavailable); each disposable worker receives exactly
`memory_budget_bytes // workers`. The total excludes the supervisor and gcloud.
Use an appropriate total on larger machines, for example 256 GiB / 90 workers
(2.844 GiB each), or 512 GiB / 180 workers (2.844 GiB each). A very small quotient
can fail the RSS check at import/startup; no silent worker reduction is made. Override
it with `--memory-budget-gib` on fetch and audit (fractions are accepted).
The job generator accepts the same option; the guest can override it with
`CORPUS_MEMORY_BUDGET_GIB`. Measurement and audit folder workers have a 10 ms
RSS watchdog and check their high-water mark during processing. Exceeding the
budget records measured peak RSS and configured bytes. Sampling can overshoot
between checks; this is not a kernel-enforced memory limit.

The native reader uses `ParquetFile(buffer_size=65536, pre_buffer=False)` and
`iter_batches(batch_size=1, use_threads=False)`. Both options matter: a one-row
batch with the default unbuffered input can still retain an entire compressed
column chunk. Numeric values stay in Arrow-backed NumPy arrays, with no Python
float list and no whole-row float64 conversion. Validation and exact counts use
nonoverlapping 65,536-value blocks. Arrow still decodes an entire row, so the
explicit hard row limit is decoded numeric storage no larger than the process
budget (32 GiB by default): more cannot fit even without decoder overhead.
A single-row group's footer can prove this before decoding; otherwise the row
is checked after decoding, with the watchdog measuring actual decoder and
consumer allocations. There is no speculative multiplier of row/group values. There
is no row-group byte/value refusal and no two-million-value row refusal.
Long rows, including the trainer's >=16,384-step rows, are retained intact.

Offline PyArrow 24 measurements compared 4 versus 34 nested float64 rows, each
2,100,001 steps: 67.2 versus 571.2 MB uncompressed in one group. Unbuffered
Arrow-view peaks grew from 290 to 797 MB; buffered peaks were 229 MB for both.
The old Python-list path peaked at 444 and 1,051 MB. The group is scaled down
from ERA5's 3.91 GB/10,000 rows; the individual rows are deliberately longer.
Regression fixtures also use the actual release types: nested float32 targets
(68 long rows) and float64 targets plus two covariates (12 long rows). Each has
a group above 512 MiB, a row above two million values, and measurement plus
matching scans in subprocesses that assert peak RSS below **512 MiB**. These
are synthetic measurements, not a measurement of the real corpus or Arrow 25.

Audit series chunks retain 65,536 points and a 512-point overlap (0.79% extra
work on long series). Fixed-bin reductions are lazy, preserving global bin phase
and evidence offsets across chunks. Long-parent identification retains the
existing precision, quantization, periodicity and strict-ramp gates using bounded
passes. Frequency discovery projects only `freq`, in 4,096-row batches. Combined
benchmark indices contain at most 262,144 points and 4,096 histories; unresolved
event batches contain at most 128 histories. Batching
adds passes without dropping coverage. Comparison reporting groups metadata
by dataset, avoiding a Python loop over every indexed history for every query.
Histogram totals are computed once per marginal, avoiding quadratic work over
unique lengths. Counts and opportunity arithmetic use Python integers; Parquet
counts and SQLite offsets are 64-bit. Groups are inspected once, including
files with 199 groups; numeric groups are decoded sequentially.

More than one million hash postings per query remains an explicit unresolved
limit before the old matcher expands them. Candidate caps, 4,096 phase coverage,
10,000 evidence records, tolerances and the 512-point overlap are unchanged;
dropped coverage is inconclusive. Calendar aggregation still needs full
benchmark timestamps and can be expensive. Each unit decodes its row group once into SQLite metadata plus native-dtype flat
numeric scratch outside `state`. Later passes map those values without Arrow
redecoding or another numeric Parquet read. Frequency discovery still projects
`freq`, and provenance hashing still reads compressed shards. A bounded group
of original benchmark batches probes each chunk in the same pass; its size is
`max(1, min(4, worker_budget // (4 * 512 * MAX_INDEX_POINTS)))`. The 512-byte/point
planning allowance reserves index construction space; the actual RSS watchdog
remains authoritative. At the example 16/90/180-core budgets the group size is four.
Benchmark metadata is filtered before histories are mapped, and dates are only
materialized for benchmark-to-corpus aggregation. Query hashes are reused across
resident indices and between posting preflight and legacy candidate discovery;
the legacy functions execute unchanged, with an isolated memoized hash table.
The nine benchmark stride indices (strides 2–4 and all phases) are also cached
within a resident batch rather than reconstructed for every missing-data query.
No shared matcher globals are changed.

Units write compact journals, summing consecutive negative-query counters and
flushing at each positive query. Merge heads contain only keys/file offsets; only the
selected payload is decoded. Parent memory and open files do not multiply by
875 potentially large positive payloads. The reducer restores the original frequency,
benchmark-batch, shard, row-group, row, channel, phase and chunk order. It applies
the original 10,000-record cutoff at the original query, including that query's
coverage/comparison counts. A unit also stops after 10,000 local matches: its
traversal is a subsequence of the global traversal, so any omitted work must
follow the global cutoff. After a unit reaches its local cap, speculative queries are limited to its
current resident batch group. Other units still finish their own work before
folder reduction. Largest groups are scheduled first using numeric leaf counts plus row overhead.
Completion order and worker count cannot reorder
evidence or turn resource losses into clean coverage. Unsupported pairs keep the
same skipped reasons; no domain crosswalk, threshold or phase is removed.

`state/audit-units` holds atomic completion markers and SHA256-bound journals;
only in-flight units need retry after interruption. A new worker count can reuse
complete units. Runtime failures retry. The guest sets `CORPUS_AUDIT_STATE_URI`:
the supervisor uploads each completed journal and then its marker, serially,
before counting the unit done, and publishes completed controls before launching
the pool. Periodic snapshots exclude live unit/control/folder checkpoint paths, so a delayed
snapshot cannot overwrite a newer completion marker; the final sync runs after
the pool exits. This closes the ten-minute snapshot gap for completed work. Upload
failure fails the folder/attempt. Ordinary local runs omit this environment
variable and make no cloud calls. A restart still rebuilds benchmark identity and
rehashes corpus files. Disposable numeric scratch is not uploaded; it is removed
when workers finish or terminate and rebuilt only for units requiring work.

A failed speculative worker is checked with the original-order folder scanner
in `audit_serial.py`, under the same per-worker RSS quota and without rerunning
controls. This preserves a match at the legacy cutoff when a malformed or
oversized row would only have been visited after that cutoff. Genuine failures
before the cutoff still fail; upload failures remain attempt failures. The
compatibility pass can reread the folder and add serial time on this rare failure
path. A successful result is durably checkpointed by folder. Progress marks
`status: fallback` / `serial_fallback: true`; its scanned/decoded counters describe
the primary pool and exclude compatibility rechecks.

`state/audit-progress.json` is atomically rewritten about once per second while
the pool runs, with schema version 1, `units_total`, `units_done`, `units_failed`,
`values_scanned` (pool matcher input points, including repeated phases/batches),
`native_values_decoded`, `elapsed_seconds`, nullable
`estimated_remaining_seconds`, and per-folder total/done/failed/status. Elapsed
is pool wall time for the current invocation; ETA extrapolates completed unit
throughput and is provisional for heterogeneous groups. The ten-minute state
sync exposes progress in the bucket; journals/markers have immediate durability.

A folder failure is checkpointed in `fetch-summary.json`, with its name and
error, while later folders continue to download, validate and publish. The
fetch CLI exits 2 after listing all failed folders. Completed folders rehash but
do not redownload or remeasure unless inputs/statistics require it. Audit scans
run in separate workers, record errors, continue, and retry failed checkpoints.
`audit.json.failed_folders` includes runtime errors (also in quarantine) and
non-clean allowed folders. Stable non-clean policy/coverage results can resume;
restarting cannot change their policy meaning. The generated job runs audit
even after fetch fails and exits nonzero after both stages; final uploads still
include every completed folder and both summaries.

## Audit cost model and the six-hour target

The independent checkpoint/scheduling unit is one Parquet row group. The supplied
pinned survey gives **150 shards, 875 units, 121 folders, 7,491,306 rows** and
22.6 GB of compressed data. Its `groups` arrays describe only 215 groups; summing
them is not the corpus total. Recorded numeric leaves give 6,287,711,751 values;
filling each of the 660 omitted groups with its shard's reported maximum gives
a conservative upper bound of 12,066,492,230. Thus the available metadata bounds
numeric values at **6.29–12.07 billion**. The largest fully described group has
977,638,380 values (ERA5); this is a material scheduling tail. Actual footer
values, benchmark batch counts, frequency distributions, missingness and
candidate density must be measured on the guest; 100 datasets alone determines
none of those batch counts.

For a nonempty channel of length `n`, identity search uses
`q(n) = 1 + max(0, ceil((n - 65536) / 65024))` queries, with the original overlap.
For a comparable fixed ratio `r`, fine-to-coarse search uses both sum and mean
for every phase `p=0..r-1`, with length `max(0, (n-p)//r)`. Apply `q` separately
to every nonempty variant. Unsupported ratios/origins still record the original
coverage event, after the unchanged identifying-parent gate. Benchmark-to-corpus
variants expand the index batch count instead. Let `B(u,c,b)` be the original
number of bounded 262,144-point/4,096-history benchmark batches for a unit's
corpus/benchmark frequency pair. A unit costs:

```
T(u) = Arrow_decode(Vu) + numeric_spool_IO(ceil(B/G) passes)
     + sum over frequency pairs/resident groups [parent_identification(Vu)
                                                + corpus_aggregation(Vu, ratio)]
     + sum over frequency pairs/batches/queries [alpha + beta * query_points
                                                + candidate/stride work]
     + sum over original batches [index_build + benchmark_aggregation]
     + journal/checkpoint/upload overhead
T(N) >= max(sum(T(u))/N, max(T(u))) + serial setup/controls/reduction
```

`G` is the resident batch group size; hashes and aggregation preparation are
shared within that group. It saves up to `G-1` numeric spool reads/preparations
per group, while one-time decoding removes all repeated numeric Parquet passes.
It does not divide the number of index comparisons by `G`. Short rows incur a
per-query cost (`alpha`), so dividing all values by 65,536 understates runtime.
Fixed aggregation must also account for input reduction work: for ratio `r`,
`sum over phases/methods r * max(0, (n-p)//r)` approaches `2*r*n` fine input
values read/added, although aggregated output approaches only `2*n` points.
Sum and mean retain their original arithmetic and are separate variants. This
corpus-side work is shared across the `G` resident batches. Benchmark-side
aggregation/index preparation repeats per row-group unit, a cost of finer
parallelism that the simple query-rate table does not include. Sharing prepared
benchmark indices across units is another semantics-preserving follow-up.
Missingness can activate the original stride searches; positive/dense candidates
can cost much more than negative probes.

Using W4a6's measured **0.035 s / 65,536-point negative query**, ignoring short-row
and candidate overhead, one full-corpus identity batch costs **0.940–1.804 CPU
hours**. The following are conditional compute estimates for an effective
value-weighted batch/phase multiplier `B_eff`, before new hash savings, serial
setup and IO. They are not a measured forecast of fev100:

| B_eff | Ideal 16 cores (h) | Ideal 90 cores (h) | Ideal 180 cores (h) | Largest-group floor (h) |
| --- | --- | --- | --- | --- |
| 10 | 0.59–1.13 | 0.10–0.20 | 0.05–0.10 | 1.46 |
| 100 | 5.88–11.28 | 1.04–2.00 | 0.52–1.00 | 14.62 |
| 1000 | 58.76–112.76 | 10.45–20.05 | 5.22–10.02 | 146.17 |

Predicted wall time must be at least the larger of the relevant ideal estimate
and its unit tail, adjusted by measured query throughput. Even **90 or 180 cores
cannot meet six hours at B_eff=100** under that baseline tail. No unconditional
six-hour or fourteen-hour promise is supportable without the benchmark batches.
If the measured tail remains above six hours, the next step is to decode a large
group once into a shared, verified numeric spool and distribute bounded row
ranges or frequency/batch groups across workers; ordered journals already
support the required deterministic merge. This must retain every phase, stride,
comparison and coverage event. It adds durable spool ownership/validation and
must be tested before deployment; increasing cores alone does not remove the tail.

An offline seeded probe with four 262,144-point indices measured 65,536-point
queries at 0.0442 s originally, 0.0229 s/index with two resident batches and
0.0156 s/index with four (about 65% less per index). A 96-point missing-data
query took 1.359 s originally, 0.855 s with cold caches and 0.00785 s with warm
stride caches; outputs were equal. With all four stride caches populated, the
process peak was 786,743,296 bytes (750 MiB); this does not measure decoder or
full-machine peaks on C3D. At the four-batch rate, the conditional B_eff=100 model
has ideal 16/90/180-core compute of 2.62–5.02 / 0.46–0.89 / 0.23–0.45 hours,
but the largest-group floor remains **6.51 hours on all three shapes**, before
setup, IO and short-row work. Thus the six-hour target is still not established.

A critical limitation of that microbenchmark: its 65,536-point negative queries
hit the unchanged posting budget (`unresolved_rows=1`); 336/4,096-point probes
also dropped groups/offsets. The latter took about 0.0066/0.147 s per index,
respectively. These rates describe the existing capped method, not a completed
clean search, and show that query cost is nonlinear in length/candidate density.
The audit retains every such coverage loss as inconclusive. Forecasting real
completion and release validity requires actual batch, length, mask and cap
measurements; do not treat a fast capped negative as clean coverage.

Peak worker RSS is bounded by `total_budget//N` with the existing sampled
watchdog, plus possible between-sample overshoot. Peak machine memory also
includes the supervisor, benchmark loading, OS/page cache and one serialized
uploader; it is not established by the CPU model. Numeric scratch can approach
4–8 bytes per concurrently active native value (up to 50–97 GB for all bounded
corpus values), plus metadata, separate from compressed corpus and benchmarks.

## Cloud guest job and release

```bash
python -m scripts.corpus_v4.make_corpus_job FULL_REPO_SHA \
  gs://YOUR_BUCKET/corpus-v4 job.sh
bash -n job.sh
```

To move the audit to a larger guest without native data refetch:

```bash
python -m scripts.corpus_v4.make_corpus_job FULL_REPO_SHA \
  gs://YOUR_BUCKET/corpus-v4 audit-only.sh --audit-only \
  --workers 90 --memory-budget-gib 256
bash -n audit-only.sh
```

Restore the same `STATE_URI` first, including the complete revision-scoped
`benchmark-cache/<FEV_DATASETS_REVISION>` produced by the original audit. Audit-only
sets `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`: it fails if benchmark cache
inputs are unavailable and never downloads data from the Hub. Installation and
Git provisioning still use their normal package/code sources. Corpus bytes come
only from the published prefix. Before scanning, `verify_published` checks the
corpus identity and all folders against the manifest, and every receipt's full
file list, size and SHA256/Git-blob hash against the checked-in pinned inventory
(`scripts/corpus_v4/pinned_inventory.json`, recorded at the same revision as the
existing release-inventory fixture). It verifies stats hashes and refuses extra,
missing, symlinked, tampered or misplaced data. Receipt claims cannot hide a
missing shard. The verifier has no Hub client/download operation.

Every audit-only invocation has a **new audit attempt ID** and the new code SHA.
It preserves the original source attempt status before replacing the public
gate, and writes `state/audit-provenance.json` with the verified receipt
fingerprint, source prefix, source code SHA and `fetch_attempt_id`. The original
fetch attempt may have an incomplete status because its audit was still running;
complete verified receipts establish fetch completeness. A source audit-only
status retains its original `fetch_attempt_id` on another restart. Both the new
`attempt-status.json` and the new `audit.json` carry this link, while their
`attempt_id` values must agree with each other, not with the fetch attempt.
The job explicitly validates that relationship, current SHA, valid/exit-0 audit
and no failed folders before publishing a complete success. Verification failure
leaves the new attempt incomplete and cannot republish a restored successful
audit. Audit-only uploads audit/status/state; it does not republish corpus bytes.
Do not run competing writers at one corpus prefix; the orchestrator must stop
the old combined job before starting its audit-only replacement.

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
checkout, and rechecks its HEAD against `REPO_SHA`. Metadata lookup excludes the
checkout's source egg-info and respects the installed distributions' path order.
The package and third-party dependencies are imported without stdin's implicit
checkout path so it cannot hide stale site-packages code; the checkout is then
added for the uninstalled `scripts.corpus_v4` modules, whose paths are checked too.
Provenance must name an absolute local file URL, so an empty path cannot resolve
to the current working directory. It records `pip freeze` in
`RUN_DIR/state/environment.txt`. NumPy/pandas/PyYAML are base dependencies;
PyArrow/datasets are in `bench`, and datasets already depends on huggingface_hub.
The guest provides Python with pip and torch, Git, gcloud, application-default
Google credentials, `RUN_DIR` and `STATE_URI`. No credentials are embedded.
The SHA must contain this implementation and frozen manifest.

If `HOME` is empty or unset, the job sets it to `RUN_DIR/home` for startup-script
execution as root. Default Hugging Face caches are under `RUN_DIR/hf-cache`,
outside the periodically uploaded state tree. Native downloads have one regular
file in staging or the published tree, with only transfer metadata alongside.
The audit's explicit benchmark dataset cache remains resumable under state; it
overrides the default dataset cache setting.

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

## Provisional endpoint sketch triage (W4s)

`audit_sketch` produces **provisional triage, never clearance**. It cannot
satisfy an audit or training release gate, even when execution and controls
pass. A negative result is not evidence that the corpus has independent lineage.
Existing audit semantics, numerical thresholds, manifest and generated audit
jobs are unchanged. This baseline does not contain the Simons v4 training
consumer: training gate integration remains an orchestrator check. The existing
audit-only guest contract refuses a renamed sketch report because it lacks
`valid` and `exit_code`; those fields are deliberately absent. Sketch output has
`artifact_type: corpus-sketch-triage`, `provisional: true`,
`release_eligible: false`, and a separate `execution_ok` diagnostic. Never rename
its output to `audit.json` or publish it over the corpus attempt-status gate.

Every numeric target and covariate is read with the buffered native reader,
including quarantined folders. Each transient sketch records complete length,
positional finite-mask digest, canonical float64 raw fingerprint, quantiles,
zero/change fractions, longest positional run, and autocorrelation at lags 1
and 7. Prefix and suffix fingerprints at 24, 64 and 256 positions use float32
values and sign-canonical local z-scores quantised in .05 steps on two shifted
grids. Float32 hashes catch round trips; local normalisation proposes positive
and negative affine copies and shifts. Endpoint positions provide the crop
offset. These grids improve recall but do not guarantee it across quantisation
boundaries. No missing positions are compacted or imputed.

Only benchmark keys persist: sorted 128-bit fingerprints plus owner and position
occupy 32 bytes per posting, at most 20 postings and 44 feature bytes per benchmark
channel (684 bytes/channel). Workers map immutable files and discard corpus
sketches after each channel. The existing native reader and its memory watchdog
remain in charge of buffered Arrow decoding and indivisible whole rows.

Exact/endpoint hits propose original-array alignments. Before calling
`audit_match.fit_segment`, the worker applies that verifier's own `nondegenerate`
function to **the exact overlap and joint finite mask**, on both sides. A failure
cannot become a verified match. Such pairs still enter proposed/unverified counts,
the rejection histogram, and evidence, with `fit: null` and an explicit `shortcut`
label; no coefficients or fit scores are invented. Exact positional masks and
bounds key bounded caches; whole-channel degeneracy never substitutes for an
aligned segment's decision. Other pairs retain every original numerical,
short-period, calendar, strict-ramp and information gate. No threshold changes.

Summary buckets use affine-invariant quantile shape and lag-1 correlation. Each
worker records its closest neighbour as a deferred proposal. Folder assembly
replays channels in native shard/row-group/row/column/channel order and applies
the original adaptive top-20 admission rule. Thus worker count and interruption
order cannot change which summary proposals count, the rejection histogram,
match sample or ranking. Ranking keys are encoded once rather than repeatedly
sorting/encoding all 20 records. Equal-key buckets still cap postings at 64 and
record unresolved diagnostics. Metadata routing never excludes value pairs.

The scheduling unit is a Parquet row group or bounded row range within a large
group, across **all** folders, largest estimated numeric value count first.
Ranges target at most 16M values from group metadata (8192 rows maximum, with an
indivisible row allowed to exceed the estimate). Thus a billion-value group is
spread across workers rather than leaving one worker with a multi-hour tail.
The unchanged reader re-decodes each range's group prefix, skipping its sketch
work. This repeats decoder/I/O work; a 1B-value group needs about 63 ranges, or roughly
32 whole-group decodes in aggregate. The supplied reader/statistics timings are
32.75s for era5's 1.987B values, 23.19s for m100-ipmi's 1.996B, and 549.88s for
gpm-mergir's 2B. These demonstrate that verification was much slower than a single
reader pass, but do not establish concurrent prefix-decoding cost. The 16M work
unit amortizes that cost; finer 2M/512-row checkpoints separately bound lost work. Each unit has an identity-bound checkpoint and immutable gzip
JSONL journal under `sketch/evidence/units`. Within a unit, closed fragments are
checkpointed every 512 rows or 2M values, whichever comes first, after finishing
the current row. An interruption loses only the last fragment's sketch work
(plus one indivisible row). The unchanged reader may re-decode a completed prefix
on resume; its fingerprints, lookups and verifications are reused. Fragment gzip
headers fix timestamp/name, so completed bytes and summaries are deterministic.

Folder `*.jsonl` journals are small **unit-reference JSONL** files, each line naming
an immutable unit journal and SHA256. `evidence.format` distinguishes these from
legacy full pair journals. Unit gzip files retain pair/cap evidence and deferred
summary events; assembly records admitted counts, first 100 verified matches and
best 20 unverified pairs. Deferred events excluded by the original admission rule
do not count as proposals. Allowed and quarantined counts remain separate; a
folder never becomes clean. Full completed journals and their referenced bytes
are rehashed before reuse. Corrupt folder references can be rebuilt from valid
unit checkpoints. Corrupt/missing unit evidence replays that unit or its valid
fragments. Growing output, controls, benchmark downloads/spool and index builds
live in a sibling scratch directory **outside state** and enter state only after
closing and atomic rename. Scratch, state and output (including evidence/unit
directories) must be on the same filesystem. Startup checks their devices before
preparation or workers and rejects a different device with a clear error; separate
filesystems are not supported.

Controls select numerically eligible benchmark histories of 96–65,536 positions,
without screening for retrieval success. Whole, positive/negative affine, shift,
float32, prefix and suffix plants travel through the native reader, full crowded
index, worker and verifier. Interior crops are a deliberate miss; a crop whose
endpoint keys match its parent is skipped and named in `skipped_cases`, since a
periodic interior can be a genuine endpoint-preserving copy. Positive plants
are never screened for retrieval success. Constants,
ramps and a calendar motif are negative controls. Correct benchmark and exact
overlap are required. A passing outcome and its evidence persist in
`state/sketch-controls.json`; unchanged resumes rehash and reuse it. Failed or
corrupt controls rerun. Controls demonstrate only the planted instances; changed
masks, unstable endpoints and all aggregation phases remain unsupported.
Failed cases are named with their benchmark, exact-overlap outcome, verifier
rejection reasons and retrieval diagnostics in JSON and Markdown. A prefix copy
can be missed when its common endpoint postings are capped and the bounded
summary shortlist does not propose the parent; planting a copy guarantees
neither retrieval nor verification.

The CLI first calls `verify_published` for the **whole** corpus, even with a folder
selection. Benchmark reuse binds the pinned revision, complete task definitions,
all local sources executed by benchmark preparation (`audit_sketch`,
`audit_fev_overlap`, `native`, `matching`, task `specs`, `pretrain.audit`,
`audit_match` and `audit_info`) and SHA256 of SQLite, values and timestamps;
production refuses injected synthetic or incomplete benchmark spools. Index reuse binds benchmark
identity, method/source and every index file hash. `state/sketch-input.json`
records the full current scan contract and its fingerprint for inspection.
Folder/unit reuse binds the complete manifest, **all corpus shard bytes**, full
benchmark identity, index bytes, opportunity bound, method and implementation
sources (sketch, verifier, information, matching, reader). Worker count, memory
budget and selection do not change numerical identity. A source/input change,
missing identity, malformed marker, result digest mismatch or evidence hash
mismatch forces replay; excluded folders with unusable summaries fail closed.

Legacy `sketch1` benchmark markers and folder summaries are not reused: they lack
the complete loader-source contract. The next cloud run rescans all 121 folders.
Resumes written by the current code continue to reuse matching, rehashed spool,
index, control, unit and folder artifacts. No cloud reuse was established by the
offline tests.

Generate a separate guest job using the existing `RUN_DIR` / `STATE_URI` host
restore/periodic/preemption-copy contract:

```bash
python -m scripts.corpus_v4.make_sketch_job FULL_REPO_SHA \
  gs://veliu-forecast-train-us-east1/corpus-v4 \
  gs://veliu-forecast-train-us-east1/reports/sketch2 sketch-job.sh \
  --state-prefix gs://veliu-forecast-train-us-east1/runs/sketch2/state \
  --workers 48 --memory-budget-gib 320 --skip-finished
bash -n sketch-job.sh
```

Start this run with its own state URI and scan all folders. Subsequent resumes
restore its complete state, including closed evidence files, as the host
normally does. `--only-folders` selects folders eligible for scanning; every other folder must
have a valid completed summary. Without `--skip-finished`, completed selected
folders are rescanned, including their units and fragments; interrupted folders
retain bounded-unit resume behavior. `--skip-finished` also reuses valid selected
summaries. Ordinary invocations retain resume behavior for all folders. All
folders remain in the assembled report and global opportunity bound. Selection
does not authorize a partial corpus download or a smaller statistical bound.
Generation checks destinations and folder syntax, and performs no cloud call.

`state/sketch-progress.json` is atomically replaced every 30 seconds throughout
setup and scanning, and on completion. It contains phase, units done/total,
checkpointed values/channels (including partial units), per-folder progress,
new values/minute and estimated seconds left. Reused units and recovered partial fragments are excluded
from the throughput numerator. Metadata supplies estimated remaining values;
ETA is null before new work completes and zero at successful completion. The
generated job copies this small object directly to state/report GCS prefixes
every minute, independent of the host's slower state copy. Final evidence
publication precedes separate completion status; upload failure stays nonzero.
No audit/attempt-status gate is written.

The original unmeasured 100-minute forecast is withdrawn. The 90 supplied finished
summaries contain 410.65M values, 60,742 channels and 2,828,309 proposals, with no
verified match; 1,658,303 proposals failed non-identifying gates. Statistics total
**11.0825B remaining values / 13.773M channels / 7.474M rows**, confirming the
handoff's rounded 11.08B after normalizing quarantine paths. Reproduce the tiny synthetic profile without downloads:

```bash
OPENBLAS_NUM_THREADS=1 python -m tests.test_corpus_v4.test_audit_sketch_benchmark /tmp/sketch-bench
```

This compares locally available 0523bc6 code with the new scanner, using 120×336
short channels, 24×8192 long channels and 36×8192 constant-heavy channels against
64 tiny synthetic benchmark histories. It records per-channel/per-value time,
decode/sketch/lookup/verification/emission/ranking/assembly, bytes and cProfile
artifacts. Three fresh-process profiled repetitions per cell measured:

| Shape | Old → new µs/channel | Old → new ns/value |
|---|---:|---:|
| 336-point short | 1265 → 1397 | 3766 → 4158 |
| 8192-point long | 5223 → 5619 | 638 → 686 |
| 8192-point constant-heavy | 16509 → 3913 | 2015 → 478 |

The constant workload improved 4.22×: verification fell from 289 to 27ms and
ranking from 162 to 8ms for 36 channels / 1728 pairs; compressed evidence fell
from 1.48MB to 11KB in the direct-scan harness. Short/long informative shapes
were 10%/8% slower because the precheck adds work; their dominant costs were
sketching and benchmark mapping/verification, respectively. New-stage median
milliseconds for short/long/constant were decode 11.3/6.7/6.8, sketch
54.9/19.6/13.3, lookup 10.7/2.4/18.4, verification 35.9/47.6/27.2,
benchmark mapping 38.7/47.6/11.2, emission 2.4/0.8/12.2 and assembly
2.0/1.0/19.3. Profiling includes the reader watchdog; process import/index setup
is excluded. These small workloads do not measure full-corpus range rereads.
It cannot measure the full benchmark index, real distribution of
masks/proposals, cold downloads or Spot/GCS throughput. Cloud observations were
about 1M values per worker-minute; planning below allows **1–2×** overall speedup
and 30–60 minutes for setup/transfer. These are conditional ranges, not promises:

| Machine | Workers | Total RAM / process budget | Expected remaining pass |
|---|---:|---:|---:|
| n2d-standard-32 | 16 | 128 / 104 GiB | 6.3–12.7 h |
| n2d-standard-64 | 32 | 256 / 208 GiB | 3.4–6.8 h |
| n2d-standard-96 | 48 | 384 / 320 GiB | 2.4–4.9 h |
| c3d-standard-16 | 8 | 64 / 52 GiB | 12.2–24.3 h |

N2D 96 fits the stated 200-vCPU quota; C3D is limited to the stated 24-vCPU quota.
[Google’s machine table](https://docs.cloud.google.com/compute/docs/general-purpose-machines#c3d_machine_types)
lists C3D 16 then 30 vCPUs; the latter exceeds the 24-vCPU quota.
C3D uses the same conservative per-worker rate because it was not measured.
A few hours is plausible on N2D 96; a four-hour finish is **not established**.
Candidate verification and per-channel overhead, rather than decoding alone,
limit the estimate. More informative/masked proposals, oversized row groups,
poor CPU scaling, evidence volume, disk/GCS contention, a cold benchmark load or
preemption can exceed these ranges. Budget at least 500 GiB disk pending measured
compressed evidence volume. The full cloud run must establish index build time,
per-shape throughput, compressed evidence bytes, decoder RSS, controls against
the full index, ETA calibration and interruption recovery/upload behavior.
