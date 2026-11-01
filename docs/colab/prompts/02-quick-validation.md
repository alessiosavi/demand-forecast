# Prompt 02 — quick validation on the seven weekly tasks

Paste these instructions into a Claude session that can drive Colab (see
`docs/colab.md`). Fill `<SHA>` with the full public commit SHA containing this prompt,
`configs/fev-v2-quick.yaml` and `scripts/fev_report.py`. This authorizes ONE labelled
quick-validation run, seed 0, policy `first_window`, on all seven weekly tasks; it
authorizes no tuning, second seed, configuration change or variant selection.
For the authorized two-task rerun after the non-finite-input fix, §5 overrides the
run directory, task selection and corresponding counts; all other rules still apply.

You are the executing agent. Create, run and save a Jupyter notebook on my **hosted
Colab free-tier kernel**, T4 GPU, Runtime version **2026.07** (Python 3.12), using
`https://github.com/alessiosavi/demand-forecast` at **`<SHA>`**. Follow
`docs/protocol.md` and `docs/colab.md` at that commit. Save the notebook and artifacts
on Google Drive. Never execute this run on my local machine, download Timestral
weights, or edit either the quick or frozen configuration. The frozen protocol run
is separate and will move to EC2.

## 0. Rules that override convenience

- The label sentence, verbatim from the protocol and quick YAML header, is:
  **Quick validation of the pipeline, NOT the protocol: scores from this config must never inform a configuration decision.**
  Quick benchmark scores are pipeline checks only and must never inform a
  configuration, early-stopping, tuning or ablation decision. Use this sentence as
  the first line of the hand-composed `report.md` and above every table shared elsewhere.
- **No quick-model benchmark score is viewed before §4.** Before then,
  `summaries.csv` may only be read programmatically to display rows filtered to
  `model_name == "seasonal_naive"`, row counts, `run_identity` values and file hashes;
  never print a quick-model row or metric column, and do not run `fev leaderboard`
  or `scripts/fev_report.py`. Inner-validation numbers in logs/`run.json`/
  `progress.json` (`best_score`, `inner_SQL`) are diagnostics and may be read.
  Reuse only prerequisite evidence from prompt 01; do not expose its frozen-model
  benchmark scores while preparing this run.
- A `run identity mismatch` stops the run: preserve and report the error; never
  delete or move artifacts to force a resume.
- Use `FEV_RUN_DIR=$FEV_DRIVE_ROOT/runs/02-quick-v2-<SHA>` throughout, with
  provenance under `$FEV_RUN_DIR/provenance-$FEV_SHA/`.
  `TRIAL` is unused; do not set it or create a trial directory for this prompt.
- Long commands use the detached Python launcher in `docs/colab.md` §4. For each
  command below, set its command list to the shown command (including
  `/usr/bin/time -v` where shown); retain the script's `set -euo pipefail` and
  `trap 'echo $? > "$LOG.exit"' EXIT`. Run the command WITHOUT `tee` in the script:
  `subprocess.Popen(["bash", script], stdout=open(LOG, "a"), stderr=subprocess.STDOUT, start_new_session=True)`
  already redirects output. Give every log, script and `.exit` path a fresh UTC
  launch timestamp (`%Y%m%dT%H%M%S%fZ`), keep the fresh-file assert and process handle,
  and preserve earlier launches. Use distinct baseline, quick-run and report names.
- Start the same one-second `nvidia-smi` VRAM poller as prompt 01 §1 and template §4,
  with a fresh UTC launch token and `start_new_session=True`. Both launcher and
  poller must be isolated from the kernel's process-group interrupt/restart signals.
  Poll log/`.exit` paths independently of Python handles, including after a kernel
  restart; use short cells with `tail -n 5 "$LOG"` every few minutes. Colab has no
  background cells and `%%bash` output appears only when the cell finishes.
  For a deliberate stop, import `signal` and use `os.killpg(process.pid, signal.SIGINT)`.
- `.exit` holds the raw status: 0 is success; an absent `.exit` after a VM reset is
  unknown/interrupted, never success. No deliberate interruption trial is scheduled
  here. If handling an expected exit 1, preserve the prompt 01 §0 pattern:
  `set +e; <cmd> 2>&1 | tee -a "$LOG"; rc=${PIPESTATUS[0]}; set -e; test "$rc" -eq 1`
  for a foreground cell; for the detached launcher, keep the raw `.exit` and assert
  1 afterwards in the notebook, without adding `tee`. Unexpected nonzero exits
  remain failures; preserve the traceback and mark completion accurately.
- Record every measurement and failure; label unverified claims. At session end,
  stop/wait for the poller and copy its local log to Drive with template §4's cell.
  The maximum is a one-second sampled device-wide VRAM maximum, distinct from
  allocated tensor memory. A VM reset loses the uncopied local tail; only completed
  Drive copies are durable evidence.

## 1. Setup — prompt 01 rev 2 §1–§2 with explicit overrides

Follow [prompt 01 rev 2](01-fev-run-v2.md) **§1 (runtime, Drive, paths and hardware)**
and **§2 (install, dependency lock, dataset revision)**, using the template cells
from `docs/colab.md` §1–§2. Apply these overrides before executing any cell:

1. Set `FEV_SHA=<SHA>`, `FEV_SESSION=<UTC YYYYMMDDTHHMMSSZ>` and the quick run
   directory from §0 in `os.environ`, so shell cells inherit them. Keep
   `FEV_DRIVE_ROOT=/content/drive/MyDrive/demand-forecast`,
   `HF_HOME=$FEV_DRIVE_ROOT/hf`, `HF_HUB_CACHE=$FEV_DRIVE_ROOT/hf/hub` on Drive and
   `HF_DATASETS_CACHE=/content/hf-datasets` locally. Use neither prompt 01's frozen
   output directory nor its trial path. Metadata `scope` is `"quick_validation_v2"`;
   seed remains 0 and policy remains `first_window`.
2. Reuse and cite the recorded A–D prerequisite evidence at
   `$FEV_DRIVE_ROOT/runs/01-fev-run-v2-67d4bd7877e707215ef31fd7123454f5dc017b56/`.
   This historical Drive path keeps its original name; the equivalent code in the
   rewritten history is `bdb8c119f87750066026ddce48121d61aa0fdb8c` (see the
   [history-rewrite mapping](../../protocol.md#history-rewrite-2026-11-01)).
   Record each prerequisite's status and artifact paths in the notebook; missing or
   failed evidence blocks the run. Do not rerun A–D. The seasonal-naive rows in §2
   below populate this quick run directory. **Prompt 01 §6's rerun policy does not
   apply**: do not resume into any prompt 01 directory or reuse its model artifacts.
3. Keep prompt 01 §2's dependency-lock procedure in the quick directory:
   first session creates `environment-lock.txt` and the per-session freeze; later
   installs use `-c` and require an empty freeze diff. Preserve the public checkout
   SHA check, GNU time guard, closure-aware `pip check` (exit 0), and T4 AMP check
   (`torch.float16`). On **every fresh VM**, do ONE **Runtime > Restart session and
   run all** after the first editable install, even if package versions already
   match. Keep the restart-check cell. The second install must retain every version
   and produce an empty freeze diff. Import/execution failures remain fatal.
4. Before ANY dataset load in EVERY session, verify online that both
   `HfApi().dataset_info("autogluon/fev_datasets").sha` and the `main` target from
   `HfApi().list_repo_refs("autogluon/fev_datasets", repo_type="dataset")` equal
   **`d3e7e720f4158eab03ebcb021f1cce779cd92acc` exactly**. Record that revision in
   `$FEV_RUN_DIR/dataset-manifest.json`; an existing manifest must also match.
   **A different revision blocks this run**, including its first session: prompt
   01's fresh-experiment allowance is overridden. Keep online cache regeneration;
   do not enable offline mode or override the task dataset paths. After §2's first
   load, record each of the seven cached parquet paths under `HF_HUB_CACHE`, size
   and sha256, checking the hash against the LFS pointer at that revision; re-verify
   them after the first load in every later session, before continuing to §3 or §4.
5. Keep the hardware and memory gates from prompt 01 §1. Record T4 capability
   `(7, 5)`, VRAM, `/proc/meminfo` MemTotal and **MemAvailable**, and free bytes on
   Drive and `/content`. Compute `RSS_LIMIT = 0.8 × MemAvailable` in **kB** (the
   `/proc/meminfo` values are KiB) and compare directly with `/usr/bin/time -v`'s
   `Maximum resident set size (kbytes)`, without conversion. Gate peak RSS at
   `RSS_LIMIT`, sampled peak VRAM at **13 GiB**, and Drive free at **300 MB minimum**.
   Record gate outcomes, stop/report a failure, and retain the launcher and poller
   from §0. Reused memory evidence does not replace measurements of this run.
6. Create `provenance-$FEV_SHA/` before writing `commit.txt` and the protocol/task/
   profile snapshots from the template. Also copy `configs/fev-v2-quick.yaml` there;
   preserve all earlier provenance folders. Export the executed notebook to this
   quick run directory before ending each session.

```bash
set -euo pipefail
cd /content/demand-forecast
mkdir -p "$FEV_RUN_DIR/provenance-$FEV_SHA"
cp configs/fev-v2-quick.yaml "$FEV_RUN_DIR/provenance-$FEV_SHA/"
```

## 2. Seasonal naive in the quick run directory

Run this command through §0's detached launcher, with the poller running:

```bash
set -euo pipefail
cd /content/demand-forecast
/usr/bin/time -v demand-forecast fev baseline --tasks weekly --out-dir "$FEV_RUN_DIR"
```

Require exit 0 and exactly seven `seasonal_naive` rows, one per weekly task. Follow
prompt 01 **§3's SQL/MASE comparison against prompt 00 to six significant figures**,
using the prompt 00 artifact path cited there. The dataset must match, so this is an
equality check, not an informational comparison: record both values for differences
and stop/report before §3 if they differ. Display only baseline rows. Complete the
parquet manifest check from §1. Baseline resume is automatic; it has no `--resume`
flag. On a rerun require seven `Skipping completed task <name>` lines and an unchanged
`summaries.csv` sha256; do not copy frozen-model summaries into this directory.

## 3. Quick run — fixed configuration, no score-driven variant selection

Before launch, write in the notebook: **The quick configuration is fixed; no
score-driven variant selection follows this run.** The 300-step cap, validation
subsample, patience, seed and all other settings stay fixed after scores are exposed.

Use §0's detached launcher and poller for this exact command:

```bash
set -euo pipefail
cd /content/demand-forecast
/usr/bin/time -v demand-forecast fev run --tasks weekly \
  --config configs/fev-v2-quick.yaml --policy first_window --out-dir "$FEV_RUN_DIR" \
  --resume --model-name demand-forecast-v2-quick
```

Require exit 0 for a complete command. For each task, check the INFO log and then
its `run.json`/`checkpoints/progress.json`:

- `fit: max_steps=300 eval_every=50 … val_items=<n>`, with `n <= 1024`;
  `plan: … val_items=<n>/<eligible>` records the sampled and eligible counts.
  `run.json` must have `val_items_max == 1024`,
  `n_val_items == min(n_val_items_eligible, 1024)`, and
  `resolved_budget == {"max_steps": 300, "eval_every": 50}`.
- Tasks with validation finish with `early_stop` or `max_steps`, with at most 300
  optimizer steps. For `rossmann_1W` (and any other exception task), expect
  `exception=True`, `val_items=0`, no validation selection, 300 steps and
  `stop_reason=exception_budget`. Do not misclassify that expected stop as a failure.
- Every task must show `fit_result.nonfinite_losses == 0`. A `nonfinite_losses` stop
  (also published in `checkpoints/progress.json`) or a positive count is a failed run,
  never a configuration question. Preserve the checkpoint, logs and failure evidence;
  do not treat this exit 1 as an expected interruption or change the configuration.
- Preserve `panel: task=… window=… build_s=…`, `origins:` diagnostics and each
  `eval:` line with `train_s_per_step` and `eval_s`. Record `max_distinct_origins`
  in §4. Unexpected budgets, counts or stop reasons require investigation and a
  reported failure, never a configuration adjustment based on quick scores.

**About one hour is an unverified projection**, including baseline, inference and
reporting, not a measured completion time. The plan cites rev-2 panel-build
measurements of m5 ≈ 600 s, hermes ≈ 205 s and favorita ≈ 25 s per window × 10 windows;
cite the reused evidence and remeasure them here. Panel builds are expected to dominate.
`/usr/bin/time -v` around the whole command measures whole-command wall and peak RSS.
Derive per-task wall intervals from log timestamps, state their boundaries and any
unattributed time, and do not substitute trainer time for task wall. Per-task RSS
peaks are **not measured** by that command; mark them unmeasured. Report the poller's
sampled device-wide VRAM maximum separately from trainer `peak_cuda_mem_bytes`;
attribute samples to task intervals only when timestamps support it. Missing intervals
or lost log tails are unknown. Keep §1's RSS/VRAM gates.

After a disconnect, repeat **§1**, including the install/restart/lock and dataset
checks, then repeat the same command and paths with `--resume`. Poll an existing
launch before starting another; never run two commands against the same directory.
Completed tasks skip; interrupted training resumes from `checkpoints/latest.pt`.
Do not reorder, drop or add tasks. Preserve every launch log and raw `.exit`.

## 4. Report — first look at quick-model benchmark scores

When setup, baseline and run checks have been recorded, quick-model scores may be
read here, always under the label sentence. Re-verify the dataset as in §1 before
the report loads it. Use the detached launcher if this audit is long-running:

```bash
set -euo pipefail
cd /content/demand-forecast
python scripts/fev_report.py --run-dir "$FEV_RUN_DIR" --tasks weekly \
  --baseline seasonal_naive --model-name demand-forecast-v2-quick \
  --out "$FEV_RUN_DIR/report-tables.md"
```

These are generated tables, not the final labelled report. Require exit 0 and a
complete audit for a complete run; a nonzero exit still leaves diagnostic tables.
Preserve them and report partial/blocked status with the missing evidence; do not
use `--allow-partial` to claim success. Do not select another variant after reading
any score, including partial results.
Every task must show `fit_result.nonfinite_losses == 0`; record this gate per task.
A `nonfinite_losses` stop or a positive count is a failed run, never a configuration
question, even if saved predictions and benchmark metrics are finite.

Compose `$FEV_RUN_DIR/report.md` **by hand**, with the following sentence as its
**first line**, verbatim (no title before it):

```text
Quick validation of the pipeline, NOT the protocol: scores from this config must never inform a configuration decision.
```

Then record `model_name: demand-forecast-v2-quick`, config
`configs/fev-v2-quick.yaml`, producing code SHA `<SHA>` and scope
`quick_validation_v2`. Before pasting the generated tables, include one row per task
with `n_val_items`, `n_val_items_eligible` and `val_items_max` read directly from that
task's `run.json`. The generated Training table also contains those counts and limit,
alongside budgets, `exception`, steps, best step/score, stop reason and resume step.
Check that all seven tasks have these fields. Then paste `report-tables.md`, retaining
its completeness result and caveats. Any table shared elsewhere carries the label
line above it; raw generated tables must not be shared without it.

Follow prompt 01 **§7's evidence/report style**, adapted as follows. Paste this
information back with the labelled tables:

```text
Scope/status: quick_validation_v2; quick run complete|partial|blocked; sessions used:
Model/config/code SHA: demand-forecast-v2-quick; configs/fev-v2-quick.yaml; <SHA>:
UTC sessions, Drive folder, executed notebook path:
Reused A–D: each pass|fail|missing, artifact paths in the rev-2 prompt 01 directory:
Lock: creation session; later freeze diffs empty (yes|no); pip-check closure result:
Hardware: GPU, capability, amp dtype, VRAM, MemTotal/MemAvailable, RSS_LIMIT in kB; gate outcomes:
Dataset manifest: required Hub revision; both online checks per session; seven parquet paths/sizes/sha256 and LFS checks; equal to prompt 00 (yes|no):
Baseline: seven seasonal_naive rows; per-task SQL/MASE equality to six significant figures:
Per task: steps, stop_reason, best_step, n_val_items/n_val_items_eligible, val_items_max:
Per task: fit_result.nonfinite_losses == 0 gate; cold_covariate_columns, floored_covariate_columns, clipped_covariate_values:
Per task: timestamp-derived wall with interval boundaries, panel: build_s per window, train_s_per_step, eval_s, inference seconds, max_distinct_origins:
Memory: whole-command peak RSS in kB; per-task RSS unmeasured; sampled VRAM maximum and interval attribution if available; peak_cuda_mem_bytes per task:
Quick scores: SQL ratio vs seasonal_naive (or not comparable), 80% coverage/width, fallback counts, metrics/per-knot pinball, exclusions and completeness from report-tables.md:
EC2 performance evidence: panel build per task/window, first-evaluation batch-build seconds if logged (otherwise unmeasured), per-step seconds; distinguish measurements from projections:
Whole-command timings: baseline, quick run and report where measured; interruptions, gaps and unknowns:
Information-set caveat: walmart's early-ending items are scored on periods other items' available past covers; this is the information set fev gives every model, not an out-of-time test for those items.
Identity errors, interruptions/reconnects and tracebacks (verbatim or none):
Artifact sizes, Drive bytes used/free, /content cache size; deviations and anything unverified:
```

Verify on Drive: `report.md`, `report-tables.md`, `summaries.csv`, per-task `run.json`,
`checkpoints/progress.json`, `bundle/manifest.json`, every expected `window_*.npz`
with diagnostics, all command logs and `.exit` files, completed VRAM-log copies,
`environment-lock.txt`, per-session freezes, `pip-check-*.txt`, `run-*.json`,
`dataset-manifest.json`, provenance `commit.txt`/protocol/task/profile snapshots and
`fev-v2-quick.yaml`, plus the exported executed notebook. Include paths to reused
A–D evidence; preserve its original directory. Missing evidence stays explicit.

## 5. Rerun of walmart and rohlik_sales_1W after the fix

This is the authorized rerun of the two failed tasks at the fix commit `<SHA>`.
It is also the **T4 fp16 acceptance check for the 1e3 covariate bound**: a hosted
Colab T4 runtime is required, with capability `(7, 5)` and AMP `torch.float16`.
A CPU run, fp16 emulation or another GPU does not satisfy this acceptance check;
record it as blocked if a T4 is unavailable. Do not rerun the five clean tasks.
The quick configuration, seed 0, `first_window` policy and score-use restrictions
remain fixed. No tuning or configuration selection is authorized.

Repeat §1–§2 with these overrides applied **before any cell runs**:

- Set `FEV_SHA=<SHA>` and
  `FEV_RUN_DIR=$FEV_DRIVE_ROOT/runs/02b-quick-v2-<SHA>/` in `os.environ`.
  Use a new directory; do not resume into the failed seven-task quick run or a
  prompt 01 directory. Preserve their artifacts and producing SHAs. For a reconnect
  to this two-task rerun, use this same new directory and the usual `--resume` rules.
- Keep the install/restart, dependency lock, provenance, online dataset-revision
  verification, launcher, raw exit status, VRAM poller and memory gates. Check the
  **Drive free-space gate of 300 MB** as before. All task-specific expectations in
  §1–§4 now apply to these two tasks: two parquet manifest entries, two baseline rows,
  two quick-model rows, and two `Skipping completed task <name>` lines on each
  completed baseline/model rerun, with unchanged evidence hashes. Keep every item,
  horizon and window of both tasks; do not substitute a smaller dataset.
- Reuse prompt 00 baseline evidence and prompt 01 prerequisite A–D evidence, with
  their recorded producing SHAs; do not rerun A–D. These evidence directories may
  be restored copies on a different Drive: the Colab operator byte-copied `runs/`
  with a sha256 manifest at
  `MyDrive/demand-forecast/restore-from-old-account-<UTC>.json`.
  **Before citing any restored file, the notebook must re-verify its sha256 against
  that manifest on the fresh VM.** Record the manifest path, each cited file's path,
  expected/observed sha256 and verification result in this run's `report.md`.
  A missing manifest entry or hash mismatch blocks use of that evidence; preserve
  and report it. The `hf/` directory is **not restored**: regenerate it online using
  the existing §1–§2 steps and re-verify the pinned dataset revision and both parquet
  hashes against their LFS pointers before continuing. Restored evidence does not
  replace the fresh runtime, dataset or free-space gates.

Run the two-task baseline through §0's detached launcher and poller:

```bash
set -euo pipefail
cd /content/demand-forecast
/usr/bin/time -v demand-forecast fev baseline --tasks walmart,rohlik_sales_1W \
  --out-dir "$FEV_RUN_DIR"
```

Require exit 0 and exactly two `seasonal_naive` rows. Compare SQL/MASE for these two
tasks with the verified prompt 00 evidence to six significant figures as in §2;
stop on a mismatch. Baseline resume is automatic: a repeat must produce two skip
lines and leave `summaries.csv` sha256 unchanged. Display only baseline scores.

Then run the unchanged quick configuration, with a fresh launch token:

```bash
set -euo pipefail
cd /content/demand-forecast
/usr/bin/time -v demand-forecast fev run --tasks walmart,rohlik_sales_1W \
  --config configs/fev-v2-quick.yaml --policy first_window --out-dir "$FEV_RUN_DIR" \
  --resume --model-name demand-forecast-v2-quick
```

Require exit 0 and record the §3 budget/validation checks for both tasks before
viewing quick-model scores. In addition, record these acceptance checks per task:

- `fit_result.nonfinite_losses == 0` in `run.json`, with the cumulative count also
  zero. A `nonfinite_losses` stop or any positive count is a failed run, never a
  configuration question. Preserve terminal checkpoints and `progress.json`;
  repeatedly resuming a terminal failure does not retrain it.
- `timings.cumulative.cold_covariate_columns`, `floored_covariate_columns` and
  `clipped_covariate_values` are present and recorded, including zeros. Legacy
  `unknown` values are allowed by the general report, but do not satisfy this new
  run's counter-evidence requirement.
- `fit_result.stop_reason` and `checkpoints/progress.json["stop_reason"]` must be
  equal to each other and both be `early_stop` or `max_steps`, with matching final
  steps. Neither task is expected to take the short-history exception.
- Load the task's `checkpoints/latest.pt` on CPU. Report its GradScaler
  `scaler["scale"]` **diagnostically only**, with no exact value or lower-bound
  gate: two legitimate overflow back-offs can give 16384, and scale growth needs
  2,000 steps, beyond this quick run. A finite loss can still have overflowing
  gradients, so the zero-loss-count check alone does not prove optimizer updates.
- **Observation, not a gate:** expect `timings.cumulative.cold_covariate_columns > 0`
  for walmart (MarkDown futures inside training windows after about 2011-02) and
  `timings.cumulative.floored_covariate_columns > 0` for rohlik_sales_1W (tiny-std
  prices and discounts). A zero means "investigate and report", not a failure.
  Record both counters' values in `report.md`'s observations line together with
  the GradScaler `k = log2(65536/scale)`.
- Require finite model parameters in the checkpoint's `model` and `best_state`
  and in the saved bundle, and finite prediction `arrays` in every `window_*.npz`.
  Record checked tensor/array counts and any failures.
- Show actual optimizer-update evidence from `checkpoint["optimizer"]["state"]`:
  require nonempty AdamW state with positive finite per-parameter `step` values;
  record how many parameter states have positive steps and their minimum/maximum,
  alongside the checkpoint training step. Inspect and report finite `exp_avg` and
  `exp_avg_sq` tensors. Do not infer updates from the training step, scheduler,
  finite parameters or loss count alone, and do not require optimizer steps to
  equal training steps (GradScaler may legitimately skip gradient overflows).

Generate the audit only at §4's score-viewing stage, with the same selected tasks
and model label:

```bash
set -euo pipefail
cd /content/demand-forecast
python scripts/fev_report.py --run-dir "$FEV_RUN_DIR" --tasks walmart,rohlik_sales_1W \
  --baseline seasonal_naive --model-name demand-forecast-v2-quick \
  --out "$FEV_RUN_DIR/report-tables.md"
```

Require report exit 0 and complete evidence for both tasks; `--allow-partial` cannot
establish acceptance. Compose `report.md` using §4's exact first-line label and
evidence format, with two-task counts, the fix `<SHA>`, the restored-evidence
manifest path and fresh verification results, all gates above, measured timing and
memory, and the executed notebook path. Report T4 fp16 acceptance as passed only
when every gate passes; otherwise preserve artifacts and state failed or blocked
with the missing evidence. Keep the earlier seven-task quick report separate.
