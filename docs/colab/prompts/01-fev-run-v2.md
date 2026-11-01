# Prompt 01 — phase-5 prerequisites and the frozen v2 run on the seven weekly tasks — rev 2

**Revision note:** rev 1 at `d683bf7` stopped on walmart
(`windows with item-specific origins are not supported`, exit 2) after hierarchical_sales_1W,
rossmann_1W and favorita_stores_1W completed. Rev 2 runs at the fix SHA `<SHA>`, adds
prerequisite D and the rerun policy below.

Paste the following instructions into a Claude session that can drive Colab (see
`docs/colab.md`). Fill `<SHA>` with the full public commit SHA that contains this prompt,
`configs/fev-v2-standard.yaml` and `scripts/fev_report.py`. This prompt authorizes the phase-5
prerequisites A–D (dependency lock, seasonal naive, interruption/resume trial, memory and timing
probes, walmart item-specific-origin startup) and, only
when they pass and are recorded, ONE frozen `first_window` v2 run with seed 0 on all seven weekly
tasks. It authorizes no tuning, no second seed, no configuration change and no variant selection.

---

You are the executing agent. Create, run and save a Jupyter notebook on my **hosted Colab
free-tier kernel** (T4), using the public repository `https://github.com/alessiosavi/demand-forecast`
at commit **`<SHA>`**. Follow `docs/protocol.md` and the notebook template in `docs/colab.md` at
that commit. Return measured results in the report format below, with a durable copy of the
notebook and artifacts on Google Drive. Never run anything from this prompt on my local machine,
never download Timestral weights, never edit `configs/fev-v2-standard.yaml`.

## 0. Rules that override convenience

- The frozen command is exactly
  `demand-forecast fev run --tasks <order from §6> --config configs/fev-v2-standard.yaml --policy first_window --out-dir "$FEV_RUN_DIR" --resume`.
  No `--model-name`, no config edit, no `--stop-after` in the frozen directory.
- **No v2 benchmark score is viewed before prerequisites A–D (§3–§5b) pass and are recorded in the notebook.**
  Before §7, `summaries.csv` may only be read programmatically to display rows filtered to
  `model_name == "seasonal_naive"`, row counts, `run_identity` values and file hashes; never
  print a row or metric column of the v2 model, and do not run `fev leaderboard` or
  `scripts/fev_report.py`. Inner-validation numbers in logs/`run.json`/`progress.json`
  (`best_score`, `inner_SQL`) are diagnostics and may be read.
- A `run identity mismatch` error means artifacts from a different code/config/data state:
  stop and report; never delete or move artifacts to force a resume.
- Commands that are EXPECTED to exit 1 (§4.1, §5, §5b) must not kill the cell: use
  `set +e; <cmd> 2>&1 | tee -a "$LOG"; rc=${PIPESTATUS[0]}; set -e; test "$rc" -eq 1`.
- Long commands run detached from a Python cell with the template's launcher (docs/colab.md §4):
  a bash script with `set -euo pipefail` and `trap 'echo $? > "$LOG.exit"' EXIT` whose body runs
  the command WITHOUT `tee` (Popen already redirects: `subprocess.Popen(["bash", script],
  stdout=open(LOG, "a"), stderr=subprocess.STDOUT, start_new_session=True)`); add a fresh UTC
  launch timestamp to the log/script/`.exit` names, keeping the fresh-file assert so a relaunch
  in the same runtime preserves earlier evidence. Isolate both the launcher and the poller
  with `start_new_session=True` from the kernel's process-group interrupt/restart signals.
  For a deliberate training stop, use `os.killpg(process.pid, signal.SIGINT)` after importing
  `signal`. Poll the log/`.exit` paths independently of the Python handle, including after a
  kernel restart; use short cells (`tail -n 5 "$LOG"`
  every few minutes; polling also keeps the runtime alive). `.exit` holds the raw command status
  (0 success; 1 for an expected interruption, asserted afterwards by the notebook); an absent
  `.exit` after a VM reset means "unknown/interrupted", not success. Colab has no background
  cells and `%%bash` output appears only when the cell finishes.
- Record everything you measure, including failures; label anything unverified.

## 1. Runtime, Drive, paths and hardware

1. Runtime > Change runtime type > **T4 GPU**, Runtime version **2026.07** (Python 3.12). Ask
   the user to click Drive's mount consent popup when it appears (about two minutes before
   "mount failed").
2. Preflight cell (template §1): `assert sys.version_info[:2] == (3, 12)`; mount Drive; set in
   `os.environ`: `FEV_SHA=<SHA>`, `FEV_SESSION=<UTC YYYYMMDDTHHMMSSZ>`,
   `FEV_DRIVE_ROOT=/content/drive/MyDrive/demand-forecast`,
   `FEV_RUN_DIR=$FEV_DRIVE_ROOT/runs/01-fev-run-v2-<SHA>`, `TRIAL=$FEV_RUN_DIR/trial`,
   `HF_HOME=$FEV_DRIVE_ROOT/hf`, `HF_HUB_CACHE=$FEV_DRIVE_ROOT/hf/hub` (Drive, reused from
   prompt 00), `HF_DATASETS_CACHE=/content/hf-datasets` (local, regenerable).
   These are the default rev-2 paths; choose §6's rerun policy before running the notebook.
   For alternative (b), use the rev-1 run directory and follow §6's trial-path rule.
3. Hardware (no repository code yet): `nvidia-smi` name/VRAM, `torch.cuda.get_device_capability(0)`
   (expect `(7, 5)`), `/proc/meminfo` MemTotal and **MemAvailable**, whose kB values are KiB
   (expect 13,286,924 kB = 12.67 GiB and 9,834,636 kB = 9.38 GiB), plus Drive and `/content`
   free bytes. Compute `RSS_LIMIT` in kB as `0.8 × MemAvailable` (≈ 7.87 M kB ≈ 7.5 GiB)
   and compare directly with `/usr/bin/time -v`'s `Maximum resident set size (kbytes)`;
   no unit conversion. Other gates: `VRAM_LIMIT = 13 GiB`, Drive free ≥ 300 MB.
   Storage note: the model size depends on the
   fitted vocabularies (16 embedding weights per fitted series plus statics): m5 ≈ 0.8 M
   parameters → checkpoint ≈ 13 MB (×2 while replacing), bundle ≈ 3 MB; smaller tasks less;
   float32 npz per window ≈ 50 MB in total (hermes 19 MB, m5 14 MB, favorita 7 MB); measure the
   trial artifacts in §4 and record actual sizes in §7.
4. Start the VRAM poller from a Python cell and keep its handle. Give it a fresh UTC token
   `poller_launch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")`:
   `poller = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,memory.used", "--format=csv,noheader,nounits", "-lms", "1000"], stdout=open(f"/content/vram-{FEV_SESSION}-{poller_launch}.log", "w"), start_new_session=True)`.
   Use the template's end-of-session cell to stop the poller, wait for it, and copy the local
   log into `FEV_RUN_DIR`. A VM reset loses the uncopied local tail; direct Drive streaming
   also risks losing the unflushed tail, so only completed Drive copies are durable evidence.
   Its maximum is a 1-second sampled maximum; say so.

## 2. Install, dependency lock, dataset revision

1. Template §2: clone/checkout `<SHA>`, verify `git rev-parse HEAD`, GNU time guard, then:
   - **first session of this experiment:** `python -m pip install -e ".[bench]"`, then
     `pip freeze | grep -v '^-e ' > "$FEV_RUN_DIR/environment-lock.txt"` (the lock), plus
     `pip freeze > "$FEV_RUN_DIR/environment-$FEV_SESSION.txt"`;
   - **later sessions:** `python -m pip install -c "$FEV_RUN_DIR/environment-lock.txt" -e ".[bench]"`,
     then `pip freeze | grep -v '^-e ' | diff - "$FEV_RUN_DIR/environment-lock.txt"` must be
     empty; any difference means the environment drifted: stop and report.
   Expected: preinstalled numpy 2.0.2, pandas 2.2.2, pyarrow 18.1.0, datasets 4.0.0,
   torch 2.11.0+cu128 are kept. On a **fresh VM** always do ONE **Runtime > Restart session and
   run all** after the first editable install so the kernel loads its `.pth`/finder at interpreter
   start; keep the restart-check cell. On the rerun the second install must leave every version
   in place, with an empty freeze diff against the lock. `-c` constrains what pip installs; it
   does not recreate Colab's image, which is why the freeze diff is still required.
   Create `$FEV_RUN_DIR/provenance-$FEV_SHA/` before writing `commit.txt` and copying
   `docs/protocol.md`, `demand_forecast/bench/tasks/fev_bench_weekly.yaml` and
   `demand_forecast/bench/tasks/profiles.yaml` there, as in template §2; preserve earlier SHA folders.
2. `python scripts/pip_check_closure.py --extras bench --report "$FEV_RUN_DIR/pip-check-$FEV_SESSION.txt"`
   must exit 0 (complaints declared by packages outside the `demand-forecast[bench]` closure are
   recorded, not failures).
3. Metadata cell (template §2) with `scope` set to `"phase5_v2_first_window"`, and now the AMP
   check: `amp_dtype_for(torch.device("cuda"))` must print `torch.float16` on a T4.
4. Dataset pinning (online, by verification — offline mode does not work: `datasets` 4.0.0 in
   offline mode only reuses a prepared builder cache, and `/content/hf-datasets` is empty on
   every new runtime). Before ANY dataset load in EVERY session run a cell that reads
   `HfApi().dataset_info("autogluon/fev_datasets").sha` and `HfApi().list_repo_refs(...)`'s
   `main` target; in the first session record that commit as the experiment's revision in
   `$FEV_RUN_DIR/dataset-manifest.json`; in every later session both values must equal the
   recorded commit, otherwise stop and report before loading anything. After §3 (first session)
   extend the manifest with, per task, the cached parquet path under `HF_HUB_CACHE`, its sha256
   and size (prompt 00's check: cached sha256 == the LFS pointer at that commit); every later
   session re-verifies those sha256 values after its first load. Resume is additionally guarded
   by fev's dataset fingerprint inside the run identity (`run identity mismatch` stops a changed
   dataset). A revision different from prompt 00's `d3e7e720f4158eab03ebcb021f1cce779cd92acc`
   is allowed for a fresh experiment but makes §3's comparison informational only.

## 3. Prerequisite A — seasonal naive in this environment (no v2 score involved)

```bash
set -euo pipefail
cd /content/demand-forecast
/usr/bin/time -v demand-forecast fev baseline --tasks weekly --out-dir "$FEV_RUN_DIR" \
  2>&1 | tee -a "$FEV_RUN_DIR/baseline-$FEV_SESSION.log"
```

Seven `seasonal_naive` rows must exist in `$FEV_RUN_DIR/summaries.csv` (count rows with
`model_name == seasonal_naive` programmatically; do not display the v2 rows later written
there). Compare their SQL/MASE with prompt 00's rows
(`runs/00-profile-and-baselines-97f3ec7…/summaries.csv`): equal to six significant figures
when the dataset revision is the same; report differences with both values. A rerun prints
`Skipping completed task <name>` seven times and leaves the file unchanged (sha256).

## 4. Prerequisite B — interruption/resume trial (smallest task, trial directory)

`hierarchical_sales_1W` (118 series → 500 steps, evaluation every 100, 10 windows).
`C="demand-forecast fev run --tasks hierarchical_sales_1W --config configs/fev-v2-standard.yaml --policy first_window --out-dir $TRIAL"`.

Every command below runs as `/usr/bin/time -v <cmd>` so peak RSS is recorded.
1. Interrupt: `$C --stop-after 120` with the exit-1 pattern; the log must contain
   `Training interrupted at step 120`; `$TRIAL/hierarchical_sales_1W/checkpoints/latest.pt` and
   `checkpoints/progress.json` exist with `step == 120`, `max_steps == 500`, `eval_every == 100`,
   `timings.sessions == 1`, `timings.peak_cuda_mem_bytes > 0`, one telemetry record (step 100);
   no `bundle/`, no `window_*.npz`, no `run.json`.
2. Resume: `$C --resume` (exit 0). `checkpoints/progress.json` retains flat `max_steps == 500`
   and `eval_every == 100`. `run.json`: top-level `resumed_from_step == 120`;
   `resolved_budget == {"max_steps": 500, "eval_every": 100}`; `fit_result.steps == 500`,
   `fit_result.stop_reason == "max_steps"` (patience 5 cannot fire within 5 evaluations),
   `fit_result.best_step`, `fit_result.best_score` present; `timings.cumulative.sessions == 2`;
   `telemetry` has 5 records. `bundle/` and `window_0.npz` … `window_9.npz` exist, each NPZ
   carrying `inference_time_s`, `fallback_no_context`, `fallback_undefined_scale`, `device`.
3. Idempotence: `$C --resume` again (exit 0) prints `Skipping completed task hierarchical_sales_1W`;
   sha256 of `run.json`, `bundle/manifest.json`, every `window_*.npz` and `$TRIAL/summaries.csv`
   unchanged (the row is kept, not re-evaluated).
3b. Fresh-runtime rehearsal of the dataset rule: `rm -rf /content/hf-datasets`, rerun the §2.4
   verification cell, then `$C --resume` once more: it must rebuild the datasets cache online from
   the pinned Hub snapshot, re-verify the parquet sha256 values, and print the same skip line with
   all hashes unchanged.
4. Record from `run.json`/`progress.json`/the INFO log: cumulative `train_s`, `eval_s`,
   `checkpoint_s`, `train_s_per_step` per telemetry record, `panel:` build seconds per window,
   per-window `inference_time_s` and `device` (from the NPZ), peak RSS (`/usr/bin/time -v`), peak
   VRAM (poller maximum, labelled sampled, and `peak_cuda_mem_bytes`), artifact sizes (`du -sb`
   of checkpoints/, bundle/, window NPZ). Do not display v2 metric columns from
   `$TRIAL/summaries.csv`. Any failed expectation blocks §6.

## 5. Prerequisite C — memory and timing probes on the two 12,000-step tasks (trial directory)

For `m5_1W` and `hermes` (both resolve to 12,000 steps, evaluation every 200; hermes has
H = 52 and an expected 10,000 validation items), run `fev run --tasks <task> … --out-dir "$TRIAL" --stop-after 200`
with the exit-1 pattern, `/usr/bin/time -v` and the poller. The step-200 evaluation runs before
the interruption, so `progress.json` (`max_steps`, `eval_every`, `n_fit_ids`, `n_val_items`,
`timings`, `telemetry_tail` with `train_s_per_step`, `eval_s`, `peak_cuda_mem_bytes`) and the
INFO `eval:` line give the measurements; the `panel:` line gives the build seconds. Interrupted
probes have no `run.json`; read budgets from `progress.json`. Treat hermes' validation count as
unverified until `n_val_items` is read. Gates: peak RSS ≤ `RSS_LIMIT`, peak VRAM ≤ `VRAM_LIMIT`.
Projection per probed task = build + `max_steps × train_s_per_step` + `ceil(max_steps /
eval_every) × eval_s` + `ceil(max_steps / 200) × checkpoint_s` + inference (§4's per-item
inference seconds × items × windows), labelled as an optimizer-time extrapolation that omits
bundle serialisation and scoring.

Measured per-step time differs by task (rev 1 on a T4: 0.35 s hierarchical, 0.53 s hermes,
0.64 s m5), most plausibly with the per-sample feature assembly (covariate and static counts).
For unprobed tasks use the **slowest measured** `train_s_per_step`, scale `eval_s` by
`ceil(n_val_items / 512)` relative to the slowest probe, and label them "extrapolated".
Budgets come from `n_fit_ids` in `progress.json`/`run.json` when available, rather than treating
prompt 00 series counts as exact fit counts (rohlik keeps 3,942 of 5,243). For tasks without
trainer metadata at projection time (under (a): rossmann_1W, favorita_stores_1W, rohlik_sales_1W),
use prompt 00's `n_series` as an upper bound for `n_fit_ids` (the frozen budget function is
monotone in it) and `eligible_items_first_window` as the proxy for `n_val_items`, and label
those projections 'upper bound'. Where a task already ran under an identical identity (rev 1:
hierarchical_sales_1W, rossmann_1W, favorita_stores_1W), prefer its `run.json` values and cite
the rev-1 run directory. Apply the protocol's frozen budget function and, when trainer metadata
is available, verify against the flat `max_steps` and `eval_every` fields in `progress.json`.
Record the per-task and summed projections. Block §6 (status `blocked`) only on a memory gate failure or
when a single task projects above 6 hours (a preregistered operational limit, not a technical
one); otherwise plan sessions: each task resumes from its last checkpoint (every 200 steps), so
a long task may span sessions, and inference/evaluation stages are not checkpointed.

## 5b. Prerequisite D — walmart item-specific origins (trial directory)

Run this startup check at `<SHA>` with the exit-1 pattern, `/usr/bin/time -v` and the poller:

```bash
set -euo pipefail
cd /content/demand-forecast
LOG="$TRIAL/walmart-interrupt-$FEV_SESSION.log"
set +e
/usr/bin/time -v demand-forecast fev run --tasks walmart \
  --config configs/fev-v2-standard.yaml --policy first_window \
  --out-dir "$TRIAL" --stop-after 50 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
test "$rc" -eq 1
```

Expected in the log, in order:

```text
panel: task=walmart window=0 …
origins: task=walmart window=0 distinct=<n> earliest=<ts> latest=<ts>
identity: …
plan: …
fit: …
Training interrupted at step 50
```

Require `n > 1`; the diagnostic observed 11 origins, earliest 2011-10-14 and latest 2012-02-03.
Read `$TRIAL/walmart/checkpoints/progress.json`: require `progress.json["step"] == 50`
and record the flat `progress.json["max_steps"]` and `n_fit_ids`; require 4,404 steps only if
`n_fit_ids == 2936`, otherwise verify the protocol's frozen budget function for the observed
fit count. There must be no `run.json`, `bundle/` or `window_*.npz` under `$TRIAL/walmart/`.
D covers startup only (origins, plan, first steps); inference correctness for ragged origins
is covered by the repository's ragged-fixture tests at this SHA. Any failed expectation blocks §6.

## 6. Frozen run (only after §3–§5b passed and are recorded)

Rev 2 default (a): a **new** `FEV_RUN_DIR` under the rev-2 SHA with §3–§5b rerun. Alternative
(b), only if the user chose it in writing: `--resume` into the rev-1 run directory. (b) is valid
because the identity payload is unchanged for shared-origin windows and the skip path verifies
every artifact; conditions: keep the rev-1 `environment-lock.txt` and dataset manifest, state
whether A–C evidence is reused (allowed) or rerun (then use a fresh `$TRIAL-rev2` path), the
three completed tasks must print `Skipping completed task`, their `run.json` lacks
`earliest_origin` and their summaries rows show NaN in `max_distinct_origins` (expected;
`summaries.csv` is rewritten as later tasks complete, existing values preserved), and §7
records the producing SHA per task. For (b), record reused A–C evidence in this notebook,
run D at `<SHA>`, and retain the producing revision `d683bf7` for the three completed tasks.

Order by resolved steps, the two 12,000-step tasks last in the order of their §5 projection
(shorter first):
`hierarchical_sales_1W,rossmann_1W,favorita_stores_1W,walmart,rohlik_sales_1W,<hermes|m5_1W>,<m5_1W|hermes>`.
Launch detached with the template launcher (log `$FEV_RUN_DIR/fev-run-$FEV_SESSION-<launch>.log`, `.exit`
status file), with `/usr/bin/time -v` inside the script and the poller running; poll every few
minutes. After a disconnect: new session, §1–§2 (lock enforced, dataset revision and sha256
manifest verified), then the same command;
completed tasks print `Skipping completed task <name>`, an interrupted task continues from
`checkpoints/latest.pt`. Never reorder, drop or add tasks between sessions.

For a successful run, every task must show `fit_result.nonfinite_losses == 0` in
`run.json`. A `nonfinite_losses` stop (also published in `checkpoints/progress.json`)
or a positive count is a failed run, never a configuration question. Preserve the
checkpoint, logs and failure evidence; do not treat exit 1 from this failure as an
expected interruption or change the frozen configuration to continue.

## 7. Report (first look at any v2 score happens here)

```bash
set -euo pipefail
cd /content/demand-forecast
python scripts/fev_report.py --run-dir "$FEV_RUN_DIR" --tasks weekly --baseline seasonal_naive --out "$FEV_RUN_DIR/report-tables.md"
demand-forecast fev leaderboard "$FEV_RUN_DIR/summaries.csv" --baseline seasonal_naive | tee "$FEV_RUN_DIR/leaderboard.txt"
```

Require report exit 0 and verify that every task shows `fit_result.nonfinite_losses == 0`;
record the count per task. A `nonfinite_losses` stop or a positive count is a failed
run, never a configuration question. Preserve diagnostic tables from a nonzero report
exit and report the failure; `--allow-partial` cannot establish success.

Verify on Drive: `summaries.csv`, `leaderboard.txt`, `report-tables.md`, per-task `run.json`,
`checkpoints/progress.json`, `bundle/manifest.json`, `window_*.npz` (with diagnostics), all
logs and `.exit` files, `environment-lock.txt` + per-session freezes, `pip-check-*.txt`,
`run-*.json`, `dataset-manifest.json`, the trial directories (including D's interrupt log and
`walmart/checkpoints/progress.json`), VRAM logs, `provenance-$FEV_SHA/commit.txt`,
`provenance-$FEV_SHA/protocol.md`, `provenance-$FEV_SHA/fev_bench_weekly.yaml`,
`provenance-$FEV_SHA/profiles.yaml` for each rev-2 producing SHA (when resuming, also retain
the rev-1 `commit.txt` and protocol/task/profile snapshots at their original paths), the frozen
config copied as `fev-v2-standard.yaml`, and the executed notebook
exported from Colab. Write `report.md`
there and paste it back in this format:

```text
Scope/status: Prerequisites A–D each pass|fail; frozen run complete|partial|blocked; sessions used:
Code SHA, UTC sessions, Drive folder, notebook path:
Rerun policy: (a) fresh or (b) resume; written choice for (b); A–C evidence reused|rerun and paths:
Lock: environment-lock.txt created in session …; later sessions diff empty (yes|no); pip-check closure result:
Hardware: GPU, capability, amp dtype (expect float16), VRAM, MemTotal/MemAvailable, RSS_LIMIT used:
Dataset manifest (Hub commit, 7 parquet sha256; verified|unverified); verified before every load in every session (yes|no); rehearsal 4.3b passed (yes|no); equal to prompt 00 (yes|no):
Prerequisite A: 7 seasonal_naive rows; per-task SQL/MASE vs prompt 00 (equal|differences):
Prerequisite B: step 120 interrupt evidence; resumed_from_step; fit_result.steps/stop_reason; idempotent rerun hashes equal; timings; RSS/VRAM; artifact sizes:
Prerequisite C: per task build_s, train_s_per_step, eval_s, peak RSS/VRAM, projection; summed projection; gate decision:
Prerequisite D: distinct origins, earliest/latest origin, interrupt evidence; step, max_steps, n_fit_ids; no run.json/bundle/NPZ:
Per task: producing SHA; max_distinct_origins (n/a for rev-1 tasks):
Per task: fit_result.nonfinite_losses == 0 gate; stop_reason; any nonfinite_losses failure:
Frozen run: paste report-tables.md (per-task budgets/timings/memory, metrics incl. per-knot pinball and 80% coverage/width, v2/naive ratio or "not comparable", fallbacks and exclusions, fingerprints, sizes, completeness):
Information-set caveat: walmart's early-ending items are scored on periods other items' available past covers; this is the information set fev gives every model, not an out-of-time test for those items.
Leaderboard text:
Interruptions/reconnects, identity errors, tracebacks (verbatim or none):
Drive bytes used by the run folder and free space after; /content datasets cache size:
Deviations from this prompt and anything unverified:
```
