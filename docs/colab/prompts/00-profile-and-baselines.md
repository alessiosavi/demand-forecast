# Prompt 00 — profile the seven weekly tasks and run seasonal naive

Paste the following instructions into the Claude agent in VS Code with the Colab
extension. Fill `<SHA>` with a full public commit SHA containing Task 4. This prepares
the phase-5 real-data profiling/baseline session; it does not authorize a v2 run.

---

You are the executing agent. Create, run and save a Jupyter notebook on my **hosted
Colab free-tier kernel**, using the public repository
`https://github.com/alessiosavi/demand-forecast` at commit **`<SHA>`**. Follow
`docs/protocol.md` and the notebook cells in `docs/colab.md` at that commit. Return
measured results in the report format below, with a durable copy of the notebook and
artifacts on Google Drive. Never run the seven tasks on my local machine.

1. Confirm the notebook uses a hosted Colab kernel, Python ≥3.10, and `/content`.
   Preflight GPU model/VRAM with `nvidia-smi`, host RAM and available disk/Drive space.
   Target a free T4 if assigned; record the actual device. CPU Colab is acceptable
   for these profiling/baseline commands. Assume the session can disconnect at any
   point; do not require a paid tier or promise uninterrupted runtime.
2. Mount Drive at `/content/drive`. Use
   `/content/drive/MyDrive/demand-forecast/runs/00-profile-and-baselines-<SHA>` as
   `FEV_RUN_DIR`, and `/content/drive/MyDrive/demand-forecast` as `FEV_DRIVE_ROOT`.
   Set `FEV_SHA` to the filled-in SHA and `FEV_SESSION` to a UTC session timestamp.
   Set these in the notebook's `os.environ` so shell cells inherit them. Set
   `HF_HOME=$FEV_DRIVE_ROOT/hf`, `HF_HUB_CACHE=$HF_HOME/hub` and
   `HF_DATASETS_CACHE=$HF_HOME/datasets` before importing fev or datasets. Keep the
   cache on Drive across sessions. Use the default pinned task sources.
3. In `/content`, run `git clone https://github.com/alessiosavi/demand-forecast` if
   absent, enter `/content/demand-forecast`, then `git checkout <SHA>`. Verify the full
   `git rev-parse HEAD` equals the supplied SHA. Check for unexpected local changes
   before reusing a clone. Run `python -m pip install -e ".[bench]"`; retain the pinned
   fev/statsforecast versions and the numpy constraint. Save `pip freeze`, install
   logs, commit, protocol and the task/profile YAMLs to Drive. Restart the kernel and
   rerun setup if an already imported numerical package changed during installation.
   Use the template's post-install torch preflight and per-session metadata cell.
4. Check these actual interfaces before starting downloads:

```bash
python scripts/profile_fev.py --help
demand-forecast fev baseline --help
demand-forecast fev leaderboard --help
```

5. Run the following in separate shell notebook cells (add `%%bash` as the first
   notebook-cell line). They must execute on Colab. Keep `set -euo pipefail` so logging
   does not hide a command failure. Profiling loads all histories for diagnostics;
   those values must not guide model selection.

```bash
set -euo pipefail
cd /content/demand-forecast
export HF_HOME="$FEV_DRIVE_ROOT/hf"
/usr/bin/time -v python scripts/profile_fev.py --tasks weekly \
  --out "$FEV_RUN_DIR/profile.csv" --resume \
  2>&1 | tee -a "$FEV_RUN_DIR/profile-$FEV_SESSION.log"
```

```bash
set -euo pipefail
cd /content/demand-forecast
export HF_HOME="$FEV_DRIVE_ROOT/hf"
/usr/bin/time -v demand-forecast fev baseline --tasks weekly \
  --out-dir "$FEV_RUN_DIR" \
  2>&1 | tee -a "$FEV_RUN_DIR/baseline-$FEV_SESSION.log"
demand-forecast fev leaderboard "$FEV_RUN_DIR/summaries.csv" \
  --baseline seasonal_naive | tee "$FEV_RUN_DIR/leaderboard.txt"
```

The baseline command automatically skips completed `(task, model_name)` rows; it has
no `--resume` flag. The profiler explicitly requires `--resume` to preserve completed
rows. Both save after each task, so rerun these commands with the same paths after a
disconnect; only the unfinished task must start again. If a session is short or a
task fails, use one name at a time via `--tasks <name>` with the same output paths.
Preserve failure logs and continue independent tasks. Never silently change data,
definitions, horizons, metrics or roles to make a task succeed. Use a fresh run
directory if any input or code changes; task-level resume does not validate identities.

6. Cover all seven: `m5_1W`, `favorita_stores_1W`, `rossmann_1W`, `rohlik_sales_1W`,
   `walmart`, `hermes`, `hierarchical_sales_1W`. Validate that `profile.csv` has exactly
   those seven unique task rows and the columns documented in `docs/protocol.md`.
   Check that `summaries.csv` has one `seasonal_naive` row per task, the official
   metrics, forecast counts and dataset fingerprints. Report non-finite metrics or
   failures explicitly. Repeat the two work commands once after completion and verify
   CSV contents are unchanged and logs show saved tasks being skipped. Report this as
   task-level recovery evidence, not neural-checkpoint recovery.
7. Verify `profile.csv` and `summaries.csv` are readable in the Drive run folder.
   Copy back any locally written logs, metadata, timing/RSS evidence and leaderboard;
   export/save the executed `.ipynb` there too. Preserve the HF cache on Drive. Record
   dataset revisions from cache metadata if available, or label them unverified;
   keep fev's fingerprints from `summaries.csv`. Compare published baselines only if
   task definitions, fingerprints, forecast counts and implementation match, with a
   declared tolerance. Otherwise state that comparability is unverified.
8. Write `report.md` beside the CSVs and paste it back. If interrupted, provide partial
   results and the exact commands to resume; never mark missing tasks complete.

Report format:

```text
Status: complete / partial / failed; profiling + seasonal_naive only.
Full commit SHA; UTC session(s); Drive folder; saved notebook:
Python, torch, fev, datasets, statsforecast, numpy; environment file:
Hardware: GPU, VRAM, host RAM, CUDA and supported AMP dtype (CPU-only if applicable):
Commands, exit codes and logs; completed / failed / pending task names:
Profile table: task, n_series, length_min, length_median, length_max,
               anchor_weekday, nan_share, negative_share, zero_share,
               zero_scale_items, eligible_items_first_window, static_nan_share.
Baseline table: task, SQL, MASE, WAPE, WQL, num_forecasts,
                dataset_fingerprint, training_time_s, inference_time_s.
Rerun/recovery evidence; measured time and peak RSS/VRAM, or not measured:
Dataset revisions and published-row comparability: verified evidence or unverified:
Drive artifacts verified: profile.csv, summaries.csv, leaderboard, logs, metadata,
                         environment, protocol/task definitions, notebook, report.md.
Warnings/failures and exact next-session commands if anything remains:
```

Stop after profiling and seasonal-naive evaluation. Do not invoke the legacy training
path, tune models, select a lookback from benchmark scores or download Timestral
weights. `fev run --resume` belongs to a later implementation and a separate reviewed
phase-5 model prompt. No model bundle is expected from this prompt.
