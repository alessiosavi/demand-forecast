# Colab notebook template for the executing agent

Use this template when the user pastes a task prompt into VS Code's Colab extension
with a Claude agent. Create, execute and save a notebook on the **hosted Colab kernel**.
Real-data runs belong to phase 5; PR 0 validates the commands only with local fixtures.
Start with [prompt 00](colab/prompts/00-profile-and-baselines.md) and follow the
[frozen protocol](protocol.md). Record actual results, including failures and omissions.

Target the free tier, planning for a T4 and short, interruptible sessions. GPU type and
session duration are not guaranteed; measure the assigned hardware. The
[Colab FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html) describes these
limits and Drive behavior. Profiling and seasonal naive can run on a CPU Colab runtime;
future neural runs require a successful CUDA preflight. Do not silently execute a
notebook on the developer's Mac or request a paid accelerator.

## 1. Hosted-runtime preflight and durable paths

Replace `<SHA>` with the user-supplied full commit SHA containing this task. Keep that
SHA and the output directory across reconnects. This notebook cell establishes the
environment for later shell cells; do not rely on a shell `export` persisting across
separate notebook cells.

```python
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.colab import drive

assert Path("/content").is_dir(), "Select the hosted Colab kernel"
assert sys.version_info >= (3, 10)
pinned_sha = "<SHA>"
assert re.fullmatch(r"[0-9a-f]{40}", pinned_sha), "Fill in the immutable commit SHA"
drive.mount("/content/drive")
drive_root = Path("/content/drive/MyDrive/demand-forecast")
run_dir = drive_root / "runs" / f"00-profile-and-baselines-{pinned_sha}"
run_dir.mkdir(parents=True, exist_ok=True)
session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
os.environ.update(
    FEV_SHA=pinned_sha,
    FEV_DRIVE_ROOT=str(drive_root),
    FEV_RUN_DIR=str(run_dir),
    FEV_SESSION=session,
    HF_HOME=str(drive_root / "hf"),
    HF_HUB_CACHE=str(drive_root / "hf" / "hub"),
    HF_DATASETS_CACHE=str(drive_root / "hf" / "datasets"),
)
print({"python": sys.version, "executable": sys.executable, "run_dir": str(run_dir)})
gpu = subprocess.run(["bash", "-c", "nvidia-smi"], capture_output=True, text=True)
print(gpu.stdout or gpu.stderr)
(run_dir / f"gpu-{session}.txt").write_text(gpu.stdout + gpu.stderr)
subprocess.run(["free", "-h"], check=True)
subprocess.run(["df", "-h", "/content", str(drive_root)], check=True)
```

Check available Drive storage and host RAM before downloading. Record CPU-only status
explicitly when no GPU is assigned. Keep data and progress on Drive; `/content` is
ephemeral. Set `HF_HOME` **before** importing fev/datasets. It relocates both Hub and
datasets caches; see the [Hugging Face cache documentation](https://huggingface.co/docs/datasets/main/cache).
Do not override the task dataset source with `FEV_DATASETS_PREFIX` for this run.

## 2. Public clone, pinned checkout and installation

Run this as a shell notebook cell (`%%bash` on its first line). On a new runtime it
clones the public repository; on reconnect it checks the same commit. Do not reset
uncommitted changes in an existing clone: report an unexpected dirty checkout first.

```bash
set -euo pipefail
cd /content
if [ ! -d demand-forecast ]; then
  git clone https://github.com/alessiosavi/demand-forecast
fi
cd demand-forecast
test -z "$(git status --porcelain)"
git checkout "$FEV_SHA"
test "$(git rev-parse HEAD)" = "$FEV_SHA"
export HF_HOME="$FEV_DRIVE_ROOT/hf"
python -m pip install -e ".[bench]" 2>&1 | tee "$FEV_RUN_DIR/install-$FEV_SESSION.log"
python -m pip freeze > "$FEV_RUN_DIR/environment-$FEV_SESSION.txt"
git rev-parse HEAD > "$FEV_RUN_DIR/commit.txt"
cp docs/protocol.md demand_forecast/bench/tasks/fev_bench_weekly.yaml \
  demand_forecast/bench/tasks/profiles.yaml "$FEV_RUN_DIR/"
python scripts/profile_fev.py --help
demand-forecast fev baseline --help
demand-forecast fev leaderboard --help
```

The install retains `fev==0.10.0`, `datasets<5`, `statsforecast==2.0.1` and `numpy<2`
from the repository. Do not upgrade around conflicts or download Timestral weights.
If numpy/torch was imported before installation changed it, restart the kernel and
rerun setup cells before proceeding. A dependency conflict is a reported failure.
Save the resolved environment; a code SHA does not pin all transitive dependencies
or the remote dataset revision. Phase 5 must lock these before a frozen model run.

After installation, run and save this metadata cell. On reconnect compare it with the
previous session: a changed dataset, task definition or environment needs investigation
before reusing completed results.

```python
import importlib.metadata
import json
import platform

import torch

cuda = torch.cuda.is_available()
hardware = {
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if cuda else None,
    "vram_bytes": torch.cuda.get_device_properties(0).total_memory if cuda else 0,
    "amp_dtype": ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if cuda else None,
}
manifest = {
    "sha": os.environ["FEV_SHA"],
    "session": os.environ["FEV_SESSION"],
    "scope": "profiling_and_seasonal_naive_only",
    "protocol_seed": 0,
    "protocol_policy": "first_window",
    "hardware": hardware,
    "versions": {
        name: importlib.metadata.version(name)
        for name in ("demand-forecast", "fev", "datasets", "statsforecast", "numpy")
    },
}
run_dir = Path(os.environ["FEV_RUN_DIR"])
(run_dir / f"run-{os.environ['FEV_SESSION']}.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
```

## 3. Profile and baseline cells available in PR 0

Use separate shell cells, each with `%%bash` first. `/usr/bin/time -v` records wall time
and peak host RSS on the Linux runtime; preserve logs even on failure. The commands
write completed-task progress straight to Drive, avoiding dependence on a final copy.

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

`profile_fev.py --resume` skips saved task names. `fev baseline` resumes **automatically**
by `(task, model_name)` and has **no `--resume` option**. Both persist after each task;
an interruption during a task recomputes that task. Neither command resumes within a
task or verifies run identity. Rerun the same commands and paths after reconnecting.
Use a new run directory when code, task definitions, profiles or dataset contents change.

For a short session, replace `weekly` with a single task name or comma-separated names,
retaining the same output paths. Process all seven across sessions; record the pending
list. If a task fails, preserve its traceback, then continue independent tasks by name
and report the failed task. Do not count a partial seven-task run as complete.
Once complete, repeat both commands to verify they skip completed tasks and leave the
CSV contents unchanged. This checks task-level recovery only; it is not the future
trainer's exact interruption/resume acceptance test.

## 4. Future neural-run template (unavailable in PR 0)

Task 26 will add the following interface. This is **not an executable step for prompt
00**: PR 0 has no `fev run`, and its legacy `train` command is not the new trainer.
Only use this after phase 5 provides a reviewed config and the interruption trial,
dependency lock and memory checks pass:

```text
demand-forecast fev run --tasks weekly --config /path/to/reviewed-v2.yaml --policy first_window --out-dir /content/drive/MyDrive/demand-forecast/runs/<SHA>/v2 --resume
```

The reviewed config must freeze seed 0, the protocol's standard v2 architecture and
resolved budget. Put checkpoints, bundles and completed-window predictions on Drive.
Split tasks across sessions and use `--resume` after reconnects. Preserve identity
mismatch errors and investigate; never overwrite mismatched artifacts to force resume.

## 5. Copy-back checklist and paste-back report

Before ending each session, verify the following are actually readable on Drive:

- `profile.csv` and `summaries.csv`, with partial completion explicitly marked;
- `leaderboard.txt`, per-session command logs, tracebacks and timing/RSS measurements;
- `commit.txt`, frozen protocol, task YAML, role profiles, per-session `run-*.json`,
  resolved `environment-*.txt`, and dataset revisions/fingerprints when available;
- the executed `.ipynb`, exported/saved from VS Code into the same Drive run folder;
- for future training only: `checkpoints/`, `bundle/`, `window_*.npz`, `run.json`,
  configuration and any tuning study storage. Prompt 00 produces no model bundles.

If an artifact was written locally under `/content`, copy it to the run folder now and
verify the destination. The HF cache already lives on Drive and remains there for
resume; it is not a model artifact. `summaries.csv` records fev's dataset fingerprints.
Record the resolved dataset revision from cache metadata where possible; otherwise
label it unverified. Do not claim the dataset was pinned merely because code was pinned.

Write `report.md` on Drive and paste its text back using this format:

```text
Scope/status: profile + seasonal_naive only; complete / partial / failed.
Code SHA, UTC sessions, Drive folder, notebook path:
Python/dependency versions and environment file; GPU/VRAM/RAM, CUDA/dtype:
Exact commands and exit codes; tasks completed / failed / still pending:
Profile table: all 12 documented columns, one row per task.
Baseline table: task, SQL, MASE, WAPE, WQL, num_forecasts,
                dataset_fingerprint, training_time_s, inference_time_s.
Recovery evidence: CSVs before/after rerun; completed tasks skipped; interrupted task:
Peak host RSS / VRAM and timings: measured values, or explicitly not measured.
Dataset revisions: resolved values or unverified; published-score comparability:
Artifacts verified on Drive; exceptions, warnings, tracebacks and next commands:
No v2 benchmark, tuning or model-selection result is included.
```
