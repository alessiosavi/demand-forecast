# EC2 frozen fev run — operator runbook (2026-10-31)

Claude executes this runbook **on EC2 after the PR merges**. Local work is limited to
synthetic fixtures. The binding [protocol](protocol.md), [performance plan, Task 5](superpowers/plans/2026-10-31-fev-performance-ec2.md)
and [prompt 01 §§2–7](colab/prompts/01-fev-run-v2.md) supply the gates below, in this order.
No tuning, budget change, second seed or variant selection is authorised. Quick scores
are pipeline checks only. Do not view frozen v2 scores before all gates pass.

Commands are saved Bash scripts (`#!/bin/bash`, `set -euo pipefail`) and Python files.
Run Bash scripts with `bash path/to/script.sh`; do not put a multiword command in a zsh
variable. All paths passed between machines are explicit. Values in `<…>` must be
filled from the operator's evidence, never guessed.

## 0. Gate 0

Record the passing prompt-02 Colab quick run, merged SHA, green CI, bot findings and
critical verification in the ledger. Owner agreement is required before merging.
Do not provision the experiment until these are recorded. Keep the Colab constraints
lock, prompt-00 baseline CSV and dataset manifest available for transfer.

## 1. Provision

Use `veliu-staging`, `eu-central-1`, **one on-demand g4dn.4xlarge** (one T4, 16 vCPU,
64 GiB), 150 GB gp3, `DeleteOnTermination=true`, shutdown behaviour **stop**. Budget
is $50; the twelve-hour guard limits one compute session, not total cost. Stopped
EBS volumes remain billable. Record instance, AMI, volume, key and security-group IDs.
Select a current **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu)**, x86_64,
with Python 3.12 available. Inspect the returned AMI name before launch.

Save and run this locally as `provision-ec2.sh` (AWS CLI and jq required):

```bash
#!/bin/bash
set -euo pipefail
export AWS_PROFILE=veliu-staging AWS_REGION=eu-central-1
: "${OPERATOR_IP:?your current public IPv4 address}"
: "${VPC_ID:?selected VPC}" "${SUBNET_ID:?public subnet in that VPC}"
TOKEN=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "ec2-$TOKEN"
cd "ec2-$TOKEN"
aws ec2 describe-images --owners amazon --filters \
  'Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*' \
  Name=architecture,Values=x86_64 Name=state,Values=available \
  --query 'reverse(sort_by(Images,&CreationDate))[:5].[ImageId,Name,CreationDate]' > amis.json
cat amis.json
# Select an inspected ID from amis.json; keep that file with the evidence.
: "${AMI_ID:?inspected AMI ID}"
KEY="fev-$TOKEN"
aws ec2 create-key-pair --key-name "$KEY" --query KeyMaterial --output text > "$KEY.pem"
chmod 600 "$KEY.pem"
SG=$(aws ec2 create-security-group --group-name "$KEY" --description 'fev SSH operator only' \
  --vpc-id "$VPC_ID" --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 \
  --cidr "$OPERATOR_IP/32"
ROOT_DEVICE=$(aws ec2 describe-images --image-ids "$AMI_ID" --query 'Images[0].RootDeviceName' --output text)
jq -n --arg dev "$ROOT_DEVICE" '[{DeviceName:$dev,Ebs:{VolumeSize:150,VolumeType:"gp3",DeleteOnTermination:true}}]' > disks.json
cat > guard.sh <<'BASH'
#!/bin/bash
shutdown -h +720
BASH
aws ec2 run-instances --image-id "$AMI_ID" --instance-type g4dn.4xlarge --count 1 \
  --key-name "$KEY" --security-group-ids "$SG" --subnet-id "$SUBNET_ID" \
  --associate-public-ip-address --instance-initiated-shutdown-behavior stop \
  --block-device-mappings file://disks.json --user-data file://guard.sh \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$KEY}]" > instance.json
INSTANCE=$(jq -r '.Instances[0].InstanceId' instance.json)
aws ec2 wait instance-running --instance-ids "$INSTANCE"
aws ec2 describe-instances --instance-ids "$INSTANCE" > running.json
jq '.Reservations[].Instances[] | {InstanceId,PublicIpAddress,BlockDeviceMappings}' running.json
printf '%s\n' "$KEY" > key-name.txt
printf '%s\n' "$SG" > security-group.txt
```

After every boot/restart, run `sudo shutdown -h +720` again and record
`sudo shutdown --show`. User data normally runs only on first boot. Check the guard
and remaining budget hourly. After a stop, absent exit files mean interrupted/unknown.
Preserve existing logs and checkpoints; verify the same SHA, lock, pins, T4 and fp16
before resuming. Never mix hardware/dtype or fallback code in an existing run directory.

## 2. Setup: code, lock, pins, sequential cache warm-up

Copy the Colab `environment-lock.txt`, original `dataset-manifest.json` and prompt-00
`summaries.csv` to `/home/ubuntu/fev-inputs/` using `scp -i KEY.pem … ubuntu@IP:…`.
The baseline comparison below requires prompt 00's revision
`d3e7e720f4158eab03ebcb021f1cce779cd92acc`; a different revision blocks this experiment's
six-significant-figure gate. Keep the original manifest as `colab-dataset-manifest.json`.

Save the following as `/home/ubuntu/setup-fev.sh` and run it on EC2. Set `FEV_SHA`
to the full merged SHA. A constraints file limits versions but does not recreate an
image: install its packages and **require an empty normalised freeze diff**. A lock
containing unavailable/local Colab packages is a setup failure to resolve, not permission
to silently discard pins. Only editable paths and comments are excluded from the diff.

```bash
#!/bin/bash
set -euo pipefail
: "${FEV_SHA:?full merged SHA}"
sudo apt-get update
sudo apt-get install -y python3.12-venv git time jq rsync util-linux
sudo shutdown -h +720
mkdir -p /home/ubuntu/fev
cd /home/ubuntu/fev
git clone https://github.com/alessiosavi/demand-forecast code
cd code
git checkout --detach "$FEV_SHA"
test "$(git rev-parse HEAD)" = "$FEV_SHA"
python3.12 -m venv /home/ubuntu/fev/venv-pr
export FEV_LOCK_FILE=/home/ubuntu/fev-inputs/environment-lock.txt
env -u LD_LIBRARY_PATH /home/ubuntu/fev/venv-pr/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r "$FEV_LOCK_FILE" -c "$FEV_LOCK_FILE" -e '.[bench]'
```

Create `/home/ubuntu/fev/env.sh` with this content, filling the SHA. Every subsequent
script starts by sourcing it. `TOOLS` stays at the merged code even during fallback;
`VENV` selects the actual CLI installation. The launcher resolves its provenance using
that interpreter in isolated mode, and runs from its installed package checkout.

```bash
#!/bin/bash
export FEV_SHA='<full merged SHA>'
export TOOLS=/home/ubuntu/fev/code
export VENV=/home/ubuntu/fev/venv-pr
export FEV_LOCK_FILE=/home/ubuntu/fev-inputs/environment-lock.txt
export FEV_DATASET_MANIFEST=/home/ubuntu/fev/dataset-manifest.json
export PROMPT00=/home/ubuntu/fev-inputs/summaries.csv
export HF_HOME=/home/ubuntu/fev/hf
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export UTC='<one UTC token for this gate sequence>'
export PROBES="/home/ubuntu/fev/runs/probes-$UTC"
export OPS=/home/ubuntu/fev/ops
export ALL_TASKS=m5_1W,favorita_stores_1W,rossmann_1W,rohlik_sales_1W,walmart,hermes,hierarchical_sales_1W
unset LD_LIBRARY_PATH HF_DATASETS_OFFLINE HF_HUB_OFFLINE FEV_DATASETS_PREFIX PYTHONPATH
```

Initialise the evidence directories and verify the environment in `verify-env.sh`:

```bash
#!/bin/bash
set -euo pipefail
source /home/ubuntu/fev/env.sh
mkdir -p "$OPS" "$PROBES" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
cd "$TOOLS"
env -u LD_LIBRARY_PATH "$VENV/bin/python" -m pip freeze > "$PROBES/pip-freeze.txt"
sed '/^#/d; /^-e /d; /^$/d' "$FEV_LOCK_FILE" | sort > "$PROBES/expected-pins.txt"
sed '/^#/d; /^-e /d; /^$/d' "$PROBES/pip-freeze.txt" | sort > "$PROBES/actual-pins.txt"
diff -u "$PROBES/expected-pins.txt" "$PROBES/actual-pins.txt" > "$PROBES/lock.diff"
env -u LD_LIBRARY_PATH "$VENV/bin/python" scripts/pip_check_closure.py \
  --extras bench --report "$PROBES/pip-check.txt"
env -u LD_LIBRARY_PATH "$VENV/bin/python" - <<'PY' > "$PROBES/hardware.txt"
import sys
import torch
from demand_forecast.core.training import amp_dtype_for
assert sys.version_info[:2] == (3, 12)
assert torch.cuda.get_device_capability(0) == (7, 5)
assert amp_dtype_for(torch.device('cuda')) == torch.float16
print(sys.executable, torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_properties(0))
PY
nvidia-smi -q > "$PROBES/nvidia-smi.txt"
free -b > "$PROBES/ram.txt"
df -B1 > "$PROBES/disk.txt"
sudo shutdown --show > "$PROBES/shutdown.txt"
```

Follow [Colab §2 and its online pin rule](colab.md#2-public-clone-pinned-checkout-and-installation):
check both Hub SHA and `main` **before any load in every session**, then verify cached
parquet bytes against the pinned LFS pointers. Keep both caches writable: datasets takes
file locks. Do not set offline mode or override the task dataset paths. Save the following
as `$OPS/pin-and-warm.py`; it warms only the seven selected tasks, sequentially, and
requires one identifiable parquet per task. An unexpected repository layout fails closed
for inspection; it does not download unrelated datasets.

```python
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from demand_forecast.bench.specs import dataset_fingerprint, load_task_specs

repo = "autogluon/fev_datasets"
revision = "d3e7e720f4158eab03ebcb021f1cce779cd92acc"
manifest_path = Path(os.environ["FEV_DATASET_MANIFEST"])
previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
api = HfApi()


def verify_head():
    info = api.dataset_info(repo, files_metadata=True)
    main = next(
        b.target_commit
        for b in api.list_repo_refs(repo, repo_type="dataset").branches
        if b.name == "main"
    )
    assert info.sha == main == revision, (info.sha, main, revision)
    return info


info = verify_head()
if previous:
    assert previous["revision"] == revision
entries = []
for spec in load_task_specs("weekly"):
    candidates = [
        s
        for s in info.siblings
        if s.rfilename.endswith(".parquet")
        and (
            Path(s.rfilename).stem == spec.dataset_config
            or spec.dataset_config in Path(s.rfilename).parts
        )
    ]
    assert len(candidates) == 1, (spec.name, [s.rfilename for s in candidates])
    sibling = candidates[0]
    assert sibling.lfs is not None
    task = spec.to_task()
    fingerprint = dataset_fingerprint(task)  # sequential warm-up, never training
    path = Path(hf_hub_download(repo, sibling.rfilename, repo_type="dataset", revision=revision))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == sibling.lfs.sha256
    assert path.stat().st_size == sibling.lfs.size
    entry = dict(
        task=spec.name,
        filename=sibling.rfilename,
        path=str(path),
        sha256=digest,
        size=path.stat().st_size,
        fingerprint=fingerprint,
    )
    if previous:
        old = next(x for x in previous["files"] if x["task"] == spec.name)
        assert all(entry[k] == old[k] for k in ("filename", "sha256", "size", "fingerprint"))
    entries.append(entry)
    print(json.dumps(entry), flush=True)
verify_head()
manifest_path.write_text(
    json.dumps(dict(repo=repo, revision=revision, files=entries), indent=2) + "\n"
)
```

Run `env -u LD_LIBRARY_PATH "$VENV/bin/python" "$OPS/pin-and-warm.py" > "$PROBES/pins.log" 2>&1`.
Preserve a copy of the resulting manifest under `$PROBES`. Before each later launch,
rerun this verifier; a moved upstream `main`, changed bytes or fingerprint blocks it.

Two reusable operator scripts keep launch/status handling consistent. Save `$OPS/wait.sh`:

```bash
#!/bin/bash
set -euo pipefail
root=$1 task=$2 expected=$3
while [[ ! -f $root/$task/exit ]]; do sleep 5; done
session=$(dirname "$(readlink -f "$root/$task/exit")")
while [[ ! -f $session/sampler.exit ]]; do sleep 1; done
test "$(cat "$root/$task/exit")" -eq "$expected"
test "$(cat "$session/sampler.exit")" -eq 0
```

Missing exits after a reset or killed process require investigation, not indefinite
waiting and never an assumed success. For deliberate interruption use `--stop-after`;
for an emergency use `kill -INT -- -PID` on the recorded session PID and retain the logs.
Save `$OPS/one-task.sh` (arguments: root, task, expected exit, optional stop-after):

```bash
#!/bin/bash
set -euo pipefail
source /home/ubuntu/fev/env.sh
root=$1 task=$2 expected=$3
extra=()
[[ $# -lt 4 ]] || extra=(--stop-after "$4")
env -u LD_LIBRARY_PATH "$VENV/bin/python" "$OPS/pin-and-warm.py" > "$PROBES/pins-$(date -u +%Y%m%dT%H%M%S).log" 2>&1
bash "$TOOLS/scripts/ec2/frozen_parallel.sh" --venv "$VENV" --run-root "$root" \
  --config "$TOOLS/configs/fev-v2-standard.yaml" --tasks "$task" "${extra[@]}"
bash "$OPS/wait.sh" "$root" "$task" "$expected"
```

The launcher runs a baseline then the model under `setsid nohup env -u LD_LIBRARY_PATH
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /usr/bin/time -v`, with one output directory per task.
It snapshots executing package SHA/interpreter, freeze, lock, config, definitions,
protocol, hardware, pins and exact commands **before launching**. `--resume` is always
present. `ROOT/task/run.log` and `ROOT/task/exit` point into retained `launches/<token>`;
`sampler.exit` must also be zero. Task locks prevent concurrent writers. A launcher exit
zero means dispatched, not that training passed. Reuse a directory only for the same
code/config/pins/hardware; use fresh roots for distinct gates and fallback.

## 3. Prerequisite A — baseline parity and idempotence

Save/run `gate-a.sh`. No v2 score is involved. This is the same check in fallback,
using its new `VENV` and `PROBES`.

```bash
#!/bin/bash
set -euo pipefail
source /home/ubuntu/fev/env.sh
cd "$TOOLS"
env -u LD_LIBRARY_PATH "$VENV/bin/python" "$OPS/pin-and-warm.py" > "$PROBES/a-pins.log" 2>&1
env -u LD_LIBRARY_PATH /usr/bin/time -v "$VENV/bin/demand-forecast" fev baseline \
  --tasks weekly --out-dir "$PROBES/A" > "$PROBES/a-first.log" 2>&1
sha256sum "$PROBES/A/summaries.csv" > "$PROBES/a-before.sha256"
env -u LD_LIBRARY_PATH /usr/bin/time -v "$VENV/bin/demand-forecast" fev baseline \
  --tasks weekly --out-dir "$PROBES/A" > "$PROBES/a-rerun.log" 2>&1
sha256sum -c "$PROBES/a-before.sha256"
test "$(grep -c 'Skipping completed task' "$PROBES/a-rerun.log")" -eq 7
env -u LD_LIBRARY_PATH "$VENV/bin/python" - <<'PY' > "$PROBES/a-comparison.txt"
import os
from pathlib import Path
import pandas as pd
from demand_forecast.bench.specs import load_task_specs
current = pd.read_csv(Path(os.environ['PROBES']) / 'A/summaries.csv', float_precision='round_trip')
reference = pd.read_csv(os.environ['PROMPT00'], float_precision='round_trip')
current = current.loc[current.model_name == 'seasonal_naive'].set_index('task')
reference = reference.loc[reference.model_name == 'seasonal_naive'].set_index('task')
assert len(current) == 7 and current.index.is_unique and reference.index.is_unique
for spec in load_task_specs('weekly'):
    for key in ('SQL', 'MASE', 'WAPE', 'WQL'):
        a, b = current.loc[spec.name, key], reference.loc[spec.name, key]
        print(spec.name, key, a, b)
        assert format(a, '.6g') == format(b, '.6g')
    for key in ('dataset_fingerprint', 'num_forecasts'):
        assert current.loc[spec.name, key] == reference.loc[spec.name, key]
PY
```

## 4. P1 — exact real-data parity, including D's origin facts

EC2 only, no training. Save/run `gate-p1.sh`:

```bash
#!/bin/bash
set -euo pipefail
source /home/ubuntu/fev/env.sh
cd "$TOOLS"
env -u LD_LIBRARY_PATH /usr/bin/time -v "$VENV/bin/python" scripts/ec2/check_real_parity.py \
  --tasks m5_1W,hermes,walmart --batches 200 --out "$PROBES/p1.json" > "$PROBES/p1.log" 2>&1
```

Require every check to pass: 200 seed-0 sampler batches per task, prepared validation
batches, all first-window inference batches, and exact GPU evaluation metrics (NaN equals
NaN). Walmart must report 11 origins, earliest 2011-10-14, latest 2012-02-03,
`train_upto=2011-04-29`, `val_origin=2011-05-06`. The probe builds each first-window panel
once and uses the frozen `2863356` reference oracle. Locally only `--self-test` is allowed.
A parity failure invokes the fallback procedure below; do not continue a failing PR path.

## 5. Prerequisite B — full interruption/resume trial

Fresh root `$PROBES/B`; task `hierarchical_sales_1W`. Run these commands from a saved
Bash script after sourcing `env.sh`:

```bash
#!/bin/bash
set -euo pipefail
source /home/ubuntu/fev/env.sh
bash "$OPS/one-task.sh" "$PROBES/B" hierarchical_sales_1W 1 120
grep 'Training interrupted at step 120' "$PROBES/B/hierarchical_sales_1W/run.log"
env -u LD_LIBRARY_PATH "$VENV/bin/python" - <<'PY'
import json, os
from pathlib import Path
import torch
p = Path(os.environ['PROBES']) / 'B/hierarchical_sales_1W/hierarchical_sales_1W'
s = torch.load(p / 'checkpoints/latest.pt', map_location='cpu', weights_only=False)
x = json.loads((p / 'checkpoints/progress.json').read_text())
assert s['step'] == x['step'] == 120
assert x['max_steps'] == 500 and x['eval_every'] == 100
assert x['timings']['sessions'] == 1 and x['timings']['peak_cuda_mem_bytes'] > 0
assert [r['step'] for r in s['telemetry']] == [100]
assert not (p / 'run.json').exists() and not (p / 'bundle').exists() and not list(p.glob('window_*.npz'))
PY
bash "$OPS/one-task.sh" "$PROBES/B" hierarchical_sales_1W 0
env -u LD_LIBRARY_PATH "$VENV/bin/python" - <<'PY' > "$PROBES/b-evidence.txt"
import json, os
from pathlib import Path
import numpy as np
p = Path(os.environ['PROBES']) / 'B/hierarchical_sales_1W/hierarchical_sales_1W'
x = json.loads((p / 'run.json').read_text())
progress = json.loads((p / 'checkpoints/progress.json').read_text())
assert progress['max_steps'] == 500 and progress['eval_every'] == 100
assert x['resumed_from_step'] == 120
assert x['resolved_budget'] == dict(max_steps=500, eval_every=100)
f = x['fit_result']
assert f['steps'] == 500 and f['stop_reason'] == 'max_steps'
assert isinstance(f['best_step'], int) and np.isfinite(f['best_score'])
assert x['timings']['cumulative']['sessions'] == 2 and len(x['telemetry']) == 5
assert (p / 'bundle/manifest.json').is_file()
assert len(list(p.glob('window_*.npz'))) == 10
for i in range(10):
    with np.load(p / f'window_{i}.npz', allow_pickle=False) as z:
        assert {'inference_time_s','fallback_no_context','fallback_undefined_scale','device'} <= set(z.files)
        assert all(np.isfinite(z[k]).all() for k in ('inference_time_s','fallback_no_context','fallback_undefined_scale'))
        assert str(z['device'].item()).startswith('cuda')
        print(i, {k: z[k].item() for k in ('inference_time_s','fallback_no_context','fallback_undefined_scale','device')})
print(x['timings'], x['telemetry'])
PY
cd "$PROBES/B/hierarchical_sales_1W"
find hierarchical_sales_1W -type f \( -name run.json -o -name manifest.json -o -name 'window_*.npz' \) -print0 \
  | sort -z | xargs -0 sha256sum > "$PROBES/b-before.sha256"
sha256sum summaries.csv >> "$PROBES/b-before.sha256"
bash "$OPS/one-task.sh" "$PROBES/B" hierarchical_sales_1W 0
grep 'Skipping completed task hierarchical_sales_1W' run.log
sha256sum -c "$PROBES/b-before.sha256"
# Fresh builder-cache rehearsal (preserve the old cache instead of deleting it).
mv "$HF_DATASETS_CACHE" "${HF_DATASETS_CACHE}-before-B-$UTC"
mkdir -p "$HF_DATASETS_CACHE"
bash "$OPS/one-task.sh" "$PROBES/B" hierarchical_sales_1W 0
grep 'Skipping completed task hierarchical_sales_1W' run.log
sha256sum -c "$PROBES/b-before.sha256"
du -sb hierarchical_sales_1W/checkpoints hierarchical_sales_1W/bundle hierarchical_sales_1W/window_*.npz \
  > "$PROBES/b-sizes.txt"
```

Preserve each session's `/usr/bin/time -v` RSS, sampled GPU maxima, `panel:` build
seconds, train/eval/checkpoint timings, per-window inference time/device and artifact
sizes. Do not print model summary scores. Any failed expectation blocks the PR launch.

## 6. Prerequisite D — fresh walmart startup

```bash
#!/bin/bash
set -euo pipefail
source /home/ubuntu/fev/env.sh
bash "$OPS/one-task.sh" "$PROBES/D" walmart 1 50
grep -E 'panel:|origins:|identity:|plan:|fit:|Training interrupted' "$PROBES/D/walmart/run.log" > "$PROBES/d-startup.txt"
env -u LD_LIBRARY_PATH "$VENV/bin/python" - <<'PY'
import json, os
from pathlib import Path
import torch
p = Path(os.environ['PROBES']) / 'D/walmart/walmart'
x = json.loads((p / 'checkpoints/progress.json').read_text())
s = torch.load(p / 'checkpoints/latest.pt', map_location='cpu', weights_only=False)
assert x['step'] == s['step'] == 50
assert (x['n_fit_ids'], x['n_val_items'], x['max_steps'], x['eval_every']) == (2936, 2864, 4404, 200)
assert x['amp_dtype'] == 'float16'
assert not (p / 'run.json').exists() and not (p / 'bundle').exists() and not list(p.glob('window_*.npz'))
log = (p.parent / 'run.log').read_text()
for text in ('distinct=11 earliest=2011-10-14', 'latest=2012-02-03',
             'train_upto=2011-04-29', 'val_origin=2011-05-06', 'Training interrupted at step 50'):
    assert text in log, text
print(x)
PY
```

Inspect the retained startup sequence in order. These exact plan counts, dates and dtype
are gates, including in fallback. D is startup evidence; P1 also covers ragged inference.

## 7. P3 — repeatability at fixed lock, threads and shapes

Create a **second checkout and venv at `2863356`**, outside the merged checkout. Keep
`TOOLS` pointing at the merged code. Install using the same lock and freeze-diff check
from setup. Run every torch command under `env -u LD_LIBRARY_PATH` and the same two
thread variables. Never reuse a PR trial directory for reference code.

```bash
#!/bin/bash
set -euo pipefail
source /home/ubuntu/fev/env.sh
cd /home/ubuntu/fev
git clone https://github.com/alessiosavi/demand-forecast reference
cd reference
git checkout --detach 2863356
python3.12 -m venv /home/ubuntu/fev/venv-reference
env -u LD_LIBRARY_PATH /home/ubuntu/fev/venv-reference/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r "$FEV_LOCK_FILE" -c "$FEV_LOCK_FILE" -e '.[bench]'
```

Save `$OPS/p3-run.sh`; it runs sequentially, PR once and reference twice. The selected
venv is passed directly to the launcher, so no current-directory import can select the
wrong package. No `--model-name` override is used for the standard config.

```bash
#!/bin/bash
set -euo pipefail
source /home/ubuntu/fev/env.sh
for leg in pr ref1 ref2; do
    selected="$VENV"
    [[ $leg == pr ]] || selected=/home/ubuntu/fev/venv-reference
    bash "$TOOLS/scripts/ec2/frozen_parallel.sh" --venv "$selected" --run-root "$PROBES/P3-$leg" \
      --config "$TOOLS/configs/fev-v2-standard.yaml" --tasks hermes --stop-after 200
    bash "$OPS/wait.sh" "$PROBES/P3-$leg" hermes 1
done
bash "$OPS/one-task.sh" "$PROBES/P3-m5" m5_1W 1 200
```

Save `$OPS/compare-p3.py`, run
`env -u LD_LIBRARY_PATH "$VENV/bin/python" "$OPS/compare-p3.py" > "$PROBES/p3-comparison.txt"`.
`progress.json` has timing telemetry but no SQL; take **full-precision** SQL from checkpoint
history, never the four-decimal INFO log. Compare `model`, not `best_state`.

```python
import os
from pathlib import Path
import torch

root = Path(os.environ["PROBES"])


def load(leg):
    p = root / f"P3-{leg}/hermes/hermes/checkpoints/latest.pt"
    x = torch.load(p, map_location="cpu", weights_only=False)
    assert x["step"] == 200
    sql = next(r["SQL"] for r in x["history"] if r["step"] == 200)
    return x["model"], sql


def equal(a, b):
    return (
        a[1] == b[1]
        and a[0].keys() == b[0].keys()
        and all(torch.equal(a[0][k], b[0][k]) for k in a[0])
    )


pr, r1, r2 = load("pr"), load("ref1"), load("ref2")
m5 = torch.load(
    root / "P3-m5/m5_1W/m5_1W/checkpoints/latest.pt", map_location="cpu", weights_only=False
)
m5_sql = next(r["SQL"] for r in m5["history"] if r["step"] == 200)
print("m5 corroboration only", repr(m5_sql), "Colab", 0.8674, "delta", m5_sql - 0.8674)
print("SQL PR/ref1/ref2", repr(pr[1]), repr(r1[1]), repr(r2[1]))
print("model+SQL bitwise reference pair", equal(r1, r2), "PR/ref1", equal(pr, r1))
if equal(r1, r2):
    assert equal(pr, r1), "repeatable reference differs: fallback"
else:
    limit = 3 * abs(r1[1] - r2[1])
    print("SQL delta", abs(pr[1] - r1[1]), "limit", limit)
    print("SQL envelope passed", abs(pr[1] - r1[1]) <= limit)
    raise SystemExit("Reference variability: run ref3, investigate and record cause before launch")
```

If references disagree, run a third reference before assigning a cause:

```bash
#!/bin/bash
set -euo pipefail
source /home/ubuntu/fev/env.sh
bash "$TOOLS/scripts/ec2/frozen_parallel.sh" --venv /home/ubuntu/fev/venv-reference \
  --run-root "$PROBES/P3-ref3" --config "$TOOLS/configs/fev-v2-standard.yaml" \
  --tasks hermes --stop-after 200
bash "$OPS/wait.sh" "$PROBES/P3-ref3" hermes 1
```

Compare its full model tensors and SQL with both retained reference states using
`load("ref3")` and `equal` from the comparison script, and record a causal assessment.
The PR must meet `abs(SQL_PR-SQL_ref1) <= 3*abs(SQL_ref1-SQL_ref2)` as well as clearing
that investigation; the third reference does not redefine the envelope. Passing the numeric envelope alone does not clear
unexplained variability. Do not change deterministic-algorithm settings. A repeatable
reference with a PR mismatch selects fallback. m5's full-precision inner SQL at step 200
is compared with Colab's **0.8674 only as corroboration**, never a gate; retain it without
viewing benchmark scores.

## 8. P4 — seven-way quick dress rehearsal and capacity

Use a fresh root and the quick config; all seven tasks run concurrently. The launcher
labels summaries `demand-forecast-v2-quick`. This exercises training batch 128, inference
batch 512, all windows' panel builds, baseline, merge and report. Quick scores **never**
inform a model/configuration decision.

```bash
#!/bin/bash
set -euo pipefail
source /home/ubuntu/fev/env.sh
QUICK="$PROBES/P4"
bash "$TOOLS/scripts/ec2/frozen_parallel.sh" --venv "$VENV" --run-root "$QUICK" \
  --config "$TOOLS/configs/fev-v2-quick.yaml" --tasks "$ALL_TASKS"
IFS=, read -r -a task_names <<< "$ALL_TASKS"
for task in "${task_names[@]}"; do bash "$OPS/wait.sh" "$QUICK" "$task" 0; done
env -u LD_LIBRARY_PATH "$VENV/bin/python" "$TOOLS/scripts/fev_merge_runs.py" --run-root "$QUICK" --tasks weekly
env -u LD_LIBRARY_PATH "$VENV/bin/python" "$TOOLS/scripts/fev_report.py" --run-dir "$QUICK/merged" \
  --tasks weekly --baseline seasonal_naive --model-name demand-forecast-v2-quick \
  --out "$QUICK/merged/report-tables.md"
```

Record **sampled maxima**, not true peaks: `launches/*/resources.csv` has timestamp, PID,
RSS KiB and per-PID GPU MiB; `device.csv` records device-wide memory every second.
Require `sampler.exit=0` and non-empty numeric measurements for each task. A failed sampler
invalidates capacity evidence even when model exit is zero. `/usr/bin/time -v` also records
unsampled process maximum RSS; trainer `peak_cuda_mem_bytes` measures allocated tensors,
not CUDA context/allocator overhead. Keep these quantities separate.

For each task, take its maximum simultaneous summed session RSS and maximum per-process
GPU memory across **all** samples/windows. Conservatively sum the per-PID maxima if a task
has multiple CUDA PIDs. Sum task maxima regardless of their temporal overlap: contexts and
allocators are per process. Add a **10% margin**. For host RAM, replace quick validation
cache bytes by `nbytes_per_item × frozen_n_val_items`, obtained from `validation prepared`
and the full eligibility count in quick `run.json` (`n_val_items_eligible`). Keep the old
cache count and the substituted count in the ledger. Batch shapes and history sizes can
vary: if average quick bytes/item does not bound the full set, use the conservative bound
below, or measure the full cache on EC2. Never pass on an unverified optimistic estimate.

Save `$OPS/capacity.py` and run under the PR tools venv; it reports a conservative host
bound using a full context and maximum past length for every validation item. The bound
includes all Batch fields, float64 truth/past, and the quick payload replacement. This is
more conservative than scaling a random subsample's mean bytes/item on ragged histories.

```python
import json, os, re
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(os.environ["PROBES"]) / "P4"
tot_gpu = tot_rss = 0
for task in os.environ["ALL_TASKS"].split(","):
    base = root / task
    reports = list((base / "launches").glob("*/resources.csv"))
    assert reports
    samples = pd.concat([pd.read_csv(p) for p in reports])
    assert len(samples) and np.isfinite(samples[["rss_KiB", "gpu_MiB"]]).all().all()
    assert all((p.parent / "sampler.exit").read_text().strip() == "0" for p in reports)
    gpu = samples.groupby("pid").gpu_MiB.max().sum() * 2**20
    rss = samples.groupby("timestamp").rss_KiB.sum().max() * 2**10
    assert gpu > 0 and rss > 0, f"{task}: no observed GPU/RSS footprint"
    run = json.loads((base / task / "run.json").read_text())
    logs = "\n".join(p.read_text() for p in (base / "launches").glob("*/run.log"))
    matches = re.findall(r"validation prepared: items=(\d+) batches=\d+ bytes=(\d+)", logs)
    if matches:
        items, payload = map(int, matches[-1])
        eligible = run["n_val_items_eligible"]
        # Supply a verified upper bound per task, measured from the first-window
        # panel metadata. It must be >= every retained full-past history length.
        bounds = json.loads((Path(os.environ["PROBES"]) / "validation-bounds.json").read_text())
        b = bounds[task]
        h, c = run["settings"]["forecast"]["horizon"], run["settings"]["forecast"]["max_context"]
        enc, dec, num, cats = (b[k] for k in ("n_enc", "n_dec", "n_numeric", "n_categorical"))
        per_item = (
            6 * c
            + 5 * c * enc
            + 5 * h * dec
            + 5 * num
            + 8 * cats
            + 13 * h
            + 27
            + 8 * b["max_history"]
        )
        frozen_payload = max(per_item * eligible, payload / items * eligible)
        adjusted = rss - payload + frozen_payload
        print(
            task,
            "quick bytes/item",
            payload / items,
            "frozen items",
            eligible,
            "bounded frozen cache bytes",
            frozen_payload,
        )
    else:
        provenance = list((root / "provenance").glob("*/execution.json"))
        executing = [json.loads(p.read_text())["sha"] for p in provenance]
        assert executing
        assert (
            run["plan"]["exception"]
            or run["n_val_items"] == 0
            or all(sha.startswith("2863356") for sha in executing)
        )
        adjusted = rss
    print(task, "sampled maxima GPU bytes", gpu, "RSS bytes", rss, "adjusted RSS", adjusted)
    tot_gpu += gpu
    tot_rss += adjusted
print("with 10% margin", tot_gpu * 1.1, tot_rss * 1.1)
assert tot_gpu * 1.1 <= 14 * 2**30 and tot_rss * 1.1 <= 48 * 2**30, "split into smaller waves"
```

Populate `validation-bounds.json` on EC2 without training: for each first-window panel,
`max_history=max(s.history_length)`; `n_enc=calendar_features(first_grid, freq).shape[1]
+len(panel.roles.known_dynamic)+len(panel.roles.past_dynamic)` and
`n_dec=n_enc-len(panel.roles.past_dynamic)`; fit `StaticEncoder` on `plan.fit_ids`, then use
`encoder.n_numeric` and `len(encoder.vocab_sizes)`. Use the concrete helper below (first-window
panels built sequentially, disposed after each task); keep its output as gate evidence.

```python
# Save as $OPS/validation-bounds.py; EC2 only.
import json, os
from pathlib import Path
from demand_forecast.bench.specs import load_task_specs
from demand_forecast.bench.run import resolved_origin, window_origins
from demand_forecast.core.validation import plan_inner_validation
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.frequency import calendar_features, grid_range
from demand_forecast.data.loader_fev import panel_from_window

bounds = {}
for spec in load_task_specs("weekly"):
    task = spec.to_task()
    past, future = next(iter(task.iter_windows())).get_input_data()
    panel = panel_from_window(past, future, spec, task.freq)
    plan = plan_inner_validation(
        panel,
        resolved_origin(window_origins(future, spec.id_column, spec.timestamp_column)),
        spec.horizon,
    )
    enc = StaticEncoder().fit(panel, plan.fit_ids)
    grid = grid_range(panel.series[0].start, 1, panel.freq)
    calendar = calendar_features(grid, panel.freq).shape[1]
    bounds[spec.name] = dict(
        max_history=max(s.history_length for s in panel.series),
        n_enc=calendar + len(panel.roles.known_dynamic) + len(panel.roles.past_dynamic),
        n_dec=calendar + len(panel.roles.known_dynamic),
        n_numeric=enc.n_numeric,
        n_categorical=len(enc.vocab_sizes),
    )
    del panel, task, past, future, enc
(Path(os.environ["PROBES"]) / "validation-bounds.json").write_text(json.dumps(bounds, indent=2))
```

Run `env -u LD_LIBRARY_PATH "$VENV/bin/python" "$OPS/validation-bounds.py"` and then
`env -u LD_LIBRARY_PATH "$VENV/bin/python" "$OPS/capacity.py" > "$PROBES/capacity.txt"`.
Verify the byte formula against one observed prepared cache before accepting the bound.
Inference already exercises the one-512-item-batch GPU validation footprint. If the bound
exceeds **14 GiB VRAM or 48 GiB RSS**, split into waves whose own margin-adjusted sums fit;
record task membership and rerun the rehearsal for those waves. Keep reducing concurrency.
If even one task does not fit, block. No measured speedup is assumed before this rehearsal.

## 9. Failure handling — fallback

If a PR parity gate fails, use `/home/ubuntu/fev/venv-reference` at `2863356` and fresh
probe/run roots. Preserve the failed artifacts and reason. Change `VENV` and `PROBES`
in `env.sh` to the reference venv and a new `probes-fallback-<UTC>` sibling; keep `TOOLS`
at the merged SHA. Repeat **setup verification and pin warm-up, A, B, D and P4**, in
that order. P1's optimised-path oracle is not a prerequisite for baseline code. Record
executing SHA and interpreter from each launch, and `scaler_skips=unknown`, never zero.

The reference code has no prepared cache: P4 has no `validation prepared` log. Use measured
RSS including its one-batch validation work (no retained-cache substitution), preserve
full validation-item counts when available, and apply the same 10% margin and limits.
Do not run `validation-bounds.py` for a missing cache. Report tools run from the merged
checkout/venv, while the launcher takes the reference venv's `demand-forecast` executable.
An unexplained reference variability or a failed reference prerequisite still blocks launch.

## 10. Launch

Only after the gate ledger records acceptance, select a fresh root
`/home/ubuntu/fev/runs/ec2-frozen-<executing-sha>-<UTC>`, a sibling of the probes.
Check the twelve-hour guard and budget. Run the same online pin verification again.
Save the launch script with the accepted all-seven selection, or the recorded waves:

```bash
#!/bin/bash
set -euo pipefail
source /home/ubuntu/fev/env.sh
: "${ROOT:?fresh frozen root containing executing SHA and UTC}"
env -u LD_LIBRARY_PATH "$VENV/bin/python" "$OPS/pin-and-warm.py" > "$PROBES/launch-pins.log" 2>&1
sudo shutdown --show
bash "$TOOLS/scripts/ec2/frozen_parallel.sh" --venv "$VENV" --run-root "$ROOT" \
  --config "$TOOLS/configs/fev-v2-standard.yaml" --tasks "$ALL_TASKS"
```

For waves replace `--tasks "$ALL_TASKS"` with the approved comma-separated subset,
wait for every task in that wave, then launch the next subset into the **same root**.
No `--stop-after`, config change or model-name override in the frozen run. Poll task exits
and logs; check cost/guard hourly. A disconnect does not stop the detached tasks. After a
VM reset, recheck environment/pins/hardware and resume with the same root and config;
the launcher retains earlier sessions. Never delete artifacts to force identity acceptance.

## 11. Report

Every selected task exit must be zero, and all pollers must have valid evidence. Then
run the strict merge (no `--allow-partial`) and report using the merged tools venv:

```bash
#!/bin/bash
set -euo pipefail
source /home/ubuntu/fev/env.sh
: "${ROOT:?completed frozen run root}"
REPORT_PY=/home/ubuntu/fev/venv-pr/bin/python
REPORT_CLI=/home/ubuntu/fev/venv-pr/bin/demand-forecast
IFS=, read -r -a task_names <<< "$ALL_TASKS"
for task in "${task_names[@]}"; do bash "$OPS/wait.sh" "$ROOT" "$task" 0; done
env -u LD_LIBRARY_PATH "$REPORT_PY" "$TOOLS/scripts/fev_merge_runs.py" --run-root "$ROOT" --tasks weekly
env -u LD_LIBRARY_PATH "$REPORT_PY" "$TOOLS/scripts/fev_report.py" --run-dir "$ROOT/merged" \
  --tasks weekly --baseline seasonal_naive --out "$ROOT/merged/report-tables.md"
env -u LD_LIBRARY_PATH "$REPORT_CLI" fev leaderboard "$ROOT/merged/summaries.csv" \
  --baseline seasonal_naive > "$ROOT/merged/leaderboard.txt"
```

Merge writes only `ROOT/merged`: an atomic summary plus relative links to source artifact
directories. It refuses foreign/duplicate rows, missing tasks and a non-empty output;
exit 0 means success, 2 refusal. It does not certify model completeness; the report must
exit 0 without issues. Preserve a prior merged directory if regenerating after resume;
never overwrite evidence silently. Use `--allow-partial` only for explicitly labelled
partial troubleshooting, never final acceptance.

Record hardware/dtype, executing SHA per task, dependency/pin evidence, `scaler_skips`,
`steps/max_steps/best_step/stop_reason`, timings, maximum RSS, GPU sampled maxima, full
report/leaderboard, forecast counts, fingerprints, fallbacks/exclusions and artifact sizes.
State all failed/retried gates, fallback selection, interruptions and unverified quantities.
Walmart's early-ending items are scored on periods covered by other items' past; this is
fev's information set, not an out-of-time test for those items. Keep the unchanged stop
rule: phase-4 tuning/budget studies require preregistration and inner-validation evidence.

## 12. Copy back, verify, tear down

On EC2 create a manifest relative to the common `runs/` parent, covering ROOT and every
probe root used (including failed PR/fallback gates). Relative merge links survive
`rsync -a`; also record and compare their targets. Save this as a Bash script and pass
run/probe directory **basenames** as arguments:

```bash
#!/bin/bash
set -euo pipefail
cd /home/ubuntu/fev/runs
find "$@" -type f -print0 | sort -z | xargs -0 sha256sum > copyback.sha256
find "$@" -type l -printf '%p -> %l\n' | sort > copyback.links.txt
du -sb "$@" > copyback.bytes.txt
```

Copy the run/probe roots, manifests, operator scripts, `env.sh`, setup logs, Colab inputs
and provisioning JSON to the Mac. Example local Bash script (GNU `sha256sum` or the shown
Python verifier works on macOS):

```bash
#!/bin/bash
set -euo pipefail
: "${KEY:?SSH key}" "${HOST:?public IP}" "${DEST:?local evidence directory}"
mkdir -p "$DEST"
rsync -a -e "ssh -i $KEY" "ubuntu@$HOST:/home/ubuntu/fev/runs/" "$DEST/runs/"
rsync -a -e "ssh -i $KEY" "ubuntu@$HOST:/home/ubuntu/fev/ops/" "$DEST/ops/"
scp -i "$KEY" "ubuntu@$HOST:/home/ubuntu/fev/env.sh" "$DEST/"
cd "$DEST/runs"
python3 - <<'PY'
import hashlib
from pathlib import Path
for line in Path('copyback.sha256').read_text().splitlines():
    digest, name = line.split('  ', 1)
    with Path(name).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert actual == digest, name
for line in Path('copyback.links.txt').read_text().splitlines():
    name, target = line.split(' -> ', 1)
    p = Path(name)
    assert p.is_symlink() and str(p.readlink()) == target and p.exists(), name
print('All copied bytes and relative symlinks verified')
PY
```

Only after successful copy-back verification terminate the recorded instance. Save the
AWS responses and verify the recorded volumes are gone; a stop alone is insufficient.
Delete only this experiment's resources, never unrelated instances/volumes/groups.

```bash
#!/bin/bash
set -euo pipefail
export AWS_PROFILE=veliu-staging AWS_REGION=eu-central-1
: "${INSTANCE:?recorded instance ID}" "${SG:?recorded security group}" "${KEY_NAME:?recorded key name}"
aws ec2 describe-instances --instance-ids "$INSTANCE" > before-termination.json
aws ec2 terminate-instances --instance-ids "$INSTANCE" > termination.json
aws ec2 wait instance-terminated --instance-ids "$INSTANCE"
aws ec2 describe-volumes --filters "Name=attachment.instance-id,Values=$INSTANCE" > attached-after.json
aws ec2 describe-volumes --filters Name=status,Values=available > available-volumes.json
# Check every recorded volume ID: it must be deleted. If an experiment volume
# remains available, delete that specific ID with aws ec2 delete-volume --volume-id ID.
aws ec2 describe-instances --filters 'Name=instance-type,Values=g*' \
  'Name=instance-state-name,Values=pending,running,stopping,stopped' > remaining-g-instances.json
aws ec2 delete-key-pair --key-name "$KEY_NAME"
aws ec2 delete-security-group --group-id "$SG"
```

Inspect `remaining-g-instances.json` and the volume inventory, reconcile against the
recorded IDs, and record final teardown and total measured artifact bytes/cost evidence
in the ledger. Do not claim zero orphan resources from an empty attachment query alone.
