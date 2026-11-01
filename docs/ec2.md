# EC2 frozen fev run — operator runbook (2026-10-31)

Run this procedure on EC2 after the prerequisite changes have merged. Local work is limited to
synthetic fixtures. The binding [protocol](protocol.md)
and [prompt 01 §§2–7](colab/prompts/01-fev-run-v2.md) supply the gates below, in this order.
The NaN-fix reference **F** is `ba7fc31e569609d25cee013604f6dd299d0baa6e`.
F's original pre-rewrite SHA was `4e33a3e7d811b5fa9dcc1101a48f6c1fc454dfc7`;
historical evidence and Drive paths retain their original SHAs and names (see the
[history-rewrite mapping](protocol.md#history-rewrite-2026-11-01)).
No tuning, budget change, second seed or variant selection is authorised. Quick scores
are pipeline checks only. Do not view frozen v2 scores before all gates pass.

Commands are saved Bash scripts (`#!/bin/bash`, `set -euo pipefail`) and Python files.
Run Bash scripts with `bash path/to/script.sh`; do not put a multiword command in a zsh
variable. All paths passed between machines are explicit. Values in `<…>` must be
filled from the operator's evidence, never guessed.

## 0. Gate 0

Record in the ledger:
- the Colab prompt-02b rerun at F (T4, fp16) with `nonfinite_losses == 0` on walmart and
  rohlik_sales_1W, the two tasks that failed at `2863356`;
- the merged SHA incorporating F, green CI, bot findings and critical verification.

The other five tasks had no non-finite losses at `2863356`, and F only removes non-finite
values or shrinks covariate magnitudes. P4 (§8) then requires `nonfinite_losses == 0` on all
seven tasks at the exact SHA the frozen run executes. The older prompt-02 lock below is environment evidence;
it does not satisfy the clean quick-run gate. Owner agreement is required before merging.
Do not provision the experiment until these are recorded. The Colab session relays
these three original artifacts to the Mac before provisioning; record their full
Drive source paths in the ledger. Paths below are relative to
`/content/drive/MyDrive/demand-forecast/`:

| Input | Colab source | Filename in the Mac input directory |
| --- | --- | --- |
| Colab image lock | Prompt 02 at `2863356`: `runs/02-quick-v2-2863356212c68debda92923f7c746d288d20f4df/environment-lock.txt` | `environment-lock.txt` |
| Original dataset manifest | The same prompt-02 run directory: `dataset-manifest.json` | `colab-dataset-manifest.json` (rename only; preserve bytes) |
| Baseline CSV | Prompt 00: `runs/00-profile-and-baselines-97f3ec7…/summaries.csv` (resolve the full recorded directory) | `summaries.csv` |

Save/run this on the Mac after receiving the files. Keep the SHA-256 record with the
inputs; §2 copies them with `scp`, records their hashes again on EC2, and requires
all three hashes to match before setup.

```bash
#!/bin/bash
set -euo pipefail
: "${MAC_INPUTS:?absolute Mac directory containing the three relayed files}"
cd "$MAC_INPUTS"
shasum -a 256 environment-lock.txt colab-dataset-manifest.json summaries.csv > mac-inputs.sha256
cat mac-inputs.sha256
```

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
export AWS_PROFILE=veliu-staging AWS_REGION=eu-central-1 AWS_DEFAULT_OUTPUT=json
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

Copy Gate 0's three Colab inputs and Mac hash record to `/home/ubuntu/fev-inputs/`.
Save/run this transfer script on the Mac after provisioning:

```bash
#!/bin/bash
set -euo pipefail
: "${MAC_INPUTS:?absolute Mac input directory}" "${KEY:?absolute SSH key path}" "${HOST:?public IP}"
cd "$MAC_INPUTS"
shasum -a 256 --check mac-inputs.sha256
ssh -i "$KEY" "ubuntu@$HOST" 'mkdir -p /home/ubuntu/fev-inputs'
scp -i "$KEY" environment-lock.txt colab-dataset-manifest.json summaries.csv mac-inputs.sha256 \
  "ubuntu@$HOST:/home/ubuntu/fev-inputs/"
ssh -i "$KEY" "ubuntu@$HOST" 'bash -se' <<'BASH'
set -euo pipefail
cd /home/ubuntu/fev-inputs
sha256sum environment-lock.txt colab-dataset-manifest.json summaries.csv > ec2-inputs.sha256
cat ec2-inputs.sha256
diff -u mac-inputs.sha256 ec2-inputs.sha256
BASH
scp -i "$KEY" "ubuntu@$HOST:/home/ubuntu/fev-inputs/ec2-inputs.sha256" "$MAC_INPUTS/"
```

The hash diff must be empty; retain both records. A mismatch blocks setup.
The baseline comparison below requires prompt 00's revision
`d3e7e720f4158eab03ebcb021f1cce779cd92acc`; a different revision blocks this experiment's
six-significant-figure gate. Keep the original `colab-dataset-manifest.json` unchanged.

Save the following as `/home/ubuntu/setup-fev.sh` and run it on EC2 using the detached
outer-job recipe below. Set `FEV_SHA` to the full merged SHA. Generate
`INSTALL_CONSTRAINTS` once from the full Colab image freeze (`COLAB_LOCK`), removing
only lines containing ` @ ` and preserving the original. This filtered file is the
**verified install input of 2026-11-01**, reused in §7. Its six excluded packages are
`cucim-cu12`, `cuvs-cu12`, `en-core-web-sm`, `google-colab`, `jupyter-kernel-gateway`
and `nx-cugraph-cu12`; all were outside the recorded required closure. This is not a
claim that pip universally rejects direct references. Colab-only packages such as
tensorflow and jax need not be installed on EC2. The numerical-environment gate is
`closure_parity.py` exit 0 for every installed `demand-forecast[bench]` dependency,
excluding demand-forecast itself: normalised names and **exact version strings**,
including local tags. This includes torch `2.11.0+cu128`, its `nvidia-*` CUDA/cuDNN
wheels, numpy `2.0.2`, pandas `2.2.2`, pyarrow `18.1.0`, datasets `4.0.0`, fev,
statsforecast and all other packages in that closure. Missing pins, unavailable
closure wheels or unequal versions block setup; do not drop pins to get past the gate.
Packages outside the closure are reported for information only.

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
export COLAB_LOCK=/home/ubuntu/fev-inputs/environment-lock.txt
export INSTALL_CONSTRAINTS=/home/ubuntu/fev-inputs/install-constraints.txt
python3.12 - <<'PY'
import os
from pathlib import Path
source = Path(os.environ["COLAB_LOCK"])
target = Path(os.environ["INSTALL_CONSTRAINTS"])
with target.open("xb") as stream:
    stream.writelines(line for line in source.read_bytes().splitlines(keepends=True) if b" @ " not in line)
PY
env -u LD_LIBRARY_PATH /home/ubuntu/fev/venv-pr/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -c "$INSTALL_CONSTRAINTS" -e '.[bench]'
```

For long remote jobs (setup, gates, rehearsals, frozen launch and report), save
`/home/ubuntu/dispatch-job.sh` on EC2 **before setup**. It creates a unique directory
under `host-logs/`, saves the exact command, and uses `setsid -f` with stdin, stdout
and stderr redirected. The saved job's EXIT trap atomically publishes its exit code,
including when `set -e` stops on failure. Keep the inner
`scripts/ec2/frozen_parallel.sh` unchanged: its `$!` session ID feeds the resource sampler.

```bash
#!/bin/bash
set -euo pipefail
label=$1
shift
[[ $label =~ ^[a-zA-Z0-9_-]+$ && $# -gt 0 ]]
mkdir -p /home/ubuntu/fev/host-logs
job=$(mktemp -d "/home/ubuntu/fev/host-logs/$label.XXXXXXXX")
{
    printf '#!/bin/bash\nset -euo pipefail\n'
    printf 'status_file=%q\n' "$job/exit"
    cat <<'BASH'
: "${status_file:?absolute status path set by dispatcher}"
publish_exit() {
    local rc=$?
    printf '%s\n' "$rc" > "$status_file.tmp"
    mv "$status_file.tmp" "$status_file"
    exit "$rc"
}
trap publish_exit EXIT
BASH
    printf '%q ' "$@"
    printf '\n'
} > "$job/command.sh"
setsid -f bash "$job/command.sh" < /dev/null > "$job/run.log" 2>&1
printf '%s\n' "$job"
```

Save `/home/ubuntu/wait-job.sh` on EC2. A successful dispatch is **not completion**.
Poll the exact returned directory; only the expected status permits the next gate.
The optional timeout is seconds (default twelve hours; shorten to the remaining guard).
After interruption, absent status means **unknown**, including if the SSH dispatch
connection failed before returning a directory. Inspect retained `host-logs/` before
retrying; never infer success or automatically dispatch a duplicate job.

```bash
#!/bin/bash
set -euo pipefail
job=$1 expected=${2:-0} timeout=${3:-43200}
[[ $expected =~ ^[0-9]+$ && $timeout =~ ^[1-9][0-9]*$ ]]
deadline=$((SECONDS + timeout))
while (( SECONDS < deadline )); do
    if [[ -s $job/exit ]]; then
        actual=$(cat "$job/exit")
        [[ $actual == "$expected" ]] || { echo "FAILED: $job exit=$actual expected=$expected" >&2; exit 1; }
        exit 0
    fi
    sleep 1
done
echo "UNKNOWN: $job has no exit status before timeout; next gate blocked" >&2
exit 124
```

Save this local helper as `dispatch-and-wait.sh`. `PROVISIONING` is the absolute local
`ec2-<TOKEN>` directory from §1; keep each job-path record there. Save/copy the remote
scripts before calling it. For setup, invoke
`bash dispatch-and-wait.sh setup-fev env FEV_SHA="$FEV_SHA" bash /home/ubuntu/setup-fev.sh`.
Use the same pattern for the subsequent saved scripts; do not dispatch the next gate
until this helper returns zero. After a lost connection, use the saved job path with
`wait-job.sh` to poll again, without redispatching.

```bash
#!/bin/bash
set -euo pipefail
: "${KEY:?absolute SSH key path}" "${HOST:?public IP}" "${PROVISIONING:?local provisioning directory}"
record=$(mktemp "$PROVISIONING/outer-job.XXXXXXXX")
mv "$record" "$record.txt"
record="$record.txt"
printf -v remote '%q ' bash /home/ubuntu/dispatch-job.sh "$@"
ssh -n -i "$KEY" "ubuntu@$HOST" "$remote" > "$record"
job=$(cat "$record")
test -n "$job"
printf 'Dispatched %s; retained path: %s\n' "$job" "$record"
printf -v remote '%q ' bash /home/ubuntu/wait-job.sh "$job" 0 "${JOB_TIMEOUT_SECONDS:-43200}"
ssh -n -i "$KEY" "ubuntu@$HOST" "$remote"
```

Create `/home/ubuntu/fev/env.sh` with this content, filling the SHA. Every subsequent
gate script starts by sourcing it. `TOOLS` stays at the merged code even during fallback;
`VENV` selects the actual CLI installation. The launcher resolves its provenance using
that interpreter in isolated mode, and runs from its installed package checkout.
`COLAB_LOCK` remains the input for closure parity. After that gate passes,
`verify-env.sh` below creates `FEV_LOCK_FILE` from the PR venv's normalised freeze
at `/home/ubuntu/fev/ec2-environment-lock.txt`. This EC2 lock governs the launcher's
existing exact drift check, including packages outside the numerical closure.

```bash
#!/bin/bash
export FEV_SHA='<full merged SHA>'
export TOOLS=/home/ubuntu/fev/code
export VENV=/home/ubuntu/fev/venv-pr
export COLAB_LOCK=/home/ubuntu/fev-inputs/environment-lock.txt
export INSTALL_CONSTRAINTS=/home/ubuntu/fev-inputs/install-constraints.txt
export COLAB_MANIFEST=/home/ubuntu/fev-inputs/colab-dataset-manifest.json
export FEV_LOCK_FILE=/home/ubuntu/fev/ec2-environment-lock.txt
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

Initialise the evidence directories and verify the environment in `verify-env.sh`.
Run it initially with `VENV=/home/ubuntu/fev/venv-pr`; repeat checks on later sessions.
An existing EC2 lock is compared, never regenerated to accept drift. Normalisation
for this lock matches the launcher: remove editable lines, comments and blank lines,
then sort; all remaining freeze entries must match exactly.

```bash
#!/bin/bash
set -euo pipefail
source /home/ubuntu/fev/env.sh
mkdir -p "$OPS" "$PROBES" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
cd "$TOOLS"
env -u LD_LIBRARY_PATH "$VENV/bin/python" -I -m pip freeze > "$PROBES/pip-freeze.txt"
env -u LD_LIBRARY_PATH "$VENV/bin/python" -I "$TOOLS/scripts/ec2/closure_parity.py" \
  --colab-lock "$COLAB_LOCK" --freeze "$PROBES/pip-freeze.txt" --extras bench \
  --out "$PROBES/closure-parity.txt"
env -u LD_LIBRARY_PATH "$VENV/bin/python" scripts/pip_check_closure.py \
  --extras bench --report "$PROBES/pip-check.txt"
sed '/^#/d; /^-e /d; /^$/d' "$PROBES/pip-freeze.txt" | sort > "$PROBES/actual-pins.txt"
if [[ ! -e $FEV_LOCK_FILE ]]; then
    test "$VENV" = /home/ubuntu/fev/venv-pr
    cp "$PROBES/actual-pins.txt" "$FEV_LOCK_FILE"
fi
sed '/^#/d; /^-e /d; /^$/d' "$FEV_LOCK_FILE" | sort > "$PROBES/expected-pins.txt"
diff -u "$PROBES/expected-pins.txt" "$PROBES/actual-pins.txt" > "$PROBES/lock.diff"
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
# The redirect is intentionally unprivileged: PROBES belongs to ubuntu.
# shellcheck disable=SC2024
sudo shutdown --show > "$PROBES/shutdown.txt"
```

Follow [Colab §2 and its online pin rule](colab.md#2-public-clone-pinned-checkout-and-installation):
check both Hub SHA and `main` **before any load in every session**, then verify cached
parquet bytes against the pinned LFS pointers. Keep both caches writable: datasets takes
file locks. Do not set offline mode or override the task dataset paths. Save the following
as `$OPS/pin-and-warm.py`; it warms only the seven selected tasks, sequentially, and
requires one identifiable parquet per task. An unexpected repository layout fails closed
for inspection; it does not download unrelated datasets. The original Colab schema is
`tasks[task].files[filename]` with `bytes`; the generated EC2 schema is a `files` list
with `size`. Map the exact Colab filename to the Hub sibling, including nested paths
such as `m5/1W/train-00000-of-00001.parquet`.

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
colab = json.loads(Path(os.environ["COLAB_MANIFEST"]).read_text())
assert colab["repo"] == repo and colab["revision"] == revision
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
if previous is not None:
    assert previous["repo"] == repo and previous["revision"] == revision
entries = []
for spec in load_task_specs("weekly"):
    colab_files = colab["tasks"][spec.name]["files"]
    assert len(colab_files) == 1, (spec.name, list(colab_files))
    ((rfilename, colab_file),) = colab_files.items()
    assert rfilename.endswith(".parquet"), rfilename
    candidates = [s for s in info.siblings if s.rfilename == rfilename]
    assert len(candidates) == 1, (spec.name, [s.rfilename for s in candidates])
    sibling = candidates[0]
    assert sibling.lfs is not None
    assert sibling.lfs.sha256 == colab_file["sha256"], (spec.name, "Colab sha256 differs")
    assert sibling.lfs.size == colab_file["bytes"], (spec.name, "Colab size differs")
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
    if previous is not None:
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
root=$1 task=$2 expected=$3 timeout=${4:-43200}
[[ $expected =~ ^[0-9]+$ && $timeout =~ ^[1-9][0-9]*$ ]]
# Resolve the launch alias before the task exit exists; retain this session identity.
test -L "$root/$task/exit"
session=$(dirname "$(readlink -f "$root/$task/exit")")
deadline=$((SECONDS + timeout))
while (( SECONDS < deadline )); do
    sampler=
    if [[ -s $session/sampler.exit ]]; then
        sampler=$(cat "$session/sampler.exit")
        [[ $sampler == 0 ]] || { echo "FAILED: $session sampler=$sampler" >&2; exit 1; }
    fi
    if [[ -s $session/exit ]]; then
        actual=$(cat "$session/exit")
        [[ $actual == "$expected" ]] || { echo "FAILED: $session exit=$actual expected=$expected" >&2; exit 1; }
        [[ $sampler != 0 ]] || exit 0
    elif [[ $sampler == 0 ]]; then
        echo "UNKNOWN: sampler finished without task exit: $session" >&2
        exit 1
    fi
    sleep 1
done
echo "UNKNOWN: task/sampler completion timed out: $session; next step blocked" >&2
exit 124
```

The optional fourth argument bounds the wait in seconds (default twelve hours).
The sampler's own failure blocks immediately, even when the task session disappeared
without writing an exit. Timeout returns 124 and blocks the next step; missing exits
after a reset or killed process require investigation, never an assumed success.
For deliberate interruption use `--stop-after`;
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
env -u LD_LIBRARY_PATH "$VENV/bin/python" "$OPS/pin-and-warm.py" > "$PROBES/p1-pins.log" 2>&1
env -u LD_LIBRARY_PATH /usr/bin/time -v "$VENV/bin/python" scripts/ec2/check_real_parity.py \
  --tasks m5_1W,hermes,walmart,rohlik_sales_1W --batches 200 --out "$PROBES/p1.json" > "$PROBES/p1.log" 2>&1
```

Rohlik covers the floor/clamp guard; walmart covers the cold mask.
Historical 2026-11-01 measurement: four tasks sequentially, 36:10 elapsed, reported 3.39 GB
maximum RSS (retained `/usr/bin/time -v`: 3,390,296 KiB); this is not a future guarantee.

Require every check to pass: 200 seed-0 sampler batches per task, prepared validation
batches, all first-window inference batches, and exact GPU evaluation metrics (NaN equals
NaN). Walmart must report 11 origins, earliest 2011-10-14, latest 2012-02-03,
`train_upto=2011-04-29`, `val_origin=2011-05-06`. The probe builds each first-window panel
once and uses the frozen F (`ba7fc31e569609d25cee013604f6dd299d0baa6e`) reference oracle. Locally only `--self-test` is allowed.
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

Create a **second checkout and venv at F (`ba7fc31e569609d25cee013604f6dd299d0baa6e`)**, outside the merged checkout. Keep
`TOOLS` pointing at the merged code. Install with §2's `INSTALL_CONSTRAINTS`, then
run the merged checkout's parity helper with the **reference interpreter** so it
traverses the reference installation's closure. Require exit 0 and an empty exact
freeze diff against `FEV_LOCK_FILE`, the EC2 lock recorded from the PR venv. The
launcher repeats that EC2 drift check before each launch, ignoring the editable
checkout line. Run every torch command under `env -u LD_LIBRARY_PATH` and the same
two thread variables. Never reuse a PR trial directory for reference code.

```bash
#!/bin/bash
set -euo pipefail
source /home/ubuntu/fev/env.sh
cd /home/ubuntu/fev
git clone https://github.com/alessiosavi/demand-forecast reference
cd reference
git checkout --detach ba7fc31e569609d25cee013604f6dd299d0baa6e
test "$(git rev-parse HEAD)" = "ba7fc31e569609d25cee013604f6dd299d0baa6e"
python3.12 -m venv /home/ubuntu/fev/venv-reference
env -u LD_LIBRARY_PATH /home/ubuntu/fev/venv-reference/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -c "$INSTALL_CONSTRAINTS" -e '.[bench]'
env -u LD_LIBRARY_PATH /home/ubuntu/fev/venv-reference/bin/python -I -m pip freeze \
  > "$PROBES/reference-pip-freeze.txt"
env -u LD_LIBRARY_PATH /home/ubuntu/fev/venv-reference/bin/python -I \
  "$TOOLS/scripts/ec2/closure_parity.py" --colab-lock "$COLAB_LOCK" \
  --freeze "$PROBES/reference-pip-freeze.txt" --extras bench \
  --out "$PROBES/reference-closure-parity.txt"
env -u LD_LIBRARY_PATH /home/ubuntu/fev/venv-reference/bin/python \
  "$TOOLS/scripts/pip_check_closure.py" --extras bench --report "$PROBES/reference-pip-check.txt"
sed '/^#/d; /^-e /d; /^$/d' "$FEV_LOCK_FILE" | sort > "$PROBES/reference-expected-pins.txt"
sed '/^#/d; /^-e /d; /^$/d' "$PROBES/reference-pip-freeze.txt" | sort > "$PROBES/reference-actual-pins.txt"
diff -u "$PROBES/reference-expected-pins.txt" "$PROBES/reference-actual-pins.txt" \
  > "$PROBES/reference-lock.diff"
```

Save `$OPS/p3-run.sh`; it runs sequentially, PR once and F twice (`ref1` and `ref2`). The selected
venv is passed directly to the launcher, so no current-directory import can select the
wrong package. No `--model-name` override is used for the standard config.

```bash
#!/bin/bash
set -euo pipefail
source /home/ubuntu/fev/env.sh
for leg in pr ref1 ref2; do
    selected="$VENV"
    [[ $leg == pr ]] || selected=/home/ubuntu/fev/venv-reference
    env -u LD_LIBRARY_PATH "$selected/bin/python" "$OPS/pin-and-warm.py" > "$PROBES/p3-$leg-pins.log" 2>&1
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
env -u LD_LIBRARY_PATH /home/ubuntu/fev/venv-reference/bin/python "$OPS/pin-and-warm.py" \
  > "$PROBES/p3-ref3-pins.log" 2>&1
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
inform a model/configuration decision. Any task with `nonfinite_losses > 0` blocks
P4 acceptance; require the strict report to exit 0. Investigate `scaler_skips` beyond
the observed 0–2 start-of-run calibration before launch. The report uses F's training
count (`fit_result.nonfinite_losses`, or the cumulative count for older reports).

```bash
#!/bin/bash
set -euo pipefail
source /home/ubuntu/fev/env.sh
QUICK="$PROBES/P4"
test ! -e "$QUICK"
env -u LD_LIBRARY_PATH "$VENV/bin/python" "$OPS/pin-and-warm.py" > "$PROBES/p4-pins.log" 2>&1
bash "$TOOLS/scripts/ec2/frozen_parallel.sh" --venv "$VENV" --run-root "$QUICK" \
  --config "$TOOLS/configs/fev-v2-quick.yaml" --tasks "$ALL_TASKS"
IFS=, read -r -a task_names <<< "$ALL_TASKS"
for task in "${task_names[@]}"; do bash "$OPS/wait.sh" "$QUICK" "$task" 0; done
REPORT_PY=/home/ubuntu/fev/venv-pr/bin/python
env -u LD_LIBRARY_PATH "$REPORT_PY" "$TOOLS/scripts/fev_merge_runs.py" --run-root "$QUICK" --tasks weekly
env -u LD_LIBRARY_PATH "$REPORT_PY" "$TOOLS/scripts/fev_report.py" --run-dir "$QUICK/merged" \
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

Save `$OPS/capacity.py` and run under the PR tools venv with a rehearsal root and task
subset; reuse this exact check for each wave. It reports a conservative host
bound using a full context and maximum past length for every validation item. The bound
includes all Batch fields, float64 truth/past, and the quick payload replacement. This is
more conservative than scaling a random subsample's mean bytes/item on ragged histories.

```python
import argparse, json, os, re
from pathlib import Path
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--tasks", required=True)
parser.add_argument("--validation-bounds", type=Path, required=True)
args = parser.parse_args()
root = args.root
tasks = args.tasks.split(",")
assert (
    tasks
    and len(tasks) == len(set(tasks))
    and set(tasks) <= set(os.environ["ALL_TASKS"].split(","))
)
tot_gpu = tot_rss = 0
for task in tasks:
    base = root / task
    sessions = sorted(p for p in (base / "launches").iterdir() if p.is_dir())
    assert sessions, f"{task}: no sampler sessions"
    frames = []
    for session in sessions:
        assert (session / "sampler.exit").read_text().strip() == "0", f"{session}: sampler failed"
        frame = pd.read_csv(session / "resources.csv")
        values = frame[["timestamp", "pid", "rss_KiB", "gpu_MiB"]]
        assert len(frame) and np.isfinite(values).all().all(), f"{session}: empty/nonfinite sample"
        assert (values >= 0).all().all() and (frame.pid > 0).all()
        frame["session"] = session.name
        frames.append(frame)
    samples = pd.concat(frames)
    gpu = samples.groupby(["session", "pid"]).gpu_MiB.max().sum() * 2**20
    rss = samples.groupby(["session", "timestamp"]).rss_KiB.sum().max() * 2**10
    assert gpu > 0 and rss > 0, f"{task}: no observed GPU/RSS footprint"
    run = json.loads((base / task / "run.json").read_text())
    logs = "\n".join((p / "run.log").read_text() for p in sessions)
    matches = re.findall(r"validation prepared: items=(\d+) batches=\d+ bytes=(\d+)", logs)
    if matches:
        items, payload = map(int, matches[-1])
        eligible = run["n_val_items_eligible"]
        assert isinstance(eligible, int) and eligible >= items > 0 and payload > 0
        # Supply a verified upper bound per task, measured from the first-window
        # panel metadata. It must be >= every retained full-past history length.
        bounds = json.loads(args.validation_bounds.read_text())
        b = bounds[task]
        h, c = run["settings"]["forecast"]["horizon"], run["settings"]["forecast"]["max_context"]
        enc, dec, num, cats = (b[k] for k in ("n_enc", "n_dec", "n_numeric", "n_categorical"))
        assert all(
            isinstance(v, int) and v >= 0 for v in (h, c, enc, dec, num, cats, b["max_history"])
        )
        assert h > 0 and c > 0
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
            "quick items",
            items,
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
            or all(sha == "ba7fc31e569609d25cee013604f6dd299d0baa6e" for sha in executing)
        )
        adjusted = rss
    assert np.isfinite(adjusted) and adjusted > 0
    print(task, "sampled maxima GPU bytes", gpu, "RSS bytes", rss, "adjusted RSS", adjusted)
    tot_gpu += gpu
    tot_rss += adjusted
print("with 10% margin: GPU GiB / adjusted RAM GiB", tot_gpu * 1.1 / 2**30, tot_rss * 1.1 / 2**30)
assert tot_gpu * 1.1 <= 14 * 2**30, "GPU bound exceeded: split into smaller waves"
assert tot_rss * 1.1 <= 48 * 2**30, "adjusted RAM bound exceeded: split into smaller waves"
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

Before the panel loads, rerun §2's pin verifier with a retained
`$PROBES/validation-bounds-pins.log`, then run
`env -u LD_LIBRARY_PATH "$VENV/bin/python" "$OPS/validation-bounds.py"` (skip this
helper for fallback's missing cache). Then run the capacity check:

```bash
#!/bin/bash
set -euo pipefail
source /home/ubuntu/fev/env.sh
env -u LD_LIBRARY_PATH /home/ubuntu/fev/venv-pr/bin/python "$OPS/capacity.py" \
  --root "$PROBES/P4" --tasks "$ALL_TASKS" --validation-bounds "$PROBES/validation-bounds.json" \
  > "$PROBES/capacity.txt" 2>&1
```

Verify the byte formula against one observed prepared cache before accepting the bound.
Inference already exercises the one-512-item-batch GPU validation footprint. If the bound
exceeds **14 GiB VRAM or 48 GiB RSS**, split into waves whose own margin-adjusted sums fit;
record task membership and rerun the rehearsal for those waves. Keep reducing concurrency.
If even one task does not fit, block. No measured speedup is assumed before this rehearsal.

Historical 2026-11-01 results (including the 10% margin): P4's seven concurrent tasks
required **23.87 GiB sampled process GPU memory**, exceeding the 14 GiB limit. The
rehearsed W1 below required **12.86 GiB GPU / 34.17 GiB adjusted RAM**; W2 required
**11.01 / 4.35 GiB**. These are past measurements, not guarantees for another run.
Sampled process GPU memory includes context/allocator overhead; allocated tensor
memory (`peak_cuda_mem_bytes`) is a separate quantity and cannot replace that bound.

Define the waves **once** in `$OPS/waves.sh`, shared by rehearsal and frozen launch.
This starts with the two measured waves. If the all-seven bound passes, a single
`WAVE_TASKS=("$ALL_TASKS")` is also allowed; run the same acceptance loop. Any changed
membership requires a new rehearsal. The check rejects empty, duplicate, missing or
foreign members and requires all seven tasks exactly once.

```bash
#!/bin/bash
# Source after env.sh, from scripts with set -euo pipefail.
WAVE_TASKS=(
    'm5_1W,hermes,rohlik_sales_1W,walmart,rossmann_1W'
    'favorita_stores_1W,hierarchical_sales_1W'
)
env -u LD_LIBRARY_PATH "$VENV/bin/python" - "$ALL_TASKS" "${WAVE_TASKS[@]}" <<'PY'
import sys
all_tasks = sys.argv[1].split(",")
waves = [wave.split(",") for wave in sys.argv[2:]]
members = [task for wave in waves for task in wave]
assert len(all_tasks) == len(set(all_tasks)) == 7 and all(all_tasks)
assert waves and all(all(wave) for wave in waves), "empty wave"
assert len(members) == len(set(members)), "duplicate wave membership"
assert set(members) == set(all_tasks), "incomplete or foreign wave membership"
print("Wave membership verified:", waves)
PY
```

Save `$OPS/rehearse-waves.sh` and run via §2's outer wrapper. Every wave gets a fresh
root, online pin log, task/sampler waits, both capacity bounds and a strict subset
merge/report (including the nonfinite-loss gate). A failure stops the loop; no later
wave is dispatched. Review scaler skips as above. The acceptance record is published
only after **all** waves pass. Keep it and the copied definition with the ledger.

```bash
#!/bin/bash
set -euo pipefail
source /home/ubuntu/fev/env.sh
source "$OPS/waves.sh"
test ! -e "$PROBES/accepted-waves-root.txt"
rehearsals=$(mktemp -d "$PROBES/P4-waves.XXXXXXXX")
cp "$OPS/waves.sh" "$rehearsals/waves.sh"
REPORT_PY=/home/ubuntu/fev/venv-pr/bin/python
for index in "${!WAVE_TASKS[@]}"; do
    wave=${WAVE_TASKS[index]}
    name="W$((index + 1))"
    quick="$rehearsals/$name"
    test ! -e "$quick"
    env -u LD_LIBRARY_PATH "$VENV/bin/python" "$OPS/pin-and-warm.py" > "$rehearsals/$name-pins.log" 2>&1
    bash "$TOOLS/scripts/ec2/frozen_parallel.sh" --venv "$VENV" --run-root "$quick" \
      --config "$TOOLS/configs/fev-v2-quick.yaml" --tasks "$wave"
    IFS=, read -r -a names <<< "$wave"
    for task in "${names[@]}"; do bash "$OPS/wait.sh" "$quick" "$task" 0; done
    env -u LD_LIBRARY_PATH "$REPORT_PY" "$OPS/capacity.py" --root "$quick" --tasks "$wave" \
      --validation-bounds "$PROBES/validation-bounds.json" > "$rehearsals/$name-capacity.txt" 2>&1
    env -u LD_LIBRARY_PATH "$REPORT_PY" "$TOOLS/scripts/fev_merge_runs.py" --run-root "$quick" --tasks "$wave"
    env -u LD_LIBRARY_PATH "$REPORT_PY" "$TOOLS/scripts/fev_report.py" --run-dir "$quick/merged" \
      --tasks "$wave" --baseline seasonal_naive --model-name demand-forecast-v2-quick \
      --out "$quick/merged/report-tables.md"
done
cmp "$OPS/waves.sh" "$rehearsals/waves.sh"
printf '%s\n' "$rehearsals" > "$PROBES/accepted-waves-root.txt"
```

## 9. Failure handling — fallback

If a PR parity gate fails, use `/home/ubuntu/fev/venv-reference` at F (`ba7fc31e569609d25cee013604f6dd299d0baa6e`) and fresh
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
Save `$OPS/frozen-launch.sh`; it uses the accepted definition from §8, verifies pins
before **each** wave with a distinct retained log, and waits for every task and sampler
before the next wave. Any failure blocks later waves and §11.

```bash
#!/bin/bash
set -euo pipefail
source /home/ubuntu/fev/env.sh
: "${ROOT:?fresh frozen root containing executing SHA and UTC}"
source "$OPS/waves.sh"
accepted=$(cat "$PROBES/accepted-waves-root.txt")
cmp "$OPS/waves.sh" "$accepted/waves.sh"
mkdir "$ROOT"
cp "$OPS/waves.sh" "$ROOT/waves.sh"
for index in "${!WAVE_TASKS[@]}"; do
    wave=${WAVE_TASKS[index]}
    env -u LD_LIBRARY_PATH "$VENV/bin/python" "$OPS/pin-and-warm.py" > "$ROOT/W$((index + 1))-pins.log" 2>&1
    sudo shutdown --show
    bash "$TOOLS/scripts/ec2/frozen_parallel.sh" --venv "$VENV" --run-root "$ROOT" \
      --config "$TOOLS/configs/fev-v2-standard.yaml" --tasks "$wave"
    IFS=, read -r -a names <<< "$wave"
    for task in "${names[@]}"; do bash "$OPS/wait.sh" "$ROOT" "$task" 0; done
    printf 'WAVE_DONE %s\n' "$wave"
done
```

From the Mac, use §2's helper:
`bash dispatch-and-wait.sh frozen-launch env ROOT="$ROOT" bash /home/ubuntu/fev/ops/frozen-launch.sh`.
Retain that job's path: its `run.log` and `exit` are the outer frozen-job evidence.
All waves write into the **same root**. This loop does not merge/report; run §11 once,
after its outer status is zero and the ledger accepts all waves.
No `--stop-after`, config change or model-name override in the frozen run. Poll task exits
and logs; check cost/guard hourly. A disconnect does not stop the detached tasks. After a
VM reset, recheck environment/pins/hardware and resume with the same root and config;
the fresh-root script above intentionally refuses an existing root. Save a separate
resume script retaining the wave checks/waits, replacing `mkdir`/the definition copy
with `cmp "$OPS/waves.sh" "$ROOT/waves.sh"`, and using a new log path for each pin check;
the launcher retains earlier sessions. It checks every requested task's `run.log`/`exit`
aliases in the prepare loop, before starting any task: a regular file at either path
refuses the entire launch, so this validation cannot abort a partly dispatched wave.
Preserve such files before retrying. Never delete artifacts to force identity acceptance.

## 11. Report

Every selected task exit and the outer frozen-job exit must be zero, and all pollers
must have valid evidence. Save the following as `$OPS/report.sh` and run it once via
§2's wrapper, passing `env ROOT="$ROOT" bash /home/ubuntu/fev/ops/report.sh` after the
job label. Wait for its outer status zero before §12. It runs the strict merge
(no `--allow-partial`) and report using the merged tools venv:

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
exit 0 without issues. Any task with `nonfinite_losses > 0` blocks frozen acceptance,
including when all CLI exits are zero; never waive this with `--allow-partial`.
Preserve a prior merged directory if regenerating after resume;
never overwrite evidence silently. Use `--allow-partial` only for explicitly labelled
partial troubleshooting, never final acceptance.

Record hardware/dtype, executing SHA per task, dependency/pin evidence, `nonfinite_losses`,
`scaler_skips`, `cold_covariate_columns`, `floored_covariate_columns`,
`clipped_covariate_values` for training and inference,
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
# Include setup scripts/logs, outer job commands/logs/exits and the current manifest.
cd /home/ubuntu/fev
cp /home/ubuntu/setup-fev.sh /home/ubuntu/dispatch-job.sh /home/ubuntu/wait-job.sh host-logs/
cp dataset-manifest.json host-logs/
find ops host-logs env.sh ec2-environment-lock.txt -type f -print0 \
  | sort -z | xargs -0 sha256sum > host-evidence.sha256
cd /home/ubuntu
find fev-inputs -type f -print0 | sort -z | xargs -0 sha256sum >> /home/ubuntu/fev/host-evidence.sha256
```

Copy the run/probe roots, manifests, operator scripts, `env.sh`, `ec2-environment-lock.txt`,
`host-logs/` (including setup logs and the outer frozen-job `run.log` and `exit`),
Colab inputs with both transfer hash records, and local provisioning evidence to the Mac.
The archive layout is `runs/`, `ops/`, `host-logs/`, `fev-inputs/`, `provisioning/`,
plus the environment files and checksum record. Keep the frozen job-path record in
`provisioning/` to identify its unique directory under `host-logs/`.
Save `verify-env.sh` under `ops/`; if any setup command was run outside the wrapper,
first place its saved script and log in `host-logs/` before hashing. Run this manifest
script only after jobs complete, in the foreground: its own output must not change a
file being hashed. **No private key in evidence**; keep the SSH key outside the archive.
Example local Bash script (GNU `sha256sum` or the shown
Python 3.11+ verifier works on macOS):

```bash
#!/bin/bash
set -euo pipefail
: "${KEY:?SSH key}" "${HOST:?public IP}" "${DEST:?local evidence directory}"
: "${PROVISIONING:?absolute local ec2-TOKEN provisioning directory}"
mkdir -p "$DEST"
rsync -a -e "ssh -i $KEY" "ubuntu@$HOST:/home/ubuntu/fev/runs/" "$DEST/runs/"
rsync -a -e "ssh -i $KEY" "ubuntu@$HOST:/home/ubuntu/fev/ops/" "$DEST/ops/"
rsync -a -e "ssh -i $KEY" "ubuntu@$HOST:/home/ubuntu/fev/host-logs/" "$DEST/host-logs/"
scp -i "$KEY" "ubuntu@$HOST:/home/ubuntu/fev/env.sh" "$DEST/"
scp -i "$KEY" "ubuntu@$HOST:/home/ubuntu/fev/ec2-environment-lock.txt" "$DEST/"
scp -i "$KEY" "ubuntu@$HOST:/home/ubuntu/fev/host-evidence.sha256" "$DEST/"
rsync -a -e "ssh -i $KEY" "ubuntu@$HOST:/home/ubuntu/fev-inputs/" "$DEST/fev-inputs/"
mkdir -p "$DEST/provisioning"
# Only the provisioning JSON, ID/job-path records and saved scripts; never the PEM.
rsync -a --exclude="$(basename "$KEY")" --include='*.json' --include='*.txt' \
  --include='*.sh' --exclude='*' "$PROVISIONING/" "$DEST/provisioning/"
cd "$DEST"
python3 - "$PROVISIONING" "$KEY" <<'PY'
import hashlib
import sys
from pathlib import Path

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

for base, manifest in ((Path('runs'), 'copyback.sha256'), (Path('.'), 'host-evidence.sha256')):
    lines = (base / manifest).read_text().splitlines()
    assert lines, manifest
    for line in lines:
        expected, name = line.split('  ', 1)
        assert digest(base / name) == expected, name
for line in Path('runs/copyback.links.txt').read_text().splitlines():
    name, target = line.split(' -> ', 1)
    p = Path('runs') / name
    assert p.is_symlink() and str(p.readlink()) == target and p.exists(), name
source, key = Path(sys.argv[1]), Path(sys.argv[2])
records = []
for p in sorted(source.iterdir()):
    if p.is_file() and p.suffix in ('.json', '.txt', '.sh') and p.name != key.name:
        copied = Path('provisioning') / p.name
        assert digest(p) == digest(copied), str(p)
        assert b'PRIVATE KEY-----' not in copied.read_bytes(), str(copied)
        records.append(f'{digest(copied)}  {copied}\n')
assert records, 'no provisioning evidence'
Path('provisioning.sha256').write_text(''.join(records))
print('Run bytes, symlinks, host/setup evidence, inputs and provisioning verified')
PY
```

Only after successful copy-back verification terminate the recorded instance. Save the
AWS responses and verify the recorded volumes are gone; a stop alone is insufficient.
Delete only this experiment's resources, never unrelated instances/volumes/groups.

```bash
#!/bin/bash
set -euo pipefail
export AWS_PROFILE=veliu-staging AWS_REGION=eu-central-1 AWS_DEFAULT_OUTPUT=json
: "${INSTANCE:?recorded instance ID}" "${SG:?recorded security group}" "${KEY_NAME:?recorded key name}"
: "${DEST:?verified local evidence directory}"
cd "$DEST/provisioning"
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
