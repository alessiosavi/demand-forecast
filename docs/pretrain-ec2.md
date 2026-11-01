# Pretraining on EC2 — operator runbook (2026-11-01)

This is the opt-in Phase C/D experiment. Follow the [protocol](protocol.md); keep
its frozen YAML, identities and weekly profiles unchanged. Record the merged Phase C
SHA **P**, green CI, bot review and critical verification before provisioning. Record
the leakage audit's exclusion of `solar_1W`/`solar_1D` and absence of other overlaps
with the 12 target tasks **before downloading any corpus bytes**. Another overlap
blocks this experiment and returns the decision to the owner.

Run the saved Bash recipes with `set -euo pipefail`. Commands marked **operator** run
on the laptop; **guest** commands run on EC2. Fill evidence-derived variables before
running a block. Local work is code and synthetic tests only. Provisioning, downloads,
probes, training and evaluation below are future operator actions, not results.

## 1. Market and budget

Use `veliu-staging`, `eu-central-1`, **one on-demand g6.xlarge** (L4 24 GB, 4 vCPU,
16 GiB RAM), 200 GB gp3 root disk, `DeleteOnTermination=true`, shutdown behaviour
**stop**. Select an inspected x86_64 Deep Learning Base OSS Nvidia Driver GPU AMI
(Ubuntu 22.04/24.04) with Python 3.12 available. Reuse the same instance/root disk
between sessions. Spot is not authorised for this attempt.

Budget inputs: compute $1.0064/h plus public IPv4 $0.005/h, combined **rate 1.0114**;
EBS $0.0952/GB-month at 720 h/month; egress $0.09/decimal GB. These are ledger inputs,
not a live price quotation. Binding cap **$50**, admission limit **$47**. Pretraining
including setup/drills/probes is at most $30, evaluation hold at most $12. At sizing,
keep $3 operating reserve free of holds plus the $3 enforcement margin. A session
reserves `(minutes + 20) / 60 × rate`; settlement uses total elapsed seconds with
a 60-second minimum. EBS is funded through `cleanup_by + 24 h`, rounded up to cents.
Stopped storage still costs money. Reconcile actual billing with `record`.

## 2. Initialise the operator ledger

Use AWS CLI, SSH, rsync and Python 3 on the operator. Retain the AMI selection, VPC,
subnet and operator IP evidence using [the existing provision conventions](ec2.md#1-provision).
Use a new absolute directory for this experiment, outside the source checkout. Keep
its PEM private; it must never enter the evidence archive.

**Operator**, in every terminal that runs guard commands:

```bash
set -euo pipefail
export AWS_PROFILE=veliu-staging AWS_REGION=eu-central-1 AWS_DEFAULT_OUTPUT=json AWS_PAGER=''
export PYTHON=python3
: "${LOCAL_CODE:?absolute checkout at P}" "${EXPERIMENT:?absolute operator evidence directory}"
: "${OPERATOR_IP:?current public IPv4}" "${VPC_ID:?inspected VPC}"
: "${SUBNET_ID:?public subnet}" "${AMI_ID:?inspected AMI ID}"
export OPERATOR_IP VPC_ID SUBNET_ID AMI_ID
mkdir -p "$EXPERIMENT/ops"
export LEDGER="$EXPERIMENT/ops/cost-ledger.json"
guard() { "$PYTHON" "$LOCAL_CODE/scripts/ec2/cost_guard.py" --ledger "$LEDGER" "$@"; }
launch() { bash "$LOCAL_CODE/scripts/ec2/pretrain_launch.sh" "$@" --ledger "$LEDGER"; }
: "${CLEANUP_BY:?UTC ISO-8601 launch plus five days}"
guard init --cap 50 --enforce 47 --cleanup-by "$CLEANUP_BY" \
  --type g6.xlarge --market on-demand --price 1.0064 --ipv4-usd-per-h 0.005 \
  --volume-gb 200 --region eu-central-1 --create-role
guard status
```

`init --create-role` creates/verifies the experiment's Scheduler role and installs
its tag-conditioned stop/terminate policy. Admission fails closed on errors; do not
reset or edit the ledger to remove a commitment. `--ledger` is a **global** guard
option, before its subcommand; the launch wrapper accepts it after the operation.

## 3. Provision as the first reserved session

**Operator**, in a supervised terminal, never a detached laptop job:

```bash
launch provision --label setup --minutes 180 \
  --user-data "$LOCAL_CODE/scripts/ec2/pretrain-user-data.sh"
guard status
```

The wrapper reserves before launch and installs EXIT/ERR/INT/TERM recovery traps.
If the process exits before the AWS stop schedule is verified, `launch-trap` discovers
and force-stops the instance, waits for stopped, then settles the incurred cost. A
failed stop retains the reservation. Loss of the **host** prevents the trap; see B′
in §12. Do not call `run-instances` directly or repeat `provision` after an ambiguous
launch. Use `reconcile` and the recorded transaction instead.

Task 14 passes `--client-token <token>` and both instance and volume tags
`experiment=pretrain-<token>` in `--tag-specifications`. It creates/read-backs the
session stop schedule at `deadline + 20 min` immediately after launch and the
termination schedule at `cleanup_by`. On later `start`, the stop schedule is verified
**before** starting the instance. UTC timezone, `ENABLED`, flexible window `OFF`,
target and retry policy are checked; retain `status` showing both schedules `verified`.

The self-contained user data installs the deadline helper and unit, creates
`/home/ubuntu/fev/ops/` owned by `ubuntu`, and needs no repository, Python or network.
The oneshot unit runs after `multi-user.target` on every boot, cancels any old timer
and arms 20 minutes of grace. It reads no saved deadline. `DefaultDependencies=no`
avoids the target ordering cycle. Standard output reaches the journal and console.
Only `arm-deadline.sh` calls shutdown. Its `set` reply is exactly
`DEADLINE_SET <UTC>`; a time later than requested or over 60 seconds early fails with
`DEADLINE_MISMATCH`. The AWS stop schedule is the hard runtime backstop once verified.

Extract `KEY` from `config.key_path`, `HOST` from the current session's `public_ip`,
and `INSTANCE` from `instance_id` in the ledger. The guard arms the deadline over batch
SSH with `StrictHostKeyChecking=accept-new` and a ledger-local known-hosts file
(`config.known_hosts_path`): the first host key is pinned, a changed key is refused.
It retries every 20 seconds while sshd or cloud-init is not ready (connection refused
or timed out, helper not installed yet) until the earlier of the transaction's
`aws_called + 15 min` and the session deadline, so arming always ends before the guest's
own boot grace (boot + 20 min) can fire; a `DEADLINE_MISMATCH` or malformed reply fails
immediately. Exhaustion or a final reply means exit 4 after a safe stop; the attempts
are recorded in the session's `arm_attempts`. After the first successful arming, compare
the pinned key in `known_hosts` with the instance console's host-key fingerprint and
record the comparison. If arming failed: inspect the console and ledger, `reconcile`,
then `start` with a **new** label. Do not disable host-key verification. Refresh `HOST`
after every restart.

| Guard exit | Meaning/action |
| --- | --- |
| 0 | Command completed; inspect flags and schedule read-back as well. |
| 2 | Admission, argument, AWS, I/O or other operational failure; inspect and reconcile. |
| 3 | Spot requested; this attempt permits on-demand only. |
| 4 | Guest deadline could not be armed/verified; stop attempted, reconcile its result. |
| 6 | An open/pending session exists; settle/reconcile it before another start. |
| 7 | `EXPIRED_SESSION` blocks new commitments; reconcile and inspect the stopped instance. |

`status` can exit 0 with an `UNVERIFIED` schedule. `OVER CAP` means the **$47 admission
limit** was exceeded; it is printed even for a successful recovery command. A green
exit code alone is never evidence that resource protection or the $50 cap holds.

## 4. Setup, closure parity and 12-task warm-up

Copy the original Colab environment lock, dataset manifest, baseline CSV and hash
record using [EC2 §2](ec2.md#2-setup-code-lock-pins-sequential-cache-warm-up). Require
matching source/guest hashes. Preserve originals. Generate `install-constraints.txt`
by removing only lines containing ` @ ` from the Colab freeze, as that section does.
The setup script verifies this exact transformation; do not loosen version pins.

Save `/home/ubuntu/dispatch-job.sh`, `/home/ubuntu/wait-job.sh` and the operator's
`dispatch-and-wait.sh` **exactly from EC2 §2**, before setup. Set `PROVISIONING` to
`$EXPERIMENT/ops` and `JOB_TIMEOUT_SECONDS` to less than the session time remaining.
The dispatcher saves a unique `host-logs/` command/log/exit directory. Only an observed
exit 0 permits the next gate; absent status after interruption means **unknown**.
Inspect the recorded job before retrying so a lost SSH reply cannot duplicate work.

**Operator**:

```bash
: "${KEY:?ledger key_path}" "${HOST:?current session public_ip}" "${FEV_SHA:?full merged P}"
scp -i "$KEY" "$LOCAL_CODE/scripts/ec2/pretrain-setup.sh" "ubuntu@$HOST:/home/ubuntu/pretrain-setup.sh"
export PROVISIONING="$EXPERIMENT/ops" JOB_TIMEOUT_SECONDS=9000
bash dispatch-and-wait.sh setup-pretrain env FEV_SHA="$FEV_SHA" \
  INSTALL_CONSTRAINTS=/home/ubuntu/fev-inputs/install-constraints.txt \
  HF_HOME=/home/ubuntu/fev/hf bash /home/ubuntu/pretrain-setup.sh
```

`pretrain-setup.sh` installs the pinned `.[bench]` closure plus `huggingface_hub`,
requires `closure_parity.py` exit 0 and `pip_check_closure.py` exit 0, and creates or
compares the exact EC2 freeze. It then warms **weekly + heldout**, sequentially. Both
Hub SHA and `main` must equal `d3e7e720f4158eab03ebcb021f1cce779cd92acc` before loading
and again after warming. Each task requires exactly one parquet, a matching pinned
LFS SHA/size and a stable dataset fingerprint. Weekly tasks also match the original
Colab manifest. That seven-task manifest has no held-out entries: held-out filenames
come from the pinned Hub card's config/train mapping; absent/ambiguous metadata blocks
setup. No inferred filename, unpinned fallback or offline bypass is allowed.

Save `/home/ubuntu/fev/env.sh` with the following guest variables. Set `FEV_SHA` to P
(and later the approved P2), and keep `UTC` stable across restarts:

```bash
export FEV_SHA='<full approved SHA>' UTC='<experiment UTC token>'
export TOOLS=/home/ubuntu/fev/code VENV=/home/ubuntu/fev/venv-pr
export PY="$VENV/bin/python" PATH="$VENV/bin:$PATH"
export COLAB_LOCK=/home/ubuntu/fev-inputs/environment-lock.txt
export COLAB_MANIFEST=/home/ubuntu/fev-inputs/colab-dataset-manifest.json
export INSTALL_CONSTRAINTS=/home/ubuntu/fev-inputs/install-constraints.txt
export FEV_LOCK_FILE=/home/ubuntu/fev/ec2-environment-lock.txt
export FEV_DATASET_MANIFEST=/home/ubuntu/fev/dataset-manifest.json
export HF_HOME=/home/ubuntu/fev/hf HF_HUB_CACHE=/home/ubuntu/fev/hf/hub
export HF_DATASETS_CACHE=/home/ubuntu/fev/hf/datasets
export RUNS=/home/ubuntu/fev/runs OPS=/home/ubuntu/fev/ops
export PROBES="$RUNS/probes-$UTC"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
unset LD_LIBRARY_PATH HF_HUB_OFFLINE HF_DATASETS_OFFLINE FEV_DATASETS_PREFIX PYTHONPATH
```

Source it in every guest job and `cd "$TOOLS"`. Before each subsequent workload,
repeat the closure/lock comparison and run `"$PY" "$OPS/pin-and-warm.py"` with these
environment variables, retaining fresh evidence in `ops/`. A moved upstream main,
changed fingerprint or dependency drift blocks continuation. Verify L4/device memory
and dtype from actual host evidence; do not reuse the old T4 capability assertion.

## 5. Corpus download and manifest

There is **no** `pretrain download` CLI. Use the existing Python APIs below in a saved
guest script under `dispatch-job.sh`, during the reserved setup session. The checked-in
YAML trains on TSMixup **0–198**, with hold-out **199**, and KernelSynth **0–18**, with
hold-out **19** (resolved config defaults). Download their unions: **0–199 and 0–19**.
A manifest omitting the hold-outs cannot run validation. Expected corpus size is about
84 GiB; measure actual sizes and available disk before proceeding.

```bash
set -euo pipefail
source /home/ubuntu/fev/env.sh
cd "$TOOLS"
mkdir -p "$RUNS" "$PROBES"
"$PY" - <<'PY'
from pathlib import Path
from demand_forecast.pretrain.config import PretrainConfig
from demand_forecast.pretrain.corpus import download_shards, corpus_manifest, write_manifest
from demand_forecast.pretrain.validate import download_external
cfg = PretrainConfig.from_yaml(Path('configs/pretrain-v2-first-try.yaml'))
c = cfg.corpus
c.root.mkdir(parents=True, exist_ok=True)
ts = sorted(set(c.tsmixup_shards + c.holdout_tsmixup))
ks = sorted(set(c.kernel_synth_shards + c.holdout_kernel_synth))
assert ts == list(range(200)) and ks == list(range(20))
paths = download_shards(c.root, ts, ks)
assert all(p.is_file() for p in paths)
refs = corpus_manifest(c.root)
assert {(r.kind, r.index) for r in refs} == ({('tsmixup', i) for i in ts} | {('kernel_synth', i) for i in ks})
write_manifest(refs, c.root / 'manifest.json')
assert c.external_validation == 'monash_nn5_weekly'
download_external(c.external_validation, c.root)
PY
cp /home/ubuntu/fev/corpus/manifest.json "$OPS/corpus-manifest.json"
"$PY" scripts/profile_fev.py --tasks heldout --out "$RUNS/profile-heldout.csv"
```

Retain manifest hashes, disk measurements and setup logs. The profile is diagnostic;
`target_support` comes from preregistered semantics. **Operator**: `guard stop`, then
`guard reconcile`. Settle all preparation sessions; neither EBS nor preparation is free.

## 6. Drills before pretraining

**Operator**: `launch start --label drills --minutes 60`, then `guard status`. Refresh
HOST. In the guest, record `journalctl -b -u budget-deadline.service` and the console's
`DEADLINE_GRACE`. Reboot once (`sudo reboot`); verify a fresh grace event within 20
minutes in the current boot's journal and console. Read the scheduled time with:

```bash
sudo busctl get-property org.freedesktop.login1 /org/freedesktop/login1 \
  org.freedesktop.login1.Manager ScheduledShutdown
# SESSION_DEADLINE is the existing session's exact deadline from the operator ledger.
sudo /usr/local/sbin/arm-deadline.sh set "$SESSION_DEADLINE"
```

Require the `DEADLINE_SET` read-back within [-60, 0] seconds of that deadline. The
reboot does **not** extend the reservation. `guard status` must still show the session
stop and cleanup schedules `verified`; Task 14 checks `ENABLED` and the stop expression
at deadline + 20, rather than printing those raw fields. Archive the output.

Guest resume drill (through the saved dispatcher, with environment sourced):

```bash
demand-forecast pretrain run --config configs/pretrain-v2-first-try.yaml \
  --stop-after 200 --out "$RUNS/drill"
```

The intentional interruption exits **1**; poll its exact job with
`bash /home/ubuntu/wait-job.sh "$JOB" 1 3000`, not the exit-0 helper. **Operator**:
`guard stop`; `guard reconcile`; `launch start --label drill2 --minutes 30`. Guest:

```bash
demand-forecast pretrain run --config configs/pretrain-v2-first-try.yaml \
  --resume --stop-after 200 --out "$RUNS/drill"
```

Again expect exit 1. Inspect `checkpoints/latest.pt` and retained history for step
**400**, continuous steps and identical pretrain identity. Do not trust the exit alone.
Stop and reconcile. Any failing drill blocks the full run.

## 7. Probes in a reserved session

**Operator**: `launch start --label probes --minutes 150`. Guest: create temporary
candidate YAMLs under `ops/candidates/`, using the committed configs as templates and
changing **model fields only**. Keep these files and their hashes with the probe JSONs.
Do not edit the frozen configs or choose using fev scores.

| Candidate | d_model / nhead | Encoder + decoder | Feedforward / hidden continuous | Approx. parameters |
| --- | --- | --- | --- | --- |
| L1 | 256 / 4 | 3 + 3 | 1024 / 256 | 6.06M |
| L2 | 256 / 4 | 4 + 4 | 1024 / 256 | 7.90M |
| L3 | 320 / 5 | 4 + 4 | 1280 / 320 | 12.3M |
| L4 | 384 / 6 | 4 + 4 | 1536 / 384 | 17.7M |

Generate those candidate files on the guest before the probes:

```bash
"$PY" - <<'PY'
import os
from pathlib import Path
import yaml
root = Path(os.environ['OPS']) / 'candidates'
root.mkdir(exist_ok=True)
for label, d, heads, layers, ff in [('L1', 256, 4, 3, 1024), ('L2', 256, 4, 4, 1024),
                                  ('L3', 320, 5, 4, 1280), ('L4', 384, 6, 4, 1536)]:
    for kind, filename in [('pretrain', 'pretrain-v2-first-try.yaml'),
                           ('finetune', 'fev-v2-pretrained-finetune.yaml'),
                           ('zero-shot', 'fev-v2-pretrained-zero-shot.yaml')]:
        cfg = yaml.safe_load((Path('configs') / filename).read_text())
        cfg['model'].update(d_model=d, nhead=heads, num_encoder_layers=layers,
                            num_decoder_layers=layers, dim_feedforward=ff, hidden_continuous_size=d)
        (root / f'{label}-{kind}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
```

For each L, set `PRETRAIN_CONFIG`, `FINETUNE_CONFIG`, `ZERO_SHOT_CONFIG` to those
candidate files and `CANDIDATE` to its name. Run sequentially as detached guest jobs:

```bash
demand-forecast pretrain probe --config "$PRETRAIN_CONFIG" --steps 50 \
  --out "$PROBES/$CANDIDATE-pretrain.json"
demand-forecast pretrain probe-task --tasks weekly --config "$FINETUNE_CONFIG" \
  --out "$PROBES/$CANDIDATE-weekly"
demand-forecast pretrain probe-task --tasks heldout --config "$ZERO_SHOT_CONFIG" \
  --out "$PROBES/$CANDIDATE-heldout"
# Once for the frozen d64 control:
demand-forecast pretrain probe-task --tasks heldout --config configs/fev-v2-standard.yaml \
  --out "$PROBES/d64-heldout"
```

`probe-task` measures bounded training/validation and last-window inference with
real task shapes, even with a zero-shot configuration; it does not select models.
Do not supply a fake pretrained bundle or use `--max-series` for sizing. Record all
12 tasks' timing/memory evidence. If time is short, stop, reconcile, and continue in a
new uniquely labelled reservation. Stop/reconcile once all probes are collected.

## 8. Evaluation hold and size choice

Use only probe timing, memory, dollars and corpus operational checks. For each weekly
task resolve `(max_steps_t, eval_every_t) = resolve_budget(3000, n_fit_ids_t, 200)`.
Compute evaluations as `ceil(max_steps_t / eval_every_t) + initial_eval` and checkpoints
as `ceil(max_steps_t / 200) + initial_eval`. `initial_eval=1` for fine-tune/wide-random,
0 for frozen d64. Task seconds are:

```text
init_s + prep_s + max_steps_t * s_per_step + n_evals_t * eval_s
+ n_ckpt_t * ckpt_s + 60 + sum_over_windows(init_last_s + infer_s)
```

Sum the seven weekly tasks and multiply by two (fine-tune and wide-random); add five
held-out d64 controls; add zero-shot over all 12 tasks using
`init_s + sum_over_windows(init_last_s + infer_s)`; add 600 s baselines and 300 s session
overhead. Use the last-window measurements for every window as an upper bound.
`minutes_E = ceil(1.3 × E_s / 60)` and `E_usd = reservation_usd(minutes_E, 1.0114)`
from `scripts.ec2.cost_guard`. Require E_usd ≤ $12; drop L4 then L3 if necessary.
All reported validation/inference/evaluation CUDA peaks and peak RSS must remain
below 90% of the actual device/host memory.

**Operator**, after all prior sessions are settled:

```bash
: "${E_USD:?computed candidate evaluation reservation}"
guard hold --label eval --usd "$E_USD"
guard status
```

With committed already including EBS, this eval hold and settled sessions, compute
`remaining = enforce_usd − committed`, `pretrain_spent = sum(settled setup/drill/probe
costs)`, `D = min(30 − pretrain_spent, remaining − 3.0)` and
`B_minutes = floor(D / 1.0114 × 60) − 20`. For each L find the largest step count whose
`pretrain_duration(cfg_L, probe_L)["minutes"] ≤ B_minutes`. The helper is imported from
`demand_forecast.pretrain.train` and already includes overhead and a 15% time reserve;
do not replace it with optimizer time alone.

Choose the largest candidate reaching the soft target of 150,000 steps. Otherwise
choose L1 only if it reaches **60,000**; below that, stop and consult the owner. Release
a trial hold with `guard release --label eval` before placing the chosen hold. Record
E/D/B_minutes, candidate probes and status. No configuration follows from fev scores.

Claude makes the reviewed size commit/PR: `model.*`, `training.max_steps` and
`training.eval_every = max(5000, steps // 30)` in the pretrain YAML, **model.* only** in
the three experiment YAMLs. Owner-approved merge gives **P2**. This runbook does not
authorise changing the frozen standard/quick YAML, weekly definitions or profiles.

## 9. Pretrain and resume

**Operator**: set `PRETRAIN_MINUTES` from the chosen config/probe helper, then:

```bash
launch start --label pretrain --minutes "$PRETRAIN_MINUTES"
# In a second supervised operator terminal with the same ledger:
guard watch --interval 300
```

Guest: check out approved P2, verify the exact SHA, source `env.sh`, rerun environment
and dataset gates, and keep one stable output path:

```bash
: "${JOB_TIMEOUT_SECONDS:?less than the remaining reserved session time}"
export PRETRAIN="$RUNS/pretrain-$FEV_SHA-$UTC"
JOB=$(bash /home/ubuntu/dispatch-job.sh pretrain demand-forecast pretrain run \
  --config configs/pretrain-v2-first-try.yaml --out "$PRETRAIN")
printf '%s\n' "$JOB" > "$OPS/pretrain-job.txt"
bash /home/ubuntu/wait-job.sh "$JOB" 0 "$JOB_TIMEOUT_SECONDS"
```

A stop is not completion. After **any** stop: operator `guard reconcile`, reserve a
new uniquely labelled session (for example `launch start --label pretrain-resume
--minutes "$REMAINING_MINUTES"`), restore the same SHA/environment/config/output, and
append **`--resume`** to the dispatched `pretrain run`. Compute remaining minutes from
history plus probe overhead; never overwrite the run or extend the guest timer without
a corresponding reservation. Subsequent resumes need distinct labels.

Require `stop_reason == max_steps`, `nonfinite_losses == 0`, finite `val_sql` at every
evaluation and verified bundle hashes. Record `ext_sql` and `ext_sql_naive` for the
external diagnostic. Corpus hold-out SQL selects the checkpoint; external/fev scores
do not select capacity or training configuration. Stop and reconcile.

## 10. Evaluation sweep order

**Operator**: `launch start --label eval --minutes "$MINUTES_E" --consume-hold eval`.
Do not reserve evaluation twice. Guest, under the same dispatcher/wait discipline,
run these sweeps in order, requiring each to finish before the next:

```bash
demand-forecast fev baseline --tasks weekly --out-dir "$RUNS/baseline"
demand-forecast fev baseline --tasks heldout --out-dir "$RUNS/baseline"
demand-forecast fev run --tasks weekly --config configs/fev-v2-pretrained-zero-shot.yaml \
  --init-from "$PRETRAIN/bundle" --model-name demand-forecast-v2-zero-shot --out-dir "$RUNS/zero-shot"
demand-forecast fev run --tasks heldout --config configs/fev-v2-pretrained-zero-shot.yaml \
  --init-from "$PRETRAIN/bundle" --model-name demand-forecast-v2-zero-shot --out-dir "$RUNS/zero-shot"
demand-forecast fev run --tasks weekly --config configs/fev-v2-pretrained-finetune.yaml \
  --init-from "$PRETRAIN/bundle" --model-name demand-forecast-v2-finetuned --out-dir "$RUNS/finetune"
demand-forecast fev run --tasks weekly --config configs/fev-v2-wide-random.yaml \
  --model-name demand-forecast-v2-wide-random --out-dir "$RUNS/wide-random"
demand-forecast fev run --tasks heldout --config configs/fev-v2-standard.yaml \
  --model-name demand-forecast-v2-heldout-frozen --out-dir "$RUNS/heldout-frozen"
```

After a stop, reconcile and reserve remaining evaluation time before resuming the
same roots with `fev run --resume`; baseline already skips completed rows. Keep the
prior frozen weekly `demand-forecast-v2` evidence for the final comparison; do not
rerun/select it from experiment scores. There must be 12 `seasonal_naive` rows.

After each sweep merge its matching baseline rows into that root's `summaries.csv`
by `(task, model_name)`, preserving all columns/other rows. `TASKS` is `weekly` or
`heldout`; `MODEL` and `SWEEP_ROOT` match the sweep above. Require report exit 0:

```bash
"$PY" scripts/fev_report.py --run-dir "$SWEEP_ROOT" --tasks "$TASKS" \
  --model-name "$MODEL" --out "$SWEEP_ROOT/report-$TASKS.md"
```

Check zero-shot separately for both selectors. Stop and reconcile after evaluation.

## 11. Report and verified copy-back

Reserve `launch start --label copyback --minutes 60`. Prepare published fev CSVs at
commit `193e4fc` with a SHA manifest. Supply the prior frozen run root and all sweep
roots to the implemented report CLI. Stage the prior frozen evidence under
`$RUNS/frozen` and published inputs/hash manifest under `$RUNS/published`, so both
enter the copy-back set. `spend.json` is a measured report input with
`runs` entries named `frozen`, `zero-shot`, `finetune`, `wide-random`, `heldout-frozen`,
each with `spend_usd` and `wall_time_s`; optional top-level totals include pretraining.
It is **not** a renamed cost ledger. Keep a separate raw ledger snapshot in `ops/`.

```bash
"$PY" scripts/pretrain_report.py --frozen "$FROZEN_ROOT" --zero-shot "$RUNS/zero-shot" \
  --finetune "$RUNS/finetune" --wide-random "$RUNS/wide-random" \
  --heldout-frozen "$RUNS/heldout-frozen" --baseline "$RUNS/baseline" \
  --published "$PUBLISHED" --fev-commit 193e4fc --spend /home/ubuntu/fev/spend.json \
  --out "$RUNS/REPORT.md"
```

Quiesce all workload writers before hashing. The copy-back set is `runs/**` except
`checkpoints/*.pt`, plus `ops/`, `host-logs/`, probe JSONs (under `runs/probes-*`),
`spend.json`, environment locks, dataset/corpus manifests and setup/input evidence.
**Include bundle weights**. Hash and size omitted checkpoint files in
`omitted.sha256`; omission means their remote bytes cannot be reverified after deletion.
No PEM/private key belongs in the copy-back set. Save a ledger snapshot without its
adjacent PEM; do not recursively copy the operator ledger directory to the guest.

Create these manifests on the guest from `/home/ubuntu/fev` after the report succeeds:

```python
import hashlib
import json
from pathlib import Path

root = Path("/home/ubuntu/fev")
files, omitted, links = [], [], []
paths = set()
for name in ("runs", "ops", "host-logs"):
    paths.update(p for p in (root / name).rglob("*") if p.is_file() or p.is_symlink())
for name in ("spend.json", "env.sh", "ec2-environment-lock.txt", "dataset-manifest.json"):
    p = root / name
    assert p.is_file(), p
    paths.add(p)
for p in sorted(paths):
    name = p.relative_to(root).as_posix()
    assert "\n" not in name and p.suffix != ".pem", name
    if p.is_symlink():
        target = str(p.readlink())
        resolved = p.resolve()
        assert not Path(target).is_absolute() and resolved.is_relative_to(root) and p.exists(), name
        assert not (resolved.parent.name == "checkpoints" and resolved.suffix == ".pt"), name
        links.append(dict(path=name, target=target, bytes=len(target.encode())))
        continue
    with p.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    entry = dict(path=name, bytes=p.stat().st_size, sha256=sha)
    if p.parent.name == "checkpoints" and p.suffix == ".pt":
        omitted.append(entry)
    else:
        files.append(entry)
(root / "omitted.sha256").write_text(
    "".join(f"{x['sha256']}  {x['bytes']}  {x['path']}\n" for x in omitted)
)
p = root / "omitted.sha256"
files.append(
    dict(path=p.name, bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
)
p = root / "copyback.links.json"
p.write_text(json.dumps(links, indent=2) + "\n")
files.append(
    dict(path=p.name, bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
)
(root / "copyback.sha256").write_text("".join(f"{x['sha256']}  {x['path']}\n" for x in files))
p = root / "copyback.sha256"
files.append(
    dict(path=p.name, bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
)
files.extend(links)
(root / "copyback-files.txt").write_text("".join(x["path"] + "\n" for x in files))
# Reserve a further MiB for rsync metadata; settlement uses the receive counter.
(root / "copyback-manifest.json").write_text(
    json.dumps(dict(bytes=sum(x["bytes"] for x in files) + 1048576, files=files), indent=2) + "\n"
)
```

Relative symlinks within the evidence root (including frozen-run log aliases) are
preserved and recorded in `copyback.links.json`. Absolute, dangling, escaping links
or links to omitted weights block archiving for inspection.

Reserve **every transfer attempt**, including a failed/partial transfer and each retry.
First reserve a small metadata transfer so obtaining the manifest is also accounted
for. **Operator** (a fresh `DEST` under the plan workspace):

```bash
: "${DEST:?absolute .superpowers/sdd/plan/ec2-evidence-pretrain-UTC directory}"
mkdir -p "$DEST"
printf '{"bytes":1048576}\n' > "$EXPERIMENT/ops/metadata-transfer.json"
# Confirm the three metadata files total at most 1 MiB before this transfer.
guard transfer-begin --manifest "$EXPERIMENT/ops/metadata-transfer.json" --attempt 1
scp -i "$KEY" "ubuntu@$HOST:/home/ubuntu/fev/copyback-manifest.json" \
  "ubuntu@$HOST:/home/ubuntu/fev/copyback-files.txt" \
  "ubuntu@$HOST:/home/ubuntu/fev/copyback.sha256" "$DEST/"
META_BYTES=$(wc -c < "$DEST/copyback-manifest.json")
META_BYTES=$((META_BYTES + $(wc -c < "$DEST/copyback-files.txt") + $(wc -c < "$DEST/copyback.sha256")))
guard transfer-end --manifest "$EXPERIMENT/ops/metadata-transfer.json" --attempt 1 --bytes "$META_BYTES"
MANIFEST="$DEST/copyback-manifest.json"
guard transfer-begin --manifest "$MANIFEST" --attempt 1
LC_ALL=C rsync -a --stats --files-from="$DEST/copyback-files.txt" -e "ssh -i $KEY" \
  "ubuntu@$HOST:/home/ubuntu/fev/" "$DEST/" > "$DEST/rsync-attempt-1.log" 2>&1
TRANSFERRED_BYTES=$("$PYTHON" - "$DEST/rsync-attempt-1.log" <<'PY'
import re
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text()
match = re.search(r'^Total bytes received: ([0-9,]+)', text, re.M)
assert match, 'missing rsync receive counter; inspect log before settlement'
print(int(match[1].replace(',', '')))
PY
)
guard transfer-end --manifest "$MANIFEST" --attempt 1 --bytes "$TRANSFERRED_BYTES"
```

The reservation manifest is JSON `{"files":[{"bytes":N,...}]}` (or `{"bytes":N}`),
not a SHA text file. The generated JSON includes a conservative top-level `bytes`
reservation plus the exact file entries; settlement uses rsync's receive counter.
Retain the log and reconcile billing differences, including transport overhead. On command failure, the shell stops **before settlement**: collect
partial-byte evidence and run `transfer-end` with actual bytes. If unavailable,
retain the hold while investigating; never invent zero. Reserve a new attempt number
before retrying, and settle it independently. The same applies to metadata retries.
Do not change a manifest path between its begin/end calls.

**Operator**, verify all transferred hashes and sizes before marking copy-back:

```bash
"$PYTHON" - "$DEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
entries = json.loads((root / 'copyback-manifest.json').read_text())['files']
assert entries
for entry in entries:
    path = root / entry['path']
    assert path.resolve().is_relative_to(root), path
    if 'target' in entry:
        assert path.is_symlink() and str(path.readlink()) == entry['target'] and path.exists(), path
        assert len(entry['target'].encode()) == entry['bytes'], path
        continue
    assert path.is_file() and not path.is_symlink(), path
    assert path.stat().st_size == entry['bytes'], path
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == entry['sha256'], path
for line in (root / 'omitted.sha256').read_text().splitlines():
    sha, size, name = line.split('  ', 2)
    assert len(sha) == 64 and int(size) >= 0 and Path(name).suffix == '.pt'
print('Copied bytes and omission record verified; omitted remote weights intentionally absent')
PY
guard stop
guard reconcile
guard copyback-verified --manifest "$MANIFEST" --sha-report "$DEST/copyback.sha256"
```

`copyback-verified` checks that its two files **exist** and records the operator's
assertion; it does not verify their hashes. The preceding verification is mandatory.
Retain the operator ledger and post-stop/final spend snapshots locally as well as the
guest snapshot, so copy-back/teardown costs are not lost from the final report.

## 12. Recovery obligation and residual exposure

While **any experiment resource exists**, the operator must run `guard reconcile`
(or `guard watch --interval 300`) **at least every 24 hours**, inspect flags and act.
`OVERDUE_STORAGE` means storage exists after `cleanup_by`: reconcile/stop any session,
then **`guard terminate --abandon`**, accepting loss of uncopied data to end exposure.
`EXPIRED_SESSION` requires reconciliation; never clear the JSON flag by hand.
`OVER CAP` requires recording/reporting the $47 enforcement breach separately from
the binding $50 cap. Recovery/recording commands are allowed even over the limit.

For ambiguous transactions, preserve console evidence, then use
`guard cancel --txn "$TXN" --evidence "$EVIDENCE_TEXT"` only with confirmed disposition.
An empty discovery response is not proof that a launch never occurred. A planned
extension uses `guard extend-cleanup --to "$NEW_CLEANUP_UTC"` **before** expiry; it
reserves added EBS before updating/read-backing the schedule. Actual positive billing
adjustments use `guard record --kind other --usd "$USD" --note "$NOTE"` (kinds also
include `ebs` and `transfer`); record only the difference not already accounted for.

The plan's residual table below measures **incremental** cost beyond ledger commitments,
under the 24-hour operator recovery assumption, rate $1.0114/h and EBS ≈ $0.634/day:

| Failure | Incremental exposure | Bound | ≈ USD |
|---|---|---|---|
| A. Guest bootstrap stalls/fails and operator lost | compute from `deadline + 20` (already funded) until the stop schedule succeeds: 60 s precision + up to 10 retries within 60 min | session stop schedule | ≤ 61 min → **$1.03**; typical < $0.05 |
| B. Operator process dies in the seconds between `run-instances` and schedule creation | the launch trap stops the instance; EBS until the operator's next `reconcile` (≤ 24 h) → `terminate --abandon` | launch trap; recovery assumption | **$0.64** (EBS one day); if the guest grace also fails but the trap ran: still $0.64 |
| B′. Operator **host** lost in that window (no trap runs) **and** the guest grace fails | compute from the funded reservation end until the next `reconcile` (≤ 24 h): `(24 − (M + 20)/60) h × rate`, for the 180-min provisioning session ≤ **$20.90**, plus one EBS day | none inside the margin: two independent failures inside a window of seconds | **disclosed, not bounded** |
| C. Termination at `cleanup_by` fails after retries | EBS from `cleanup_by + 24 h` (funded) until the operator's next `reconcile` (≤ 24 h) | `OVERDUE_STORAGE` flag; recovery assumption | **$0.64** |
| D. Stop/terminate asynchronous | a few minutes of compute | — | **< $0.10** |
| E. Short sessions | none (60-s minimum is inside `session_cost_usd`) | I3 | **0** |
| F. Transfer retries | none (each attempt reserved first) | I7 | **0** |

Worst single failure **$1.03** (A); worst plausible pair inside the margin **$1.67** (A+B); margin $3 (`cap − enforce`). The double failure B′ is outside the margin and is disclosed as such rather than claimed bounded; the only further mitigation would be a pre-launch AWS-side guard, which this attempt does not build. Outside the recovery assumption the exposure grows at ≈ $0.64/day of EBS, which is why the runbook makes the 24-hour check an operator obligation.

## 13. Teardown and final evidence

Order: **stop → verify copy-back → copyback-verified → terminate → schedule deletion
(by terminate) → key pair/security group → final status**. `terminate` refuses an
open/pending session or absent copyback verification unless `--abandon` was explicit.
It waits for instance termination and every recorded volume's deletion, checks tagged
volumes, settles EBS and deletes the session/cleanup schedules. A failed operation
retains evidence/holds; reconcile and inspect before retrying.

**Operator**, after §11 verification:

```bash
launch terminate
# Read these exact experiment-owned names/IDs from the ledger config.
: "${KEY_NAME:?pretrain-token}" "${SG:?recorded security_group_id}"
aws ec2 delete-key-pair --key-name "$KEY_NAME" --region eu-central-1
aws ec2 delete-security-group --group-id "$SG" --region eu-central-1
guard reconcile
guard status > "$DEST/final-status.txt"
cp "$LEDGER" "$DEST/cost-ledger-final.json"
```

Verify deletion of the **recorded** volumes and both kinds of schedule; an empty
attachment query alone is insufficient. Task 14 records `terminated_at` and removes
the EBS hold after confirmed deletion; it does **not** implement a `storage.deleted`
field. Retain its AWS evidence and those actual ledger fields. `init --create-role`
also leaves an experiment IAM role/policy: after schedules are confirmed gone, remove
only that recorded role's `pretrain-cleanup` inline policy and role via the account's
normal cleanup procedure, retaining evidence.

Final reconciled committed cost must be ≤ **$50**, with any $47 enforcement overrun
reported independently. Update `spend.json`/the report with final copy-back and teardown
costs, retain all earlier snapshots, and record any remaining resources or uncertainty.
Claude reviews actual artifacts and obtains an independent read-only evidence review;
report the comparison, leakage verdict, external diagnostic, spend and residual events
to the owner. No fev success threshold or post-hoc configuration selection is permitted.
