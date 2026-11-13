# Google Cloud GPU operator runbook

This tooling reserves local USD commitments before provisioning a VM and passes a
Google-enforced run limit on **every** create, including resume. It makes no paid
call until `launch`; `init`, `price set`, `hold set` and `report` are local. The
ledger does not discover prices or read your credits. A billing budget is an alert,
not a spending limit. Review the training protocol separately before provisioning.

## Prerequisites and scope

Use Python 3.10+, Bash, and Google Cloud SDK (help checked against the installed
SDK 593 on 2026-11-13). The operator authenticates with their own user login; do
not copy that login to guests. An administrator must enable Compute Engine and
Cloud Storage, attach billing with suitable credits, and grant the operator VM
create/read/stop/delete, zonal operation list/read, disk label/read/delete, service-account act-as, and bucket
metadata/object read permissions. Use a dedicated project to make reconciliation
simple. Creating prerequisites is an administrator action outside the guard.

Create a **single-region** bucket in the VM's region, with uniform bucket-level
access and suitable retention/lifecycle policy. Multi-region and dual-region
buckets are refused. All run state must use this bucket; keep job scripts in a
trusted bucket. Grant the VM's default Compute Engine service account object
read/write on this bucket and read on the job bucket. The guard supplies
`--scopes=storage-rw`; scopes alone do not grant IAM roles. It uses the project's
default service account. Do not grant the guest Compute Admin, billing, or access
to unrelated data. User credentials do not supply guest permissions.

Use a Linux Deep Learning VM image with the guest agent, Google Cloud CLI, curl,
GNU coreutils (`date`, `timeout`) and the required GPU drivers already installed.
The runner executes as root; the job URI is trusted executable code. Check image
and machine compatibility before launch; a rejected create is not retried.
Families move over time: use a controlled custom family if reproducibility needs
an immutable image. The guard records the requested image family and project; it does not resolve
or pin an immutable image.

Request GPU quotas for the selected region and provisioning model. The usual
names in the console are **GPUs (all regions)**, **NVIDIA H100 GPUs**, **NVIDIA
A100 GPUs**, **NVIDIA A100 80GB GPUs**, **NVIDIA L4 GPUs**, and **NVIDIA RTX PRO
6000 GPUs** (G4), with the corresponding **Preemptible** quotas for Spot and
Flex-start. Also check CPUs / preemptible CPUs, instances, persistent disk capacity,
and Local SSD capacity. Quota names and applicable pools vary by machine family;
confirm the exact displayed metric in the project's Quotas page. Quota does not
guarantee capacity. See [GPU quotas](https://docs.cloud.google.com/compute/resource-usage#gpu_quota)
and [GPU regions and zones](https://docs.cloud.google.com/compute/docs/gpus/gpu-regions-zones).

## Local ledger, prices, and holds

Copy [operator-env.example.sh](../scripts/gcp/operator-env.example.sh) outside the
checkout, replace every placeholder from evidence, and source it in Bash. Keep the
ledger on one durable local filesystem, outside Git. All operators must use the
same ledger and lock file; copies on separate laptops do not coordinate.

```bash
source /absolute/path/operator-env.sh
guard() { python scripts/gcp/gcp_guard.py --ledger "$GCP_LEDGER" "$@"; }
guard init --project "$GCP_PROJECT" --bucket "$GCP_BUCKET" \
  --currency USD --cap "$GCP_CAP_USD" --enforce "$GCP_ENFORCE_USD" \
  --boot-disk-type "$GCP_BOOT_DISK_TYPE" \
  --boot-disk-usd-per-gb-month "$GCP_BOOT_DISK_USD_PER_GB_MONTH"
guard price set "$GCP_MACHINE|$GCP_PROVISIONING|${GCP_ZONE%-*}" \
  --usd "$GCP_HOURLY_USD" --source "$GCP_PRICE_SOURCE" --fetched-on "$GCP_PRICE_DATE"
guard hold set storage 5.00
guard report
```

`init` refuses overwrite. Currency is USD only; amounts must be finite and
nonnegative, prices and limits positive, and enforce cannot exceed cap. Price keys
are `machine_type|SPOT|region`, `machine_type|STANDARD|region`, or
`machine_type|FLEX_START|region`. Add a price for **each** machine/model/region combination. Existing sessions
retain their price snapshot when prices change. Source and ISO fetch date are
mandatory; stale prices are the operator's responsibility.

The hourly rate must cover the **whole configuration**, including separately
billed accelerator(s) and Local SSD. The machine-only rate is insufficient when
those are extra. Use separate ledgers if configurations sharing a key need distinct
rates. One ledger has one boot disk type and USD/GiB-month price; use the largest
applicable regional price or separate ledgers. Check whether your selected machine
requires Hyperdisk instead of `pd-balanced`, particularly newer families. Do not
enter zero to omit a cost. Fixed costs, bucket storage, object versions, requests,
networking, logs and other services need conservative holds. Decreasing a hold is
an operator assertion that those costs have been reconciled, not an automatic
refund. `report` includes all historical settled costs and every current hold.

The reservation is `(hours + 0.25) × hourly price`, **plus boot disk for that same
window** (`size × monthly price / 730 × (hours + 0.25)`). Reserving the extra disk
margin is conservative. Decimal arithmetic rounds costs upward to a millionth of
a dollar; cap and enforce remain exact. The admission check and atomic,
fsync-backed ledger replacement share a POSIX file lock. A process exit between
reservation and create leaves a charged reservation for reconciliation. Never
remove or rename the lock file while an operator command is running.

## Launch, inspect, resume, close, sweep

Upload a trusted restartable job to `GCP_JOB_URI` separately. No example here
uploads or provisions prerequisites automatically. Start with a dry run:

```bash
guard launch train-01 --machine-type "$GCP_MACHINE" --zone "$GCP_ZONE" \
  --provisioning "$GCP_PROVISIONING" --hours 2 --boot-disk-size 100 \
  --image-family "$GCP_IMAGE_FAMILY" --image-project "$GCP_IMAGE_PROJECT" \
  --job-uri "$GCP_JOB_URI" --dry-run
# Repeat the same command without --dry-run after inspecting it.
guard status train-01
guard status
guard report
# After interruption, with the old VM absent:
guard resume train-01 train-02 --dry-run
guard resume train-01 train-02
# Optionally change only the duration: resume ... --hours 3
# Close completed/interrupted sessions, including the predecessor of a resume:
guard close train-01 --dry-run
guard close train-01
guard sweep
guard sweep --delete --dry-run
guard sweep --delete
```

Optional `--accelerator=type=nvidia-tesla-t4,count=1` and `--local-ssd 1` add explicit
resources. Integrated-GPU machine families normally do not need `--accelerator`.
Do not add Local SSD when the machine supplies it implicitly. No automatic format
or mount is performed. Close detects Local SSD from the described instance's
`disks[].type == "SCRATCH"`, including bundled G4 disks; only those instances are
stopped with `--discard-local-ssd=true`. Labels are single-use lowercase resource names (at most
34 characters with the tokenized disk name), including cancelled labels. State
URIs use a conservative character set without shell metacharacters, list separators
or globs. Subprocesses always receive argument lists, never a shell.

The default state prefix is `gs://<bucket>/runs/<label>/state`; `--state-uri` can
select another prefix in the same bucket. Resume copies the previous launch
configuration and reserves **new** money against the same prefix. It does not
release the old reservation. Any existing instance using that prefix blocks a new
launch, even a stopped VM: close it first. A `DONE` object blocks both launch and
resume. Storage permission and transport errors are refusals, not absent markers.
`status` returns VM data and object metadata for STARTED, DONE, FAILED, TIMEOUT,
PREEMPTED and HEARTBEAT; heartbeat update time shows checkpoint liveness. STARTED
is informational and never prevents a launch or resume. Old interruption
markers persist; their bodies identify their session and time.

Dry runs print exact JSON argv arrays, execute only read-only discovery, and
change neither resources nor ledger bytes (including its lock file). They validate
prices, cap, region and markers. The resulting cloud state can change before a
real command. `close --dry-run` shows planned stop/delete operations.

A definitive stock-out, quota or invalid-argument rejection is cancelled at zero
only after inventory confirms no matching instance or disk. An ambiguous CLI
failure or timeout might have created resources: its reservation stays and its
label is burned. Disk-label failure after create also retains the reservation.
Use `status` and `sweep`, then close/settle. There are no automatic create retries.

For Flex-start only, `launch --request-valid-for 30m` lets Google wait for capacity
in the selected zone. The default `0s` requests immediately available capacity and
fails if it is unavailable; it does not queue or retry in another zone. Nonzero
waits must be 90 seconds–2 hours; `h`, `m`, `s` and combinations such as `1h2m3s`
are accepted. Resume retains this setting, including for older ledger entries
without it (which use `0s`). The create subprocess budget is the wait plus 300
seconds. A subprocess timeout retains the full reservation because creation might
still succeed. Google's runtime limit starts at RUNNING; the guest's conservative
metadata deadline starts at request submission, so capacity waits consume its
training window and can exhaust it. After create, a missing `lastStartTimestamp`
triggers one further describe with a 60-second budget. If that still provides no
start, the instance identity and reservation remain recorded, and settlement
requires operation history or manual evidence.

`close` first stops a running VM, reads `lastStartTimestamp` and
`lastStopTimestamp` from `instances describe`, saves that evidence **before**
deleting, and verifies no matching instance or disk remains. Compute cost uses
that measured interval, rounded to a minimum minute; boot disk is conservatively
charged from `creationTimestamp` until verified deletion. A changed or missing original start timestamp
prevents automatic settlement because it could hide earlier runtime. Do not
restart an instance externally: create a new reserved session with resume instead.
See the [instance fields](https://docs.cloud.google.com/compute/docs/reference/rest/v1/instances).

When Google's deadline or Spot preemption already deleted the instance and there
is no saved runtime evidence, `close` queries `compute operations list` with
`--zones=<session zone> --filter=targetId=<recorded instance ID>`. It checks the
returned identity, zone and instance link. A single successful insert's `endTime`
(or the recorded first start) begins compute time; the earliest successful
`compute.instances.preempted`, `stop` or `delete` operation's `startTime` ends it.
Boot disk cost runs from the recorded creation time (or insert `insertTime`)
through the completed delete's `endTime`, rather than through today's close time.
A completed successful delete and verified absence of both VM and disk are required.
The ledger retains the operation fields used, omitting the operation's user field,
as `settlement_operations` and `runtime_evidence.operations`.

Missing/expired operations, failed or unfinished operations, invalid timestamps,
multiple inserts, or additional start/resume/reset operations retain the full
reservation and require manual reconciliation. Operations have limited retention;
close promptly. `close` exits nonzero with `needs_settlement` when history cannot
establish the lifetime. Neither a missing VM nor any guest-written marker proves
the billed cost. After reconciling actual billing/export or retained cloud evidence, use:

```bash
guard close train-01 --settled-usd 1.23 --evidence 'billing export row / saved evidence path'
```

The amount is the **total session cost**, including its disk. Evidence text and
amount are retained. An orphan disk blocks any settlement; reconcile/delete it
first. Cost above a reservation is recorded honestly and blocks new commitments
when enforce is exceeded. Manual settlement does not verify billing evidence for
you. Keep the original evidence with the ledger.

Run `sweep` at session start/end and regularly while any resource exists. It
reports unknown token-bearing instances/disks, resources of settled sessions,
overdue running VMs, stopped VMs needing cleanup, orphan disks, and open sessions
without VMs. Boot disks are named `gcp-<ledger token>-<label>` and labeled after
create: the name scan covers a crash before disk labeling. Findings always give
exit 1, even with `--delete`; rerun to check cleanup. Read/API errors give exit 2.
`--delete` prints findings before deleting their resources, without settling money.
It may destroy an overdue job's unflushed state; inspect the dry run first.

## Guest contract and recovery

The guest reads `guard-label`, `guard-state-uri`, `guard-job-uri` and
`guard-reserved-end` from instance metadata. It checks DONE before restoring state
or fetching the job. It then writes STARTED with its label and UTC boot time,
creating the prefix before restore: real `gcloud storage rsync` rejects a source
prefix with no objects. STARTED is excluded from restore and is not a terminal
marker. Other restore errors still fail closed without running the job.
Data in the state prefix is restored to `$RUN_DIR/state`;
markers and logs are excluded. `RUN_DIR` defaults to `/var/lib/gcp-run/<label>`;
`STATE_URI` and `RUN_DIR` are exported to the Bash job. The job must reconstruct
itself from `$RUN_DIR/state` and write **only there** what it needs to continue.
Everything else, including Local SSD and the boot disk, is disposable.

Publish self-validating checkpoint generations using local atomic renames; verify
completeness on restore and fall back to an older valid generation. Object copies
are not a transactional multi-file snapshot. Do not rely on upload order or store
required resume data only in process memory. Jobs must handle signals and leave
usable checkpoints throughout training, not only on successful exit.

GNU timeout ends the job ten minutes before the conservative metadata deadline,
with TERM then KILL after 30 seconds. Every ten minutes the runner copies state
and updates HEARTBEAT after a successful copy. Failed periodic copies are logged
and retried next round without killing training. At exit it attempts a final copy,
writes DONE/FAILED/TIMEOUT with exit status, uploads the per-session job log, then
powers off. A failed final copy cannot write DONE. Metadata/bootstrap failures
also power off; Google's limit protects the VM if the guest agent never starts.
Storage commands have separate bounded timeouts:

| Operation | Default | Guest environment override |
| --- | --- | --- |
| Restore | 1800 s | `GCP_RESTORE_TIMEOUT` |
| Job fetch | 120 s | `GCP_JOB_FETCH_TIMEOUT` |
| Periodic state copy | 540 s | `GCP_PERIODIC_COPY_TIMEOUT` |
| Final state copy | 480 s | `GCP_FINAL_COPY_TIMEOUT` |
| Marker read/write | 60 s | `GCP_MARKER_TIMEOUT` |
| Log upload | 60 s | `GCP_LOG_TIMEOUT` |

Overrides use GNU timeout durations, such as `2s` or `0.2s`, and must be supplied
in the guest's environment. The final copy, marker and log budgets total the
600-second final-copy window; process termination/escalation can consume extra
seconds, so this is best effort and does not replace Google's deadline.
Interrupted jobs and periodic writers
are terminated and reaped before the final copy. The entire preemption hook is
bounded to 25 seconds, including metadata and marker/log writes. It attempts one
copy and PREEMPTED; normal completion
is distinguished with a local finished flag. Spot's short notice cannot guarantee
a checkpoint, marker or log reaches storage.

Power-off can leave a stopped instance and billable boot disk: stopping clears
Compute Engine's termination timestamp. The guest's final power-off is **not**
proof of deletion. Close promptly; sweep flags stopped instances. Google's run
limit bounds compute runtime even if the laptop disappears, but it does not
bound bucket storage or stopped/detached disk lifetime.

## Verified flag differences and sources

Only `gcloud ... --help` was run against the real SDK; no account/API calls or
cloud mutations were made during implementation. Offline fake-CLI tests cover
both create paths for all three provisioning models. On 2026-11-13, SDK 593 help
and these official documents established:

- [`instances create`](https://docs.cloud.google.com/sdk/gcloud/reference/compute/instances/create):
  `--provisioning-model` accepts SPOT, STANDARD and **FLEX_START** (underscore).
  `--max-run-duration` accepts duration syntax; `--instance-termination-action=DELETE`
  applies to preemption and automatic time limits. The limit is relative to the
  last RUNNING transition, not request submission; stop/restart resets it.
- [Runtime limits](https://docs.cloud.google.com/compute/docs/instances/limit-vm-runtime):
  Standard and Spot support limits (legacy `--preemptible` does not). Public bounds
  are 30 seconds–120 days, and termination can start up to 30 seconds late. This
  runner requires **over ten minutes** so it can reserve its final-copy window.
- [Flex-start create](https://docs.cloud.google.com/compute/docs/instances/create-flex-start-vm):
  A2/G2/G4 standalone VMs use the same create API, plus
  `--request-valid-for-duration`. The guard defaults to `0s`, an immediate-capacity
  attempt with no retry; `--request-valid-for` enables Google's optional zonal wait
  of 90–7200 seconds. Runtime
  is 10 minutes–7 days. The guard restricts `a3-highgpu-1g` to Spot as dispatched;
  it does not implement A3 MIG/resize-request provisioning.
- SDK help confirms `--metadata-from-file=startup-script=...`, `--labels`,
  `--scopes=storage-rw`, `--maintenance-policy=TERMINATE` and
  `--no-restart-on-failure`. Guest write access needs the explicit write scope.
  `--local-ssd` takes a property list, e.g. `interface=NVME`; repeat for more disks.
  `--create-disk` explicitly sets the boot disk name, image, size, type,
  `boot=yes` and `auto-delete=yes`; GA has no `--boot-disk-name` flag.
  GA create-disk help lacks disk labels,
  so the guard adds them separately and scans tokenized names.
- SDK stop help confirms `--discard-local-ssd=true` accepts an explicit boolean.
  SDK `operations/list.py` selects zonal operations with `--zones`; the generated
  v1 `Operation` message and [zoneOperations reference](https://docs.cloud.google.com/compute/docs/reference/rest/v1/zoneOperations)
  define `targetId`, `zone`, `targetLink`, `status`, `startTime`, `endTime` and
  `insertTime`, and name `insert`/`delete` operation types.
  [Spot preemption history](https://docs.cloud.google.com/compute/docs/instances/create-use-spot#detect-preemption)
  explicitly documents `compute.instances.preempted` and its `startTime`.
  The operation type is a string, not an exhaustive SDK enum; `start`/`stop` are
  handled as instance operation names but their live returned strings remain
  a canary check.
- [Deep Learning VM images](https://docs.cloud.google.com/deep-learning-vm/docs/images):
  image project is `deeplearning-platform-release`. Current base/PyTorch families
  include `common-cu129-ubuntu-2204-nvidia-580` and
  `pytorch-2-9-cu129-ubuntu-2204-nvidia-580`. Family availability, driver and hardware
  compatibility were checked in documentation, not with live image discovery.

## Limits of protection

The hard cap is for commitments in **this local ledger at configured list rates**,
not a Google billing-account limit. External creates, other ledgers, tampered files,
price changes, wrong/missing prices, underestimated holds, credit eligibility,
taxes, currency conversion, disk/network/storage overruns, and infrastructure
outside the token/name scan are outside that cap. Keep an enforcement margin below
credits and use independent billing alerts. Do not put credentials or personal
account identifiers in job scripts, evidence text or logs.

The server limit is the final compute guard; guest copies and laptop uptime are
not required for it. Successful completion, checkpoint durability, capacity, IAM,
quota, actual billing accuracy, full guest-agent integration and real Spot recovery
still require a separately authorized canary. No live cloud canary was performed.
Specifically unverified until that canary: actual zonal operation fields/types and
their completeness for Spot and deadline deletion; first-prefix and resumed
restores with the real storage CLI; real server deadline deletion, including
auto-deletion of the boot disk. Fake-CLI tests do not establish these live behaviours.
