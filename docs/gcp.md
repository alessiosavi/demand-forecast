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

Copy `scripts/gcp/operator-env.example.sh` outside the
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
# Repeat for each additional boot disk type; --usd is USD/GiB-month for a disk key.
GCP_HYPERDISK_USD_PER_GB_MONTH='REPLACE_FROM_CATALOGUE'
guard price set hyperdisk-balanced --usd "$GCP_HYPERDISK_USD_PER_GB_MONTH" \
  --source "$GCP_PRICE_SOURCE" --fetched-on "$GCP_PRICE_DATE"
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
rates. `init` sets the default boot disk type and its USD/GiB-month price. The
ledger's `disk_prices` table stores a positive price per disk type, with source
and fetch date when set via `price set <disk-type>`. For each disk type, use the
largest applicable regional price or separate ledgers. `launch --boot-disk-type`
overrides the default and refuses an unpriced type **before reserving money**.
Existing ledgers without this table still load: the old
`config.boot_disk_usd_per_gb_month` becomes the price of
`config.boot_disk_type`; the next write persists the table. Existing sessions keep
their recorded disk hourly rate; new sessions also snapshot the disk price.
For Hyperdisk, account for any separately billed provisioned IOPS/throughput with
conservative holds; the table covers capacity. Do not
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

## Boot disk and network compatibility

Verified in Google's public reference on 2026-11-13. These are boot-disk choices
and the VPC NIC requirements for the machine types used by this runbook:

| Machine family / shape | Boot disk types | VPC NIC |
| --- | --- | --- |
| G4 (`g4-standard-48`) | `hyperdisk-balanced` only; Persistent Disk is unsupported | `GVNIC` required |
| A3 High (`a3-highgpu-1g`) | `pd-balanced`, `pd-ssd`, `hyperdisk-balanced` | `GVNIC` required |
| A2 Standard / Ultra (`a2-highgpu-1g`, `a2-ultragpu-1g`) | `pd-standard`, `pd-balanced`, `pd-ssd` | `VIRTIO_NET` or `GVNIC` |

Sources: [G4 disk support](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines#g4-series),
[A3 disk support](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines#a3-vms),
[A2 disk support](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines#a2-vms),
and [gVNIC requirements](https://docs.cloud.google.com/compute/docs/networking/using-gvnic#when_to_use)
([machine generations](https://docs.cloud.google.com/compute/docs/machine-resource#vm_terminology)).
Generation 3 and later require gVNIC; A2 needs it for supported 50–100 Gbps
configurations. The two A2 one-GPU shapes above do not require those bandwidths.
A3 Mega/Edge have the same listed boot choices as High; A3 Ultra uses
`hyperdisk-balanced` for boot. Hyperdisk ML/Throughput/Extreme are data-disk
types, not boot choices; see [Hyperdisk limitations](https://docs.cloud.google.com/compute/docs/disks/hyperdisks#limitations).
Other variants, multi-NIC/RDMA topologies and images require their own checks.

Pass `launch --nic-type GVNIC` explicitly for G4 and A3, using an image with the
gVNIC driver and GVNIC guest feature. The option becomes
`--network-interface=nic-type=GVNIC` in the create argv (SDK 593's
`compute/instances/flags.py` and the [gVNIC create reference](https://docs.cloud.google.com/compute/docs/networking/using-gvnic#create_a_vm_with_gvnic_support)).
Omitting it leaves NIC selection to gcloud/image defaults. Both disk type and NIC
choice are recorded per launch and retained by resume. Old sessions without
these fields use the ledger's default disk type and omit the NIC override.
The guard passes the choices through; it does not maintain a hardware compatibility
database or automatically change disk/NIC settings based on the machine name.

For G4, store a `hyperdisk-balanced` capacity price first (the variable above must
be filled from catalogue evidence), then add these options to the launch example:
`--boot-disk-type hyperdisk-balanced --nic-type GVNIC`.

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
launch, even a stopped VM: close it first. `DONE` means that **its job URI** has
finished, rather than that the whole prefix has finished. Both launch and resume
refuse a DONE whose `job_uri` equals the job being started. Resume retains the
old session's job URI; it cannot start the next leg of a chain. Use a new label
and a different job URI with `launch --state-uri` for that.

A legacy DONE without a nonempty `job_uri` is conservatively treated as the same
job and blocks both commands and guest startup. The refusal explains recovery:
use a new state prefix, or verify which job completed and manually copy the legacy
DONE into `history/`, then remove its root object only after the copy succeeds,
before launching a different job on the shared checkpoints. Do not remove a DONE
to rerun a completed job accidentally. Give each leg a distinct job URI; changing
the contents at the same URI does not change its identity.

For example, after publishing two distinct job scripts, run a chain sequentially:

```bash
CHAIN_STATE="gs://$GCP_BUCKET/runs/chain-01/state"
guard launch chain-leg-1 --machine-type "$GCP_MACHINE" --zone "$GCP_ZONE" \
  --provisioning "$GCP_PROVISIONING" --hours 2 --boot-disk-size 100 \
  --image-family "$GCP_IMAGE_FAMILY" --image-project "$GCP_IMAGE_PROJECT" \
  --state-uri "$CHAIN_STATE" --job-uri "gs://$GCP_BUCKET/jobs/leg-1.sh"
guard status chain-leg-1
# Wait for leg-1's DONE, then verify/settle deletion of its VM and disk.
guard close chain-leg-1
guard launch chain-leg-2 --machine-type "$GCP_MACHINE" --zone "$GCP_ZONE" \
  --provisioning "$GCP_PROVISIONING" --hours 2 --boot-disk-size 100 \
  --image-family "$GCP_IMAGE_FAMILY" --image-project "$GCP_IMAGE_PROJECT" \
  --state-uri "$CHAIN_STATE" --job-uri "gs://$GCP_BUCKET/jobs/leg-2.sh"
guard status chain-leg-2
```

Add any hardware-specific disk/NIC options described above to each launch.
Leg 2 restores leg 1's final checkpoint from the same prefix; it archives leg 1's
DONE before writing its own STARTED. To continue an interrupted leg instead,
use `resume chain-leg-2 chain-leg-2-retry` after its VM is gone, keeping that
leg's job URI. Close and settle each session separately.

`launch --copy-interval SECONDS` sets the periodic state-copy cadence (whole seconds,
60–3600, default 600). The value is stored in the session's `launch.copy_interval`,
passed as `guard-copy-interval` in create metadata, and retained by `resume`.
Each cycle sleeps the interval after its preceding copy; copies never overlap.
Old ledger sessions without the option resume with 600 seconds.

Storage permission and transport errors are refusals, not absent markers.
`status` returns VM data and object metadata plus `job_uri` for each STARTED,
DONE, FAILED, TIMEOUT, PREEMPTED and HEARTBEAT (null for a legacy marker);
PREEMPTED also exposes `source=watcher` or `source=hook` (null for older markers).
The heartbeat update time shows checkpoint liveness. STARTED is informational and
never prevents a launch or resume.

Dry runs print exact JSON argv arrays, execute only read-only discovery, and
change neither resources nor ledger bytes (including its lock file). They validate
prices, cap, region and markers. The resulting cloud state can change before a
real command. `close --dry-run` shows planned stop/delete operations.

When create returns a CLI error (positive nonzero exit), the guard reads instance
and disk inventory twice, three seconds apart, then reads zonal insert operations
filtered by `targetLink` and `operationType=insert`. It cancels at zero **only if
both inventory reads are clean and no matching insert can still produce a VM**.
An insert in PENDING or RUNNING, or DONE without an error, retains the reservation.
DONE with an error and clean inventory proves rejection; a clean empty operations
list also permits cancellation of a returned create error. Missing recorded
operations, unreadable or partial operation results, unknown status/identity,
timeout, signal termination and unexpected responses retain the reservation.
The original error, timestamped inventory results and operation fields are saved
as `error`, `create_failure_inventory` and `create_failure_operations`; each read
is persisted before the next. Names and labels both count, even before disk labeling.
The label stays burned in every case. Error text alone never proves rejection.
Disk-label failure after a synchronous create also retains the reservation.
Use `status` and `sweep`, then close/settle. There are no automatic create retries.
For inventory and operation reads, any stderr output also prevents cancellation: SDK 593 can
return partial inventory with exit zero and a warning about an unreachable scope.
Evidence stores resource names, IDs, zones/regions, status and labels; account and
guest metadata fields are omitted.

For Flex-start only, `launch --request-valid-for 30m` lets Google wait for capacity
in the selected zone. The default `0s` requests immediately available capacity and
fails if it is unavailable; it does not queue or retry in another zone. Nonzero
waits must be 90 seconds–2 hours; `h`, `m`, `s` and combinations such as `1h2m3s`
are accepted. Resume retains this setting, including for older ledger entries
without it (which use `0s`). Flex-start always submits create with `--async`, using
a 300-second submission budget, and records an `open` session with `operation_name`
and `create_operation` immediately. It does not hold a CLI poll open throughout
the capacity wait or try to label a disk that might not exist yet; the tokenized
disk name keeps it discoverable. A subprocess timeout retains the full reservation
because creation might still succeed. Google's runtime limit starts at RUNNING; the guest's conservative
metadata deadline starts at request submission, so capacity waits consume its
training window and can exhaust it. `status` and `wait` never extend this metadata
deadline. For synchronous Spot/Standard create, a missing `lastStartTimestamp`
triggers one further describe with a 60-second budget. If that still provides no
start, the instance identity and reservation remain recorded, and settlement
requires operation history or manual evidence.

```bash
guard status flex-01
guard wait flex-01 --timeout 300 --poll-interval 5
guard wait flex-01 --timeout 0  # one observation
```

Flex-start status includes `queue.state`, the described instance and saved insert
operation evidence. States include `pending`, `running`, `failed`, `expired`,
and `unknown` when evidence is insufficient. Expiry is identified from the
operation error; elapsed laptop time alone does not prove Google rejected it.
Other observed instance states, such as `terminated`, are reported directly.
`wait` polls with separate bounded read calls, releasing the ledger lock between
polls. Its local timeout bounds polling, with up to 60 seconds per cloud read;
it exits 0 for running, 1 for a terminal/unknown state or local wait expiry, and
2 for a read/usage error. A local wait timeout never cancels or releases money.
Status/wait save observed instance identity and first start for later settlement.
An unresolved Flex-start insert also blocks resume or another launch on its state
prefix, even while inventory contains no VM, so two jobs cannot later share it.
For a failed or expired request with nothing created, `close` settles zero only
after a matching DONE-with-error insert and the same two clean inventory reads.
Missing operations, unfinished/successful inserts or uncertain inventory keep
`needs_settlement` and the full reservation.

`close` first stops a running VM, reads `lastStartTimestamp` and
`lastStopTimestamp` from `instances describe`, saves that evidence **before**
deleting, and verifies no matching instance or disk remains. Compute cost uses
that measured interval, rounded to a minimum minute; boot disk is conservatively
charged from `creationTimestamp` until verified deletion. A changed or missing original start timestamp
prevents automatic settlement because it could hide earlier runtime. Do not
restart an instance externally: create a new reserved session with resume instead.
See the [instance fields](https://docs.cloud.google.com/compute/docs/reference/rest/v1/instances).

When the guest is already `STOPPING`, or `TERMINATED` without its runtime
timestamps, `close` polls describe for up to **180 seconds**, every **5 seconds**,
and durably saves the timestamps before deleting. Configure this with
`guard close train-01 --runtime-timeout 180 --poll-interval 5`; a zero runtime
timeout skips polling. Each poll has a read timeout of at most 60 seconds and
the remaining wait budget. Read timeouts retry within that budget; expiry
proceeds to deletion and the operation-history fallback below. Dry runs do not wait. The polling budget does
not include the initial describe, stop, delete, or inventory calls.

When Google's deadline or Spot preemption already deleted the instance and there
is no saved runtime evidence, `close` queries `compute operations list` with
`--zones=<session zone> --filter=targetId=<recorded instance ID>`. It checks the
returned identity, zone and instance link. A single successful insert's `endTime`
(or the recorded first start) begins compute time; the earliest successful
`compute.instances.preempted`, `stop` or `delete` operation's `startTime` ends it.
Boot disk cost runs from the recorded creation time (or insert `insertTime`)
through the completed delete's `endTime`, rather than through today's close time.
For this explicit-delete path, a completed successful delete and verified absence
of both VM and disk are required.

A separate Spot DELETE path accepts **exactly one** successful DONE `insert` and
**exactly one** successful DONE `compute.instances.preempted`, with no `delete`.
The only additional operation types accepted are `setLabels` and
`simulateMaintenanceEvent`; all operations must be DONE without errors, match the
recorded instance ID/zone/link, and have consistent timestamps. Any unknown type,
second insert/preemption, start/resume/reset, stop, missing recorded start/creation,
or contradictory inventory keeps the reservation. Both the instance and the
session's tokenized boot disk must be absent in **two** clean inventory reads,
stored in `settlement_inventory`. Compute is charged from the recorded first start
to the preemption operation's **endTime** (minimum one minute), and disk from
recorded creation to that **same instant**. `settlement_evidence` states this rule.
This measured-history rule does not use the guest marker as billing evidence.
The ledger retains the operation fields used, omitting the operation's user field,
as `settlement_operations` and `runtime_evidence.operations`.

A guest power-off has its own narrow history rule: **exactly one** successful
DONE `insert`, **exactly one** successful DONE
`compute.instances.guestTerminate`, and optionally one successful DONE `delete`
whose `insertTime` is at or after the power-off's `endTime`. Only `setLabels` and
`simulateMaintenanceEvent` may accompany them. All operations must be DONE
without errors, match the recorded instance ID/zone/link, and have consistent
timestamps. The recorded first start and creation time are required; two
power-offs, a start/resume/reset/automatic restart, another stop or preemption,
or any unknown/failed operation keeps the reservation.

With a delete operation, compute runs from the recorded first start through the
delete's **endTime**, conservatively including the guest shutdown/deletion gap;
disk runs from recorded creation to that same verified deletion. Without a
delete operation, both instance and disk must be absent in **two** clean
inventory reads; compute and disk end at the guest power-off's **endTime**.
Any remaining instance or disk blocks settlement. `settlement_evidence` states
the chosen rule, and sanitized operation evidence is retained as above. A changed
start observed during polling, or a described stop earlier than a retained
matching guest power-off event, also keeps the reservation held.

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
An already-absent disk after its instance was deleted counts as successful cleanup.
Late resources belonging to a cancelled session have the distinct finding
`cancelled_session_resource`. `close <label>` on such a session durably reopens it
as `needs_settlement`, restoring the full reservation before attempting cleanup.
An enforcement/cap overrun is recorded in `reopened` and reported, rather than
refusing to account for resources that may already have spent money. Close then
uses the normal runtime/evidence rules; if those cannot establish cost, reconcile
with `--settled-usd` and `--evidence`. A cancelled session with no resources stays
settled. Sweep deletion itself never settles money.

## Guest contract and recovery

The guest reads `guard-label`, `guard-state-uri`, `guard-job-uri` and
`guard-reserved-end` from instance metadata. It reads DONE before restoring state
or fetching the job, powering off if it belongs to this job URI or has no job URI.
Before STARTED, it copies terminal markers DONE, FAILED, TIMEOUT and PREEMPTED to
`<state URI>/history/<UTC time>-<new label>-<MARKER>`, deleting each root object
only after that copy succeeds. This includes interruption/failure markers of the
same job when resuming; a matching or legacy DONE is never archived automatically.
An archive read/copy/delete failure powers off before state upload or job execution,
preserving every marker that was not successfully copied. STARTED and HEARTBEAT
are refreshed in place, rather than archived.
Every marker body records `label`, `job_uri`, `exit_status` and UTC `time`.
It then writes STARTED with its UTC boot time as well,
creating the prefix before restore: real `gcloud storage rsync` rejects a source
prefix with no objects. STARTED is excluded from restore and is not a terminal
marker. Other restore errors still fail closed without running the job.
Data in the state prefix is restored to `$RUN_DIR/state`;
markers, `history/` and logs are excluded. `RUN_DIR` defaults to `/var/lib/gcp-run/<label>`;
`STATE_URI` and `RUN_DIR` are exported to the Bash job. The job must reconstruct
itself from `$RUN_DIR/state` and write **only there** what it needs to continue.
Everything else, including Local SSD and the boot disk, is disposable.

Publish self-validating checkpoint generations using local atomic renames; verify
completeness on restore and fall back to an older valid generation. Object copies
are not a transactional multi-file snapshot. Do not rely on upload order or store
required resume data only in process memory. Jobs must handle signals and leave
usable checkpoints throughout training, not only on successful exit.

GNU timeout ends the job ten minutes before the conservative metadata deadline,
with TERM then KILL after 30 seconds. At the configured cadence the runner copies state
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
Interrupted jobs and periodic writers are terminated and reaped before a normal
final copy. A separate Python 3 monitor watches `instance/preempted` while the job
runs, polling at one-second intervals with bounded requests and reconnecting after
HTTP errors/timeouts. Only the literal `TRUE` response (with an optional newline)
triggers; `FALSE`, empty responses and transport failures never do.

The watcher owns a single preemption outcome, serialized against normal finish:

1. Publish PREEMPTED with `job_uri` and `source=watcher`, using the [Storage JSON media-upload API](https://docs.cloud.google.com/storage/docs/json_api/v1/objects/insert) and a metadata-server access token; fall back to `gcloud`
   within the **same 3 s** budget. Only HTTP 200 confirms the JSON upload. No credential is saved to disk or logged.
2. Cancel and reap any periodic copy, without signalling the job. Atomically write
   a fresh nonce plus newline to `$RUN_DIR/checkpoint-request`. Wait up to **8 s**
   (guest environment `GCP_CHECKPOINT_WAIT`, clamped to 0–8 s). A supporting job
   publishes an immutable checkpoint generation, then atomically replaces
   `$RUN_DIR/checkpoint-done` with that exact nonce. Stale acknowledgements do not
   count. Jobs ignoring these files simply continue running.
3. Copy regular state files in descending modification-time order, preserving
   relative paths, within **8 s total**. New checkpoint files precede old logs;
   this is best effort, never a transactional snapshot. Symlinks are skipped.
4. Copy the job log within **1 s**, then reserve up to **3 s** for completion
   reconciliation. If the job ended during saving and every file was copied with
   an unchanged final inventory, archive the event and publish DONE/FAILED/TIMEOUT;
   otherwise keep PREEMPTED. If publishing the completion marker and restoring
   PREEMPTED both fail, the event remains archived. Normal finish cannot race a
   second terminal upload.
5. Leave the job running. Sending TERM could interrupt a checkpoint write; the
   job must publish immutable generations by atomic rename. Google's imminent
   shutdown owns termination. The watcher never signals the job to checkpoint.

An independent **24 s wall-clock alarm**, including child-process cancellation
and directory traversal, bounds the entire save sequence to fit the 25 s window.
Every upload has a deadline and its own process group; stuck children are killed
and reaped. A job already finished when TRUE is observed takes the normal finish
path without being labelled PREEMPTED. PREEMPTED records an event and an
interrupted durable-state save; it cannot promise that all computation stopped.

`guest-stop.sh` remains installed as a **marker-only, 5 s best-effort fallback**
with `source=hook`; it starts no checkpoint request or state/log copy and cannot
replace a watcher marker. It is **not relied upon**: four measured shutdown hooks
on real Spot machines produced no marker, copy or log. Normal completion is
claimed locally before final copying so that the hook cannot mislabel power-off.

Power-off can leave a stopped instance and billable boot disk: stopping clears
Compute Engine's termination timestamp. The guest's final power-off is **not**
proof of deletion. Close promptly; sweep flags stopped instances. Google's run
limit bounds compute runtime even if the laptop disappears, but it does not
bound bucket storage or stopped/detached disk lifetime.

## Verified flag differences and sources

During worker implementation, real SDK calls were limited to `gcloud ... --help`;
workers made no account/API calls or cloud mutations. The operator's separate live
canary on 2026-11-13 covered the behaviours listed below. Offline fake-CLI tests cover
launch and resume for all three provisioning models. On 2026-11-13, SDK 593 help
and these official documents established:

- [`instances create`](https://docs.cloud.google.com/sdk/gcloud/reference/compute/instances/create):
  `--provisioning-model` accepts SPOT, STANDARD and **FLEX_START** (underscore).
  `--max-run-duration` accepts duration syntax; `--instance-termination-action=DELETE`
  applies to preemption and automatic time limits. The limit is relative to the
  last RUNNING transition, not request submission; stop/restart resets it.
  `--async` returns the insert operation without waiting for completion; SDK 593's
  create implementation returns operation JSON when `--format=json` is explicit.
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
  GA create-disk help lacks disk labels, so the guard adds them after synchronous
  creates and scans tokenized names for both synchronous and asynchronous creates.
- SDK stop help confirms `--discard-local-ssd=true` accepts an explicit boolean.
  SDK `operations/list.py` selects zonal operations with `--zones`; the generated
  v1 `Operation` message and [zoneOperations reference](https://docs.cloud.google.com/compute/docs/reference/rest/v1/zoneOperations)
  define `targetId`, `zone`, `targetLink`, `status`, `startTime`, `endTime` and
  `insertTime`, and name `insert`/`delete` operation types. `status` is PENDING,
  RUNNING or DONE; `error.errors` and `httpErrorStatusCode` describe failures.
  Target-name insert queries use `--zones` and a `targetLink`/`operationType` filter,
  so they work before an instance ID is available. The target filter matches the
  full project/zone/instance path suffix, independent of API URL hostname; its
  regex syntax was checked against the installed SDK's offline filter evaluator.
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
not required for it. The operator's live canary on **2026-11-13** covered:

- A first `g2-standard-8` Spot launch end to end: restore from an empty state
  prefix, heartbeat, DONE, guest power-off and close using instance timestamps.
- Spot stock-out cancellations and the `g4-standard-48` boot-disk-type rejection.
- A Flex-start request that remained PENDING after the CLI operation poll failed
  with HTTP 502; inventory showed the instance and its reservation was retained.

Four additional simulated-maintenance tests on 2026-11-13 used one
`a2-highgpu-1g` training VM and three `c3d-standard-16` canaries (UTC):

| Session | Event | Metadata TRUE | Guest shutdown | Preemption operation, insertTime → endTime |
| --- | --- | --- | --- | --- |
| l1-a1 | 10:06:48 | not watched | not observed | 10:08:41.780 → 10:08:53.216 |
| canary-p1 | 10:32:56 | not watched | 10:33:31.2 | 10:33:54.130 → 10:34:03.814 |
| canary-p2 | 10:47:06 | 10:47:15.3 | console not captured | 10:48:14.016 → 10:48:23.373 |
| canary-p3 | 10:54:07 | 10:54:12.3 | console not captured | 10:54:59.865 → 10:55:11.956 |

The flag was usable 5–9 s after the event; `maintenance-event` remained `NONE`.
Canary-p1's console shows the shutdown script discovered at 10:33:31.9 and ends
mid-line at 10:33:32.15. Even an instrumented first-action 0.13 s JSON upload
failed to appear, including with cached DNS. A job-side poller could upload at
TRUE. Measured guest times: tiny `gcloud cp` 0.9 s, 190 MB `cp` 1.66 s, small
`rsync` 1.84 s, JSON media upload with metadata token 0.13 s (HTTP 200).

All four histories contained insert, optional harmless operations and successful
`compute.instances.preempted`, **no delete**. Instance and auto-delete boot disk
subsequently disappeared. The former guard refused them; the operator settled
manually through preemption endTime. Resume of l1-a1 from the last bucket copy
worked, losing about 12 minutes with unsynchronized ten-minute checkpoints/copies.
Sanitized operation fixtures live in `tests/test_gcp/fixtures/spot_preemptions.json`.

Still unverified live: this new watcher's delivery timing, the nonce checkpoint
handshake, newest-first copy durability under real shutdown, and automatic
settlement of a new preemption. A canary must timestamp TRUE, check that a
watcher-sourced PREEMPTED precedes checkpoint-request and copies. Instrument the
canary job to log request/done nonces and UTC timestamps after atomic checkpoint
publication; the protocol files themselves are local, outside the copied state.
Confirm that the uploaded job log contains matching nonces, the fresh checkpoint
reaches the bucket before older files in that save, and resume validates that
exact checkpoint generation. After deletion, `close` must show two absent VM/disk
reads, machine/disk endTime at the preemption end, and the stated charge rule.
Also unverified: the asynchronous Flex-start submission/status/wait flow and
server deadline deletion. The pending-request
incident did not verify a clean-inventory polling failure. Fake-CLI tests cover
that case and the new accounting rules but cannot establish live control-plane
behaviour, checkpoint durability or billing accuracy.
