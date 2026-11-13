#!/usr/bin/env python3
"""Local USD commitment ledger for bounded, single-use Compute Engine sessions.

No SDK imports, shell commands, price discovery, automatic retries or credentials.
Every cloud subprocess uses argv and JSON formatting. See docs/gcp.md for limits.
"""

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path

MARKERS = ("STARTED", "DONE", "FAILED", "TIMEOUT", "PREEMPTED", "HEARTBEAT")
OPEN = {"reserved", "open", "needs_settlement"}
HERE = Path(__file__).resolve().parent
CENT = Decimal("0.000001")
OP_FIELDS = {
    "name",
    "id",
    "zone",
    "targetId",
    "targetLink",
    "operationType",
    "status",
    "insertTime",
    "startTime",
    "endTime",
    "error",
    "httpErrorStatusCode",
}


class GuardError(Exception):
    """A refusal that preserves existing commitments."""


def number(value):
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise GuardError("expected a finite nonnegative decimal") from exc
    if not result.is_finite() or result < 0 or isinstance(value, bool):
        raise GuardError("expected a finite nonnegative decimal")
    return result


def money(value):
    return str(number(value).quantize(CENT, rounding=ROUND_CEILING))


def now():
    return datetime.now(timezone.utc)


def stamp(value):
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def inventory_evidence(resources):
    fields = ("name", "id", "zone", "region", "status", "labels")
    return {
        kind: [{key: item[key] for key in fields if key in item} for item in items]
        for kind, items in resources.items()
    }


def instant(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise GuardError("timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def name(value):
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,39}", value) or value.endswith("-"):
        raise argparse.ArgumentTypeError("name must be 1–40 lowercase letters, digits or hyphens")
    return value


def uri(value):
    # These conservative characters also avoid metadata list and glob expansion.
    if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+/[A-Za-z0-9_./:@+=-]+", value):
        raise argparse.ArgumentTypeError(
            "expected a gs://bucket/path without wildcards or metacharacters"
        )
    if any(part in {".", ".."} for part in value[5:].split("/")):
        raise argparse.ArgumentTypeError("URI must not contain dot path components")
    return value.rstrip("/")


def model(value):
    result = value.upper().replace("-", "_")
    if result not in {"SPOT", "STANDARD", "FLEX_START"}:
        raise argparse.ArgumentTypeError("provisioning must be SPOT, STANDARD or FLEX_START")
    return result


def request_seconds(value):
    match = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", value)
    if not match or not any(part is not None for part in match.groups()):
        raise GuardError("request-valid-for must use hours, minutes and seconds, e.g. 1h2m3s")
    seconds = sum(int(part or 0) * scale for part, scale in zip(match.groups(), (3600, 60, 1)))
    if seconds != 0 and not 90 <= seconds <= 7200:
        raise GuardError("Flex-start wait must be 0s or between 90s and 2h")
    return seconds


def copy_interval(value):
    try:
        seconds = int(value)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("copy interval must be 60–3600 whole seconds") from exc
    if str(seconds) != str(value) or not 60 <= seconds <= 3600:
        raise argparse.ArgumentTypeError("copy interval must be 60–3600 whole seconds")
    return seconds


def committed(ledger):
    return sum(
        (
            number(s["reservation_usd"] if s["settled_cost_usd"] is None else s["settled_cost_usd"])
            for s in ledger["sessions"]
        ),
        sum((number(h["usd"]) for h in ledger["holds"].values()), Decimal(0)),
    )


def operation_failed(operation):
    error = operation.get("error")
    errors = error.get("errors") if isinstance(error, dict) else None
    code = operation.get("httpErrorStatusCode")
    return bool(
        (
            isinstance(errors, list)
            and errors
            and all(
                isinstance(item, dict) and isinstance(item.get("code"), str) and item["code"]
                for item in errors
            )
        )
        or (isinstance(code, int) and 400 <= code <= 599)
    )


def validate(ledger):
    cfg = ledger["config"]
    if ledger["schema_version"] != 1 or cfg["currency"] != "USD":
        raise GuardError("unsupported ledger schema or currency")
    if not 0 < number(cfg["enforce_usd"]) <= number(cfg["cap_usd"]):
        raise GuardError("require 0 < enforce <= cap")
    for disk_type, price in ledger["disk_prices"].items():
        name(disk_type)
        if number(price["usd_per_gb_month"]) <= 0:
            raise GuardError("boot disk price must be positive")
    if not re.fullmatch(r"[0-9a-f]{24}", cfg["token"]):
        raise GuardError("invalid ledger token")
    labels = set()
    for session in ledger["sessions"]:
        if session["label"] in labels or session["status"] not in OPEN | {"closed", "cancelled"}:
            raise GuardError("invalid or duplicate session")
        labels.add(session["label"])
        number(session["reservation_usd"])
        number(session["hourly_usd"])
        number(session["disk_hourly_usd"])
        if session["settled_cost_usd"] is not None:
            number(session["settled_cost_usd"])
        elif session["status"] in {"closed", "cancelled"}:
            raise GuardError("settled session has no cost")
    for price in ledger["prices"].values():
        if number(price["hourly_usd"]) <= 0:
            raise GuardError("hourly price must be positive")
    committed(ledger)


@contextmanager
def locked(path, dry_run):
    lock_path = path.with_name(path.name + ".lock")
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
    # A dry run never creates even a lock file. A real ledger always has one.
    with lock_path.open("rb" if dry_run else "a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


class Guard:
    def __init__(self, path, dry_run=False):
        self.path = path
        self.dry_run = dry_run
        self.data = json.loads(path.read_text()) if path.exists() else None
        if self.data is not None:
            if "disk_prices" not in self.data:
                cfg = self.data["config"]
                self.data["disk_prices"] = {
                    cfg["boot_disk_type"]: {
                        "usd_per_gb_month": cfg.pop("boot_disk_usd_per_gb_month"),
                        "source": "legacy ledger default",
                        "fetched_on": None,
                    }
                }
            validate(self.data)

    @property
    def config(self):
        if self.data is None:
            raise GuardError("initialise the ledger first")
        return self.data["config"]

    def save(self):
        validate(self.data)
        if self.dry_run:
            return
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, delete=False) as out:
                temporary = out.name
                json.dump(self.data, out, indent=2, allow_nan=False)
                out.write("\n")
                out.flush()
                os.fsync(out.fileno())
            os.replace(temporary, self.path)
            fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def cloud(self, *args, mutate=False, absent=False, timeout=300, clean_stderr=False, raw=False):
        command = [
            "gcloud",
            *args,
            f"--project={self.config['project']}",
            "--format=json",
            "--quiet",
        ]
        if self.dry_run:
            print(json.dumps(command))
            if mutate:
                return None
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as exc:
            text = exc.stderr or ""
            if (
                absent
                and not re.search(r"403|PERMISSION_DENIED|Permission denied", text)
                and re.search(r"\b404\b|\bNOT_FOUND\b|\bnot found\b|No such object", text)
            ):
                return None
            raise
        # Compute list can report partial failures as warnings with exit zero.
        if clean_stderr and result.stderr.strip():
            raise GuardError("inventory stderr; absence unverified: " + result.stderr.strip())
        # storage cat emits the object body even with --format=json.
        if raw:
            return result.stdout
        return json.loads(result.stdout) if result.stdout.strip() else None

    def init(self, args):
        if self.data is not None:
            raise GuardError("ledger already exists; refusing overwrite")
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,221}", args.bucket):
            raise GuardError("bucket must be a bucket name, without gs:// or a path")
        self.data = {
            "schema_version": 1,
            "config": {
                "project": args.project,
                "bucket": args.bucket,
                "currency": args.currency,
                "cap_usd": str(number(args.cap)),
                "enforce_usd": str(number(args.enforce)),
                "boot_disk_type": args.boot_disk_type,
                "token": os.urandom(12).hex(),
                "created_at": stamp(now()),
            },
            "prices": {},
            "disk_prices": {
                args.boot_disk_type: {
                    "usd_per_gb_month": str(number(args.boot_disk_usd_per_gb_month)),
                    "source": "init --boot-disk-usd-per-gb-month",
                    "fetched_on": None,
                }
            },
            "sessions": [],
            "holds": {},
        }
        self.save()
        self.report(args)

    def price(self, args):
        date.fromisoformat(args.fetched_on)
        if number(args.usd) <= 0 or not args.source.strip():
            raise GuardError("price and source are required and price must be positive")
        if "|" not in args.key:
            self.data["disk_prices"][name(args.key)] = {
                "usd_per_gb_month": str(number(args.usd)),
                "source": args.source,
                "fetched_on": args.fetched_on,
            }
            self.save()
            return
        parts = args.key.split("|")
        if (
            len(parts) != 3
            or not re.fullmatch(r"[a-z][a-z0-9-]+", parts[0])
            or not re.fullmatch(r"[a-z]+-[a-z]+[0-9]+", parts[2])
        ):
            raise GuardError("price key must be machine_type|provisioning|region")
        parts[1] = model(parts[1])
        self.data["prices"]["|".join(parts)] = {
            "hourly_usd": str(number(args.usd)),
            "source": args.source,
            "fetched_on": args.fetched_on,
        }
        self.save()

    def session(self, label):
        for session in self.data["sessions"]:
            if session["label"] == label:
                return session
        raise GuardError(f"unknown session: {label}")

    def inventory(self, session=None, include_all=False):
        # Include tokenized boot disks left before the post-create label call.
        resources = {}
        for kind in ("instances", "disks"):
            items = self.cloud("compute", kind, "list", clean_stderr=session is not None)
            if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
                raise GuardError(f"invalid {kind} list; absence unverified")
            resources[kind] = [
                item
                for item in items
                if include_all
                or item.get("labels", {}).get("guard-ledger") == self.config["token"]
                or (
                    session is not None
                    and item.get("name")
                    == (session["label"] if kind == "instances" else session["boot_disk"])
                )
                or (
                    kind == "disks"
                    and item.get("name", "").startswith(f"gcp-{self.config['token']}-")
                )
            ]
        return resources

    def marker(self, state_uri, key):
        marker_uri = state_uri + "/" + key
        metadata = self.cloud("storage", "objects", "describe", marker_uri, absent=True)
        if metadata is None:
            return None
        body = self.cloud("storage", "cat", marker_uri, raw=True)
        job_uri = next(
            (
                line.removeprefix("job_uri=")
                for line in body.splitlines()
                if line.startswith("job_uri=")
            ),
            None,
        )
        source = next((line[7:] for line in body.splitlines() if line.startswith("source=")), None)
        return dict(metadata, job_uri=job_uri or None, source=source)

    def markers(self, session):
        return {key: self.marker(session["state_uri"], key) for key in MARKERS}

    def labels(self, session):
        return f"guard-ledger={self.config['token']},guard-session={session['label']}"

    def insert_operations(self, session, evidence_key):
        """Read by target name even when an insert has not produced an instance ID."""
        zone = session["launch"]["zone"]
        target = f"/projects/{self.config['project']}/zones/{zone}/instances/{session['label']}"
        evidence = session[evidence_key] = {"observed_at": stamp(now())}
        try:
            operations = self.cloud(
                "compute",
                "operations",
                "list",
                f"--zones={zone}",
                f"--filter=targetLink~'{target}$' AND operationType=insert",
                clean_stderr=True,
                timeout=60,
            )
            if not isinstance(operations, list) or any(
                not isinstance(op, dict) for op in operations
            ):
                raise GuardError("invalid insert operation list; rejection unverified")
            evidence["operations"] = [
                {key: value for key, value in op.items() if key in OP_FIELDS} for op in operations
            ]
            if any(
                not all(
                    isinstance(op.get(key), str) for key in ("name", "zone", "targetLink", "status")
                )
                or not op["name"]
                or op.get("operationType") != "insert"
                or op.get("zone", "").rsplit("/", 1)[-1] != zone
                or not op.get("targetLink", "").endswith(target)
                or op.get("status") not in {"PENDING", "RUNNING", "DONE"}
                for op in operations
            ):
                raise GuardError("ambiguous insert operation identity or status")
            if session.get("operation_name") and not any(
                op["name"] == session["operation_name"] for op in operations
            ):
                raise GuardError("recorded insert operation missing; rejection unverified")
            return evidence["operations"]
        except (GuardError, ValueError, OSError, subprocess.SubprocessError) as exc:
            evidence["error"] = str(getattr(exc, "stderr", "") or exc)
            return None
        finally:
            self.save()

    def rejection_verified(self, session, require_operation=False):
        """Two clean absence reads AND no insert that could still create a VM."""
        reads = session["create_failure_inventory"] = []
        for attempt in range(2):
            if attempt:
                time.sleep(3)
            observed = {"observed_at": stamp(now())}
            try:
                observed.update(inventory_evidence(self.inventory(session)))
            except (GuardError, ValueError, OSError, subprocess.SubprocessError) as exc:
                observed["error"] = str(getattr(exc, "stderr", "") or exc)
            reads.append(observed)
            self.save()
        operations = self.insert_operations(session, "create_failure_operations")
        return (
            all(
                "error" not in read
                and not any(
                    self.matches(
                        {kind: read[kind] for kind in ("instances", "disks")}, session
                    ).values()
                )
                for read in reads
            )
            and operations is not None
            and (not require_operation or bool(operations))
            and all(op["status"] == "DONE" and operation_failed(op) for op in operations)
        )

    def launch(self, args, previous=None):
        if any(s["label"] == args.label for s in self.data["sessions"]):
            raise GuardError("label was used before; choose a new label")
        if not re.fullmatch(r"[a-z]+-[a-z]+[0-9]+-[a-z]", args.zone):
            raise GuardError("expected a zonal location, e.g. europe-west4-a")
        region = args.zone.rsplit("-", 1)[0]
        hours = number(args.hours)
        seconds = hours * 3600
        limit = 7 * 86400 if args.provisioning == "FLEX_START" else 120 * 86400
        if seconds != seconds.to_integral_value() or not 600 < seconds <= limit:
            raise GuardError(
                "runtime must be whole seconds, over 10 minutes, at most 7 days for Flex-start or 120 days otherwise"
            )
        if args.machine_type == "a3-highgpu-1g" and args.provisioning != "SPOT":
            raise GuardError("a3-highgpu-1g is restricted to SPOT by this runbook")
        request_valid_for = getattr(args, "request_valid_for", None)
        if request_valid_for is not None and args.provisioning != "FLEX_START":
            raise GuardError("--request-valid-for is only supported with FLEX_START")
        request_seconds(request_valid_for or "0s")
        interval = copy_interval(getattr(args, "copy_interval", 600))
        key = f"{args.machine_type}|{args.provisioning}|{region}"
        if key not in self.data["prices"]:
            raise GuardError(f"no stored price for {key}")
        state_uri = args.state_uri or f"gs://{self.config['bucket']}/runs/{args.label}/state"
        if state_uri.split("/")[2] != self.config["bucket"]:
            raise GuardError("state URI must use the configured bucket")
        price = self.data["prices"][key]
        disk_type = getattr(args, "boot_disk_type", None) or self.config["boot_disk_type"]
        if disk_type not in self.data["disk_prices"]:
            raise GuardError(f"no stored boot disk price for {disk_type}")
        disk_price = self.data["disk_prices"][disk_type]
        disk_rate = Decimal(args.boot_disk_size) * number(disk_price["usd_per_gb_month"]) / 730
        if args.boot_disk_size <= 0:
            raise GuardError("boot disk size must be positive")
        # The 15-minute margin includes boot disk provisioning/cleanup time too.
        reservation = money((hours + Decimal("0.25")) * (number(price["hourly_usd"]) + disk_rate))
        if committed(self.data) + number(reservation) > number(self.config["enforce_usd"]):
            raise GuardError("committed + reservation exceeds enforcement amount")
        bucket = self.cloud("storage", "buckets", "describe", f"gs://{self.config['bucket']}")
        if (
            bucket.get("location", "").lower() != region
            or bucket.get("location_type", bucket.get("locationType", "")).lower() != "region"
        ):
            raise GuardError("bucket must be a single-region bucket in the zone's region")
        inventory = self.inventory()
        for old in self.data["sessions"]:
            if old["state_uri"] == state_uri and any(
                v.get("name") == old["label"] for v in inventory["instances"]
            ):
                raise GuardError("state prefix already has an instance; close it before resuming")
            if (
                old["state_uri"] == state_uri
                and old["status"] in OPEN
                and old["launch"]["provisioning"] == "FLEX_START"
            ):
                operations = self.insert_operations(old, "queue_operations")
                if not operations or any(op["status"] != "DONE" for op in operations):
                    raise GuardError(
                        "state prefix has a pending or uncertain insert; close it before resuming"
                    )
        done = self.marker(state_uri, "DONE")
        if done is not None:
            if not done["job_uri"]:
                raise GuardError(
                    "DONE has no job URI (legacy format); refusing launch/resume. "
                    "Use a new state prefix, or verify the completed job and archive "
                    "the legacy DONE before launching a different job."
                )
            if done["job_uri"] == args.job_uri:
                raise GuardError("job URI is DONE; refusing launch/resume")
        session = {
            "label": args.label,
            "state_uri": state_uri,
            "status": "reserved",
            "settled_cost_usd": None,
            "reservation_usd": reservation,
            "hourly_usd": price["hourly_usd"],
            "price": dict(price),
            "disk_hourly_usd": str(disk_rate),
            "disk_price": dict(disk_price),
            "created_at": stamp(now()),
            "reserved_end": stamp(now() + timedelta(seconds=int(seconds))),
            "boot_disk": f"gcp-{self.config['token']}-{args.label}",
            "launch": {
                key: getattr(args, key)
                for key in (
                    "machine_type",
                    "zone",
                    "provisioning",
                    "hours",
                    "boot_disk_size",
                    "image_family",
                    "image_project",
                    "job_uri",
                    "accelerator",
                    "local_ssd",
                )
            },
        }
        session["launch"]["request_valid_for"] = request_valid_for
        session["launch"]["boot_disk_type"] = disk_type
        session["launch"]["nic_type"] = getattr(args, "nic_type", None)
        session["launch"]["copy_interval"] = interval
        if len(session["boot_disk"]) > 63:
            raise GuardError(
                "label is too long for tokenized boot disk name (maximum 34 characters)"
            )
        if previous:
            session["resumed_from"] = previous["label"]
        metadata = ",".join(
            f"guard-{key}={value}"
            for key, value in (
                ("label", args.label),
                ("state-uri", state_uri),
                ("job-uri", args.job_uri),
                ("reserved-end", session["reserved_end"]),
                ("copy-interval", interval),
            )
        )
        command = [
            "compute",
            "instances",
            "create",
            args.label,
            f"--zone={args.zone}",
            f"--machine-type={args.machine_type}",
            f"--provisioning-model={args.provisioning}",
            f"--max-run-duration={int(seconds)}s",
            "--instance-termination-action=DELETE",
            "--maintenance-policy=TERMINATE",
            "--no-restart-on-failure",
            f"--labels={self.labels(session)}",
            "--scopes=storage-rw",
            f"--metadata={metadata}",
            f"--metadata-from-file=startup-script={HERE / 'guest-run.sh'},shutdown-script={HERE / 'guest-stop.sh'}",
            f"--create-disk=name={session['boot_disk']},boot=yes,auto-delete=yes,"
            f"size={args.boot_disk_size}GB,type={disk_type},"
            f"image-family={args.image_family},image-project={args.image_project}",
        ]
        if args.provisioning == "FLEX_START":
            command.extend(["--async", f"--request-valid-for-duration={request_valid_for or '0s'}"])
        if args.accelerator:
            command.append(f"--accelerator={args.accelerator}")
        if session["launch"]["nic_type"]:
            command.append(f"--network-interface=nic-type={session['launch']['nic_type']}")
        command.extend("--local-ssd=interface=NVME" for _ in range(args.local_ssd))
        self.data["sessions"].append(session)
        self.save()  # Lock covers admission and this durable reservation, before create.
        try:
            result = self.cloud(*command, mutate=True, timeout=300)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            session["error"] = str(getattr(exc, "stderr", "") or exc)
            self.save()
            # Polling/transport errors can leave an insert queued with no VM yet.
            if (
                isinstance(exc, subprocess.CalledProcessError)
                and exc.returncode > 0
                and self.rejection_verified(session)
            ):
                session.update(status="cancelled", settled_cost_usd="0.000000")
                self.save()
            raise GuardError("create failed; " + session["error"]) from exc
        if self.dry_run and args.provisioning == "FLEX_START":
            return
        if not self.dry_run:
            if not isinstance(result, list) or len(result) != 1:
                raise GuardError("unexpected create response; reservation retained; run sweep")
            vm = result[0]
            if args.provisioning == "FLEX_START":
                target = (
                    f"/projects/{self.config['project']}/zones/{args.zone}/instances/{args.label}"
                )
                if (
                    not isinstance(vm, dict)
                    or not vm.get("name")
                    or vm.get("operationType") != "insert"
                    or vm.get("zone", "").rsplit("/", 1)[-1] != args.zone
                    or not vm.get("targetLink", "").endswith(target)
                ):
                    raise GuardError(
                        "unexpected async insert response; reservation retained; run sweep"
                    )
                session.update(
                    status="open",
                    operation_name=vm["name"],
                    create_operation={key: value for key, value in vm.items() if key in OP_FIELDS},
                )
                self.save()
                print(json.dumps(session))
                # There may be no disk to label until capacity is available.
                # Tokenized disk names already make that gap visible to sweep.
                return
            session["instance_id"] = vm.get("id")
            session["instance_created_at"] = vm.get("creationTimestamp")
            session["last_start"] = vm.get("lastStartTimestamp")
            session["status"] = "open"
            self.save()  # Keep identity durable even if the follow-up describe fails.
            if not session["last_start"]:
                try:
                    described = self.cloud(
                        "compute",
                        "instances",
                        "describe",
                        args.label,
                        f"--zone={args.zone}",
                        absent=True,
                        timeout=60,
                    )
                except (subprocess.SubprocessError, OSError):
                    described = None
                    session["start_time_error"] = "bounded follow-up describe failed"
                if described:
                    if session["instance_id"] and str(described.get("id")) != str(
                        session["instance_id"]
                    ):
                        raise GuardError(
                            "instance identity changed during follow-up describe; reservation retained"
                        )
                    session["instance_id"] = described.get("id")
                    session["instance_created_at"] = (
                        described.get("creationTimestamp") or session["instance_created_at"]
                    )
                    session["last_start"] = described.get("lastStartTimestamp")
            if session["last_start"]:
                session["reserved_end"] = stamp(
                    instant(session["last_start"]) + timedelta(seconds=int(seconds))
                )
            self.save()
        self.cloud(
            "compute",
            "disks",
            "add-labels",
            session["boot_disk"],
            f"--zone={args.zone}",
            f"--labels={self.labels(session)}",
            mutate=True,
        )
        if not self.dry_run:
            print(json.dumps(session))

    def resume(self, args):
        previous = self.session(args.old_label)
        launch = dict(previous["launch"], label=args.label, state_uri=previous["state_uri"])
        if args.hours is not None:
            launch["hours"] = args.hours
        self.launch(argparse.Namespace(**launch), previous=previous)

    def adopt(self, session, vm=None):
        """Recover an ambiguous insert without trusting a matching name alone."""
        if session["status"] not in OPEN or (
            session.get("instance_id") and not session.get("adoption")
        ):
            return
        zone = session["launch"]["zone"]
        if session["boot_disk"] != f"gcp-{self.config['token']}-{session['label']}":
            raise GuardError("reservation token mismatch; reservation retained")
        if vm and (
            vm.get("name") != session["label"]
            or vm.get("zone", "").rsplit("/", 1)[-1] != zone
            or vm.get("labels", {}).get("guard-ledger") != self.config["token"]
            or vm.get("labels", {}).get("guard-session") != session["label"]
            or not vm.get("id")
            or (session.get("instance_id") and str(vm["id"]) != str(session["instance_id"]))
        ):
            raise GuardError("instance ownership mismatch; reservation retained")
        if session.get("instance_id") and session.get("adoption"):
            # Initial adoption already proved the insert identity. Its operation
            # may expire long before the owned VM does; refresh from that VM
            # without making inspection or manual settlement depend on retention.
            if vm and vm.get("lastStartTimestamp"):
                started = vm["lastStartTimestamp"]
                try:
                    if not instant(session["instance_created_at"]) <= instant(started) <= now():
                        raise ValueError("invalid instance start")
                except (KeyError, TypeError, ValueError, GuardError) as exc:
                    raise GuardError(
                        "instance timestamps unverified; reservation retained"
                    ) from exc
                previous_start = session.get("last_start")
                if session["adoption"].get("charge_from_insert_request") or not previous_start:
                    session["last_start"] = started
                    session["adoption"]["charge_from_insert_request"] = False
                elif instant(started) != instant(previous_start):
                    session["runtime_uncertain"] = True
                self.save()
            return
        # Save the pre-existing evidence before the fresh query replaces anything.
        retained = session.get("create_failure_operations", {}).get("operations", [])
        if session.get("create_operation"):
            retained = [session["create_operation"]]
        operations = self.insert_operations(session, "adoption_operations")
        if not operations or len(operations) != 1:
            if vm:
                raise GuardError("insert identity unverified; reservation retained")
            return
        op = operations[0]
        if (
            op["status"] not in {"RUNNING", "DONE"}
            or operation_failed(op)
            or not op.get("targetId")
            or (vm and str(op["targetId"]) != str(vm["id"]))
            or (session.get("instance_id") and str(op["targetId"]) != str(session["instance_id"]))
        ):
            if vm:
                raise GuardError("insert identity or outcome unverified; reservation retained")
            return
        if not vm and not any(
            all(
                str(recorded.get(k)) == str(op.get(k))
                for k in ("name", "targetId", "targetLink", "zone", "insertTime")
            )
            for recorded in retained
        ):
            return  # No historical ownership correlation: keep the reservation held.
        try:
            requested = op["insertTime"]
            if (
                not instant(session["created_at"])
                <= instant(requested)
                <= min(now(), instant(session["reserved_end"]))
            ):
                raise ValueError("insert outside reservation")
            if not instant(requested) <= instant(op["startTime"]) <= now():
                raise ValueError("invalid insert start")
            if (
                op["status"] == "DONE"
                and not instant(op["startTime"]) <= instant(op["endTime"]) <= now()
            ):
                raise ValueError("invalid insert end")
            creation = vm.get("creationTimestamp") if vm else None
            started = vm.get("lastStartTimestamp") if vm else None
            if creation and not instant(session["created_at"]) <= instant(creation) <= now():
                raise ValueError("invalid instance creation")
            if started and not instant(creation or requested) <= instant(started) <= now():
                raise ValueError("invalid instance start")
        except (KeyError, TypeError, ValueError, GuardError) as exc:
            raise GuardError("insert timestamps unverified; reservation retained") from exc
        # Never infer a start from insert endTime: that could omit billable runtime.
        # Only a provisional request bound can be replaced by an actual start.
        previous_start = session.get("last_start")
        provisional = session.get("adoption", {}).get("charge_from_insert_request", False)
        if (
            started
            and previous_start
            and not provisional
            and instant(started) != instant(previous_start)
        ):
            session["runtime_uncertain"] = True
        recorded_start = (
            previous_start
            if previous_start and not provisional
            else started or previous_start or requested
        )
        session.update(
            instance_id=op["targetId"],
            instance_created_at=creation or session.get("instance_created_at") or requested,
            last_start=recorded_start,
            adoption={
                "observed_at": stamp(now()),
                "source": "owned instance + insert operation"
                if vm
                else "retained insert operation + tokenized reservation",
                "operation": op["name"],
                "charge_from_insert_request": not bool(started)
                and (
                    not session.get("last_start")
                    or session.get("adoption", {}).get("charge_from_insert_request", False)
                ),
            },
        )
        if session["status"] == "reserved":
            session["status"] = "open"
        self.save()  # Keep the original reserved_end: adoption never extends a deadline.

    def queue(self, session):
        """Observe Flex-start without waiting in gcloud or extending the guest deadline."""
        vm = self.cloud(
            "compute",
            "instances",
            "describe",
            session["label"],
            f"--zone={session['launch']['zone']}",
            absent=True,
            clean_stderr=True,
            timeout=60,
        )
        if session["status"] == "reserved" or session.get("adoption"):
            self.adopt(session, vm)
        elif vm:
            if (
                vm.get("name") != session["label"]
                or vm.get("zone", "").rsplit("/", 1)[-1] != session["launch"]["zone"]
                or (session.get("instance_id") and str(vm.get("id")) != str(session["instance_id"]))
            ):
                raise GuardError("instance identity changed; reservation retained")
            session["instance_id"] = vm.get("id")
            session["instance_created_at"] = session.get("instance_created_at") or vm.get(
                "creationTimestamp"
            )
            if not session.get("last_start") and vm.get("lastStartTimestamp"):
                session["last_start"] = vm["lastStartTimestamp"]
            self.save()
        operations = self.insert_operations(session, "queue_operations")
        result = {"state": "unknown", "instance": vm, "evidence": session["queue_operations"]}
        if vm and vm.get("status") == "RUNNING":
            result["state"] = "running"
        elif operations and len(operations) == 1:
            operation = operations[0]
            if operation["status"] == "DONE" and operation_failed(operation):
                result["state"] = (
                    "expired"
                    if re.search(
                        r"expir|deadline.?exceeded", json.dumps(operation.get("error", {})), re.I
                    )
                    else "failed"
                )
            elif operation["status"] in {"PENDING", "RUNNING"}:
                result["state"] = "pending"
            elif vm:
                result["state"] = vm.get("status", "unknown").lower()
        return result

    def status(self, args):
        sessions = [self.session(args.label)] if args.label else self.data["sessions"]
        inventory = self.inventory()
        instances = {
            session["label"]: [
                vm
                for vm in inventory["instances"]
                if vm.get("name") == session["label"]
                and vm.get("zone", "").rsplit("/", 1)[-1] == session["launch"]["zone"]
                and vm.get("labels", {}).get("guard-ledger") == self.config["token"]
                and vm.get("labels", {}).get("guard-session") == session["label"]
            ]
            for session in sessions
        }
        for session in sessions:
            if (
                session["launch"]["provisioning"] != "FLEX_START"
                and session["status"] in OPEN
                and (not session.get("instance_id") or session.get("adoption"))
            ):
                matches = instances[session["label"]]
                if len(matches) > 1:
                    raise GuardError("ambiguous instance identity; reservation retained")
                self.adopt(session, matches[0] if matches else None)
        print(
            json.dumps(
                [
                    {
                        "session": session,
                        "instances": instances[session["label"]],
                        "markers": self.markers(session),
                        "queue": self.queue(session)
                        if session["launch"]["provisioning"] == "FLEX_START"
                        else None,
                    }
                    for session in sessions
                ],
                indent=2,
            )
        )

    def matches(self, inventory, session):
        return {
            kind: [
                item
                for item in items
                if (
                    item.get("labels", {}).get("guard-session") == session["label"]
                    or item.get("name") in {session["label"], session["boot_disk"]}
                )
                # Unknown zones still block settlement; another zone does not.
                and item.get("zone", "").rsplit("/", 1)[-1] in {"", session["launch"]["zone"]}
            ]
            for kind, items in inventory.items()
        }

    def delete_resource(self, kind, item):
        location = "region" if item.get("region") else "zone"
        value = item.get(location, "").rsplit("/", 1)[-1]
        if not value:
            raise GuardError("resource location missing; refusing deletion")
        self.cloud(
            "compute",
            kind,
            "delete",
            item["name"],
            f"--{location}={value}",
            mutate=True,
            absent=True,
        )

    def operations_evidence(self, session):
        """Recover a single boot's lifetime from retained zonal control-plane history."""
        zone = session["launch"]["zone"]
        operations = self.cloud(
            "compute",
            "operations",
            "list",
            f"--zones={zone}",
            f"--filter=targetId={session['instance_id']}",
            clean_stderr=True,
        )
        if not isinstance(operations, list) or any(not isinstance(op, dict) for op in operations):
            return None
        # Retain only settlement fields, never the operation's user/account field.
        fields = {
            "name",
            "id",
            "zone",
            "targetId",
            "targetLink",
            "operationType",
            "status",
            "insertTime",
            "startTime",
            "endTime",
            "error",
            "httpErrorStatusCode",
        }
        session["settlement_operations"] = [
            {key: value for key, value in op.items() if key in fields} for op in operations
        ]
        self.save()
        target = f"/projects/{self.config['project']}/zones/{zone}/instances/{session['label']}"
        if not operations or any(
            str(op.get("targetId")) != str(session["instance_id"])
            or not isinstance(op.get("zone"), str)
            or op["zone"].rsplit("/", 1)[-1] != zone
            or not isinstance(op.get("targetLink"), str)
            or not op["targetLink"].endswith(target)
            or op.get("operationType")
            not in {
                "insert",
                "stop",
                "delete",
                "compute.instances.preempted",
                "compute.instances.guestTerminate",
                "setLabels",
                "simulateMaintenanceEvent",
            }
            for op in operations
        ):
            return None
        # An insert is the first boot; a later start/resume/reset can hide runtime.
        inserts = [op for op in operations if op.get("operationType") == "insert"]
        if len(inserts) > 1 or any(
            op.get("operationType")
            in {"start", "resume", "reset", "compute.instances.automaticRestart"}
            for op in operations
        ):
            return None
        if any(op.get("operationType") == "compute.instances.guestTerminate" for op in operations):
            return self.guest_termination_evidence(session, operations, inserts)
        ends = [
            op
            for op in operations
            if op.get("operationType") in {"compute.instances.preempted", "stop", "delete"}
        ]
        deletes = [op for op in ends if op["operationType"] == "delete"]
        if not deletes:
            # DELETE-on-preemption produces no delete operation on real Spot VMs.
            # This is a separate, deliberately narrow proof, not a general fallback.
            harmless = {"setLabels", "simulateMaintenanceEvent"}
            preemptions = [
                op for op in ends if op["operationType"] == "compute.instances.preempted"
            ]
            reads = session.get("settlement_inventory", [])
            if (
                session["launch"]["provisioning"] != "SPOT"
                or len(inserts) != 1
                or len(preemptions) != 1
                or any(
                    op.get("operationType")
                    not in harmless | {"insert", "compute.instances.preempted"}
                    for op in operations
                )
                or any(
                    op.get("status") != "DONE" or op.get("error") or op.get("httpErrorStatusCode")
                    for op in operations
                )
                or len(reads) != 2
                or any(read.get("instances") or read.get("disks") for read in reads)
            ):
                return None
            try:
                inserted, preempted = inserts[0], preemptions[0]
                start = session.get("last_start")
                creation = session.get("instance_created_at")
                end = preempted["endTime"]
                if (
                    not start
                    or not creation
                    or not (
                        instant(creation)
                        <= instant(start)
                        <= instant(preempted["startTime"])
                        <= instant(end)
                        <= now()
                        and instant(inserted["insertTime"])
                        <= instant(inserted["startTime"])
                        <= instant(inserted["endTime"])
                        <= instant(preempted["startTime"])
                        and (
                            instant(inserted["startTime"]) <= instant(start)
                            or (
                                session.get("adoption", {}).get("charge_from_insert_request")
                                and instant(start) == instant(inserted["insertTime"])
                            )
                        )
                    )
                ):
                    return None
                if any(
                    not instant(inserted["insertTime"])
                    <= instant(op["startTime"])
                    <= instant(op["endTime"])
                    <= instant(end)
                    for op in operations
                ):
                    return None
            except (KeyError, TypeError, ValueError, GuardError):
                return None
            return {
                "start": start,
                "stop": end,
                "creation": creation,
                "deleted": end,
                "billing_rule": "spot-preemption-end",
                "source": "compute.operations.list targetId + zone; Spot DELETE: "
                + (
                    "machine and disk charged from insert request (no instance start recorded)"
                    if session.get("adoption", {}).get("charge_from_insert_request")
                    else "machine from recorded start and disk from creation"
                )
                + " through preemption endTime; two clean instance/disk absence reads",
                "operations": session["settlement_operations"],
            }
        used = inserts + ends
        if len(deletes) != 1 or any(
            op.get("status") != "DONE" or op.get("error") or op.get("httpErrorStatusCode")
            for op in used
        ):
            return None
        try:
            start = session.get("last_start") or (inserts[0]["endTime"] if inserts else None)
            creation = session.get("instance_created_at") or (
                inserts[0]["insertTime"] if inserts else None
            )
            stop = min(ends, key=lambda op: instant(op["startTime"]))["startTime"]
            deleted = deletes[0]["endTime"]
            if (
                not creation
                or not start
                or not (
                    instant(creation)
                    <= instant(start)
                    <= instant(stop)
                    <= instant(deletes[0]["startTime"])
                    <= instant(deleted)
                    <= now()
                )
            ):
                return None
            if inserts and not instant(inserts[0]["startTime"]) <= instant(
                inserts[0]["endTime"]
            ) <= instant(stop):
                return None
        except (KeyError, ValueError, GuardError):
            return None
        return {
            "start": start,
            "stop": stop,
            "creation": creation,
            "deleted": deleted,
            "source": "compute.operations.list targetId + zone; first boot/termination and completed delete",
            "operations": [
                {key: value for key, value in op.items() if key in fields} for op in used
            ],
        }

    def guest_termination_evidence(self, session, operations, inserts):
        """Prove one guest power-off, conservatively including any later deletion."""
        guests = [
            op for op in operations if op["operationType"] == "compute.instances.guestTerminate"
        ]
        deletes = [op for op in operations if op["operationType"] == "delete"]
        reads = session.get("settlement_inventory", [])
        if (
            len(inserts) != 1
            or len(guests) != 1
            or len(deletes) > 1
            or any(
                op["operationType"]
                not in {
                    "insert",
                    "compute.instances.guestTerminate",
                    "delete",
                    "setLabels",
                    "simulateMaintenanceEvent",
                }
                or op.get("status") != "DONE"
                or op.get("error")
                or op.get("httpErrorStatusCode")
                for op in operations
            )
            or len(reads) != 2
            or any(reads[-1].get(kind) for kind in ("instances", "disks"))
            or (not deletes and any(read.get("instances") or read.get("disks") for read in reads))
        ):
            return None
        try:
            inserted, guest = inserts[0], guests[0]
            start, creation = session.get("last_start"), session.get("instance_created_at")
            end = deletes[0]["endTime"] if deletes else guest["endTime"]
            if (
                not start
                or not creation
                or not (
                    instant(creation)
                    <= instant(start)
                    <= instant(guest["startTime"])
                    <= instant(guest["endTime"])
                    <= instant(end)
                    <= now()
                    and instant(inserted["insertTime"])
                    <= instant(inserted["startTime"])
                    <= instant(inserted["endTime"])
                    <= instant(guest["startTime"])
                    and (
                        instant(inserted["startTime"]) <= instant(start)
                        or (
                            session.get("adoption", {}).get("charge_from_insert_request")
                            and instant(start) == instant(inserted["insertTime"])
                        )
                    )
                )
            ):
                return None
            if deletes and not (
                instant(guest["endTime"])
                <= instant(deletes[0]["insertTime"])
                <= instant(deletes[0]["startTime"])
                <= instant(end)
            ):
                return None
            if any(
                not instant(inserted["insertTime"])
                <= instant(op["startTime"])
                <= instant(op["endTime"])
                <= instant(end)
                or (
                    op.get("insertTime") is not None
                    and not instant(inserted["insertTime"])
                    <= instant(op["insertTime"])
                    <= instant(op["startTime"])
                )
                for op in operations
            ):
                return None
        except (KeyError, TypeError, ValueError, AttributeError, GuardError):
            return None
        return {
            "start": start,
            "stop": end,
            "creation": creation,
            "deleted": end,
            "billing_rule": "guest-termination-delete-end" if deletes else "guest-termination-end",
            "source": "compute.operations.list targetId + zone; single guestTerminate: "
            "machine from recorded start and disk from creation through "
            + (
                "completed delete endTime (conservative)"
                if deletes
                else "guestTerminate endTime; two clean instance/disk absence reads"
            ),
            "operations": session["settlement_operations"],
        }

    def wait_for_runtime(self, session, vm, args):
        """Wait only for an in-progress guest stop, never delaying deletion indefinitely."""
        deadline = time.monotonic() + args.runtime_timeout
        while vm:
            if session.get("instance_id") and str(vm.get("id")) != str(session["instance_id"]):
                raise GuardError("instance identity changed; refusing to settle")
            if (
                vm.get("lastStartTimestamp")
                and session.get("last_start")
                and instant(vm["lastStartTimestamp"]) != instant(session["last_start"])
            ):
                session["runtime_uncertain"] = True
                self.save()  # A later snapshot must not erase contradictory runtime evidence.
            if self.dry_run or not (
                vm.get("status") == "STOPPING"
                or (
                    vm.get("status") == "TERMINATED"
                    and not (vm.get("lastStartTimestamp") and vm.get("lastStopTimestamp"))
                )
            ):
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(args.poll_interval, remaining))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                vm = self.cloud(
                    "compute",
                    "instances",
                    "describe",
                    session["label"],
                    f"--zone={session['launch']['zone']}",
                    absent=True,
                    timeout=min(60, remaining),
                )
            except subprocess.TimeoutExpired:
                continue  # Retry only within the overall budget, then fall through to delete.
        return vm

    def close(self, args):
        if not 0 <= args.runtime_timeout < float("inf") or not 0 < args.poll_interval <= 60:
            raise GuardError(
                "close requires finite runtime-timeout >= 0 and 0 < poll-interval <= 60"
            )
        session = self.session(args.label)
        if session["status"] == "closed":
            raise GuardError("session is already settled")
        if (args.settled_usd is None) != (args.evidence is None) or (
            args.evidence is not None and not args.evidence.strip()
        ):
            raise GuardError("manual settlement requires both --settled-usd and --evidence")
        resources = self.matches(self.inventory(session), session)
        if session["status"] in OPEN and (
            not session.get("instance_id") or session.get("adoption")
        ):
            if len(resources["instances"]) > 1:
                raise GuardError("ambiguous instance identity; reservation retained")
            self.adopt(session, resources["instances"][0] if resources["instances"] else None)
            if resources["instances"] and not session.get("instance_id"):
                raise GuardError("instance ownership unverified; reservation retained")
        if session["status"] in OPEN:
            session["status"] = "needs_settlement"
            self.save()
        session["settlement_inventory"] = [
            {"observed_at": stamp(now()), **inventory_evidence(resources)}
        ]
        had_resources = any(resources.values())
        if session["status"] == "cancelled":
            if not had_resources:
                raise GuardError("cancelled session has no resources to re-open")
            session.update(status="needs_settlement", settled_cost_usd=None)
            total = committed(self.data)
            session["reopened"] = {
                "observed_at": stamp(now()),
                "reason": "cancelled_session_resource",
                "committed_usd": money(total),
                "enforce_overrun_usd": money(
                    max(Decimal(0), total - number(self.config["enforce_usd"]))
                ),
                "cap_overrun_usd": money(max(Decimal(0), total - number(self.config["cap_usd"]))),
            }
            self.save()  # Restore all money before stop/delete or any possible failure.
            if total > number(self.config["enforce_usd"]):
                print(
                    "gcp-guard: re-opened cancelled session; overrun "
                    + json.dumps(session["reopened"]),
                    file=sys.stderr,
                )
        if resources["instances"] and session.get("instance_id") and args.settled_usd is None:
            self.operations_evidence(session)
            if session["launch"]["provisioning"] == "SPOT" and any(
                op.get("operationType") == "compute.instances.preempted"
                for op in session.get("settlement_operations", [])
            ):
                raise GuardError(
                    "preemption history contradicts present instance; reservation retained; reconcile manually"
                )
        for item in resources["instances"]:
            vm = self.cloud(
                "compute",
                "instances",
                "describe",
                item["name"],
                f"--zone={session['launch']['zone']}",
                absent=True,
            )
            if vm is None:
                continue
            if session.get("instance_id") and str(vm.get("id")) != str(session["instance_id"]):
                raise GuardError("instance identity changed; refusing to settle")
            if vm.get("status") == "RUNNING":
                stop = [
                    "compute",
                    "instances",
                    "stop",
                    vm["name"],
                    f"--zone={session['launch']['zone']}",
                ]
                if any(disk.get("type") == "SCRATCH" for disk in vm.get("disks", [])):
                    stop.append("--discard-local-ssd=true")
                self.cloud(*stop, mutate=True)
                if not self.dry_run:
                    vm = self.cloud(
                        "compute",
                        "instances",
                        "describe",
                        item["name"],
                        f"--zone={session['launch']['zone']}",
                        absent=True,
                    )
            vm = self.wait_for_runtime(session, vm, args)
            if (
                vm
                and vm.get("lastStartTimestamp")
                and vm.get("lastStopTimestamp")
                and vm.get("status") == "TERMINATED"
            ):
                start, stop = instant(vm["lastStartTimestamp"]), instant(vm["lastStopTimestamp"])
                if stop < start or stop > now() + timedelta(seconds=30):
                    raise GuardError("invalid runtime timestamps; reservation retained")
                target = (
                    f"/projects/{self.config['project']}/zones/{session['launch']['zone']}"
                    f"/instances/{session['label']}"
                )
                guest_ends = [
                    op.get("endTime")
                    for op in session.get("settlement_operations", [])
                    if op.get("operationType") == "compute.instances.guestTerminate"
                    and op.get("status") == "DONE"
                    and not op.get("error")
                    and not op.get("httpErrorStatusCode")
                    and str(op.get("targetId")) == str(session.get("instance_id"))
                    and str(op.get("zone", "")).rsplit("/", 1)[-1] == session["launch"]["zone"]
                    and str(op.get("targetLink", "")).endswith(target)
                ]
                try:
                    guest_conflict = any(instant(end) > stop for end in guest_ends)
                except (TypeError, ValueError, AttributeError, GuardError):
                    guest_conflict = True
                # Repeated starts can hide earlier runtime. Only this guard's
                # first observed boot can be automatically settled.
                if (
                    not session.get("last_start")
                    or start != instant(session["last_start"])
                    or guest_conflict
                ):
                    session["runtime_uncertain"] = True
                else:
                    session["runtime_evidence"] = {
                        "start": vm["lastStartTimestamp"],
                        "stop": vm["lastStopTimestamp"],
                        "source": "instances.describe lastStartTimestamp/lastStopTimestamp",
                        "creation": vm["creationTimestamp"],
                    }
                self.save()  # Persist timestamp evidence before irreversible deletion.
            self.delete_resource("instances", item)
        if self.dry_run:
            return
        remaining = self.matches(self.inventory(session), session)
        session["settlement_inventory"].append(
            {"observed_at": stamp(now()), **inventory_evidence(remaining)}
        )
        self.save()
        if any(remaining.values()):
            raise GuardError("instance or disk remains; reservation retained; use sweep")
        evidence = session.get("runtime_evidence")
        if (
            not had_resources
            and not evidence
            and not session.get("last_start")
            and not session.get("instance_created_at")
            and not session.get("reopened")
            and args.settled_usd is None
        ):
            session["status"] = "needs_settlement"
            self.save()
            if self.rejection_verified(session, require_operation=True):
                session.update(
                    status="closed",
                    settled_cost_usd="0.000000",
                    closed_at=stamp(now()),
                    settlement_evidence="failed insert operation + two clean instance/disk absence reads",
                )
                self.save()
                print(json.dumps(session))
                return
        if not evidence and args.settled_usd is None and session.get("instance_id"):
            session["status"] = "needs_settlement"
            self.save()
            evidence = self.operations_evidence(session)
            if evidence:
                session["runtime_evidence"] = evidence
                self.save()
        if args.settled_usd is not None:
            cost, reason = money(args.settled_usd), args.evidence
        elif evidence and not session.get("runtime_uncertain"):
            seconds = max(
                60, (instant(evidence["stop"]) - instant(evidence["start"])).total_seconds()
            )
            deleted = instant(evidence["deleted"]) if evidence.get("deleted") else now()
            disk_seconds = max(0, (deleted - instant(evidence["creation"])).total_seconds())
            cost = money(
                Decimal(str(seconds)) / 3600 * number(session["hourly_usd"])
                + Decimal(str(disk_seconds)) / 3600 * number(session["disk_hourly_usd"])
            )
            reason = evidence["source"]
            if evidence.get("billing_rule") not in {"spot-preemption-end", "guest-termination-end"}:
                reason += "; boot disk through verified deletion (conservative)"
        else:
            session["status"] = "needs_settlement"
            self.save()
            raise GuardError(
                "instance gone or runtime unknown; reservation retained; close with --settled-usd and --evidence"
            )
        session.update(
            status="closed",
            settled_cost_usd=cost,
            settlement_evidence=reason,
            closed_at=stamp(now()),
        )
        self.save()
        print(json.dumps(session))

    def sweep(self, args):
        inventory = self.inventory()
        findings, targets = [], []
        for kind, items in inventory.items():
            for item in items:
                known = next(
                    (
                        s
                        for s in self.data["sessions"]
                        if item.get("name")
                        == (s["label"] if kind == "instances" else s["boot_disk"])
                        and item.get("zone", "").rsplit("/", 1)[-1] == s["launch"]["zone"]
                        and (
                            kind != "instances"
                            or not s.get("instance_id")
                            or str(item.get("id")) == str(s["instance_id"])
                        )
                    ),
                    None,
                )
                reason = None
                cancelled = next(
                    (
                        s
                        for s in self.data["sessions"]
                        if s["status"] == "cancelled"
                        and (
                            item.get("labels", {}).get("guard-session") == s["label"]
                            or item.get("name") in {s["label"], s["boot_disk"]}
                        )
                    ),
                    None,
                )
                if cancelled:
                    reason = "cancelled_session_resource"
                elif known is None:
                    reason = "unknown_resource"
                elif known["status"] not in OPEN:
                    reason = "settled_session_resource"
                elif kind == "instances" and item.get("status") in {"TERMINATED", "SUSPENDED"}:
                    reason = "stopped_instance"
                elif (
                    kind == "instances"
                    and item.get("status") == "RUNNING"
                    and now() > instant(known["reserved_end"])
                ):
                    reason = "past_reserved_end"
                elif kind == "disks" and not any(
                    v.get("name") == known["label"] for v in inventory["instances"]
                ):
                    reason = "orphan_disk"
                if reason:
                    finding = {"finding": reason, "kind": kind, "name": item["name"]}
                    if cancelled:
                        finding["label"] = cancelled["label"]
                    findings.append(finding)
                    targets.append((kind, item))
        for session in self.data["sessions"]:
            if session["status"] in OPEN and not any(
                v.get("name") == session["label"] for v in inventory["instances"]
            ):
                findings.append(
                    {
                        "finding": "missing_instance",
                        "label": session["label"],
                        "reservation_retained": True,
                    }
                )
        for finding in findings:
            print(json.dumps(finding), flush=True)
        if args.delete:
            for kind, item in targets:
                self.delete_resource(kind, item)
        return 1 if findings else 0

    def hold(self, args):
        old = self.data["holds"].get(args.name, {"usd": "0"})
        usd = money(args.usd)
        if committed(self.data) - number(old["usd"]) + number(usd) > number(
            self.config["enforce_usd"]
        ):
            raise GuardError("hold would exceed enforcement amount")
        self.data["holds"][args.name] = {"usd": usd, "updated_at": stamp(now())}
        self.save()

    def report(self, args):
        print(
            json.dumps(
                {
                    "currency": self.config["currency"],
                    "committed_usd": money(committed(self.data)),
                    "cap_usd": self.config["cap_usd"],
                    "enforce_usd": self.config["enforce_usd"],
                    "holds": self.data["holds"],
                    "sessions": self.data["sessions"],
                },
                indent=2,
            )
        )


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--ledger", type=Path, required=True)
    commands = result.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--project", required=True, type=name)
    init.add_argument("--bucket", required=True)
    init.add_argument("--cap", required=True)
    init.add_argument("--enforce", required=True)
    init.add_argument("--currency", choices=["USD"], default="USD")
    init.add_argument("--boot-disk-usd-per-gb-month", required=True)
    init.add_argument("--boot-disk-type", default="pd-balanced", type=name)
    price = (
        commands.add_parser("price")
        .add_subparsers(dest="price_command", required=True)
        .add_parser("set")
    )
    price.add_argument("key", help="machine_type|provisioning|region, or a boot disk type")
    price.add_argument(
        "--usd", required=True, help="USD/hour for a machine; USD/GiB-month for a disk"
    )
    price.add_argument("--source", required=True)
    price.add_argument("--fetched-on", required=True)
    launch = commands.add_parser("launch")
    launch.add_argument("label", type=name)
    for key in ("machine-type", "zone", "image-family", "image-project"):
        launch.add_argument("--" + key, required=True, type=name)
    launch.add_argument("--provisioning", required=True, type=model)
    launch.add_argument("--hours", required=True)
    launch.add_argument("--boot-disk-size", required=True, type=int)
    launch.add_argument("--boot-disk-type", type=name, help="default: ledger boot disk type")
    launch.add_argument("--nic-type", type=str.upper, choices=["GVNIC", "VIRTIO_NET"])
    launch.add_argument("--job-uri", required=True, type=uri)
    launch.add_argument("--state-uri", type=uri)
    launch.add_argument("--accelerator")
    launch.add_argument("--local-ssd", type=int, default=0, choices=range(25))
    launch.add_argument(
        "--copy-interval",
        type=copy_interval,
        default=600,
        help="state copy interval in seconds (60–3600; default: 600)",
    )
    launch.add_argument(
        "--request-valid-for", help="Flex-start capacity wait: 0s (default), or 90s–2h"
    )
    launch.add_argument("--dry-run", action="store_true")
    status = commands.add_parser("status")
    status.add_argument("label", nargs="?", type=name)
    wait = commands.add_parser("wait")
    wait.add_argument("label", type=name)
    wait.add_argument(
        "--timeout", type=float, default=300, help="local wait budget in seconds; 0 observes once"
    )
    wait.add_argument("--poll-interval", type=float, default=5)
    resume = commands.add_parser("resume")
    resume.add_argument("old_label", type=name)
    resume.add_argument("label", type=name)
    resume.add_argument("--hours")
    resume.add_argument("--dry-run", action="store_true")
    close = commands.add_parser("close")
    close.add_argument("label", type=name)
    close.add_argument("--settled-usd")
    close.add_argument("--evidence")
    close.add_argument(
        "--runtime-timeout",
        type=float,
        default=180,
        help="seconds to wait for guest-stop timestamps before deleting (default: 180; 0 disables)",
    )
    close.add_argument("--poll-interval", type=float, default=5)
    close.add_argument("--dry-run", action="store_true")
    sweep = commands.add_parser("sweep")
    sweep.add_argument("--delete", action="store_true")
    sweep.add_argument("--dry-run", action="store_true")
    hold = (
        commands.add_parser("hold")
        .add_subparsers(dest="hold_command", required=True)
        .add_parser("set")
    )
    hold.add_argument("name", type=name)
    hold.add_argument("usd")
    commands.add_parser("report")
    return result


def wait_for_session(path, args):
    if not 0 <= args.timeout < float("inf") or not 0 < args.poll_interval <= 60:
        raise GuardError("wait requires finite timeout >= 0 and 0 < poll-interval <= 60")
    deadline = time.monotonic() + args.timeout
    while True:
        # Release between polls so close and other operators can use the ledger.
        with locked(path, False):
            guard = Guard(path)
            session = guard.session(args.label)
            queue = guard.queue(session)
            print(json.dumps({"label": args.label, "queue": queue}), flush=True)
        if queue["state"] == "running":
            return 0
        if queue["state"] != "pending" or time.monotonic() >= deadline:
            return 1
        time.sleep(min(args.poll_interval, max(0, deadline - time.monotonic())))


def main():
    args = parser().parse_args()
    try:
        dry_run = getattr(args, "dry_run", False)
        path = args.ledger.resolve()
        if args.command == "wait":
            return wait_for_session(path, args)
        with locked(path, dry_run):
            guard = Guard(path, dry_run)
            if args.command != "init":
                guard.config
            return getattr(guard, args.command)(args) or 0
    except (
        GuardError,
        ValueError,
        KeyError,
        OSError,
        InvalidOperation,
        argparse.ArgumentTypeError,
        subprocess.SubprocessError,
    ) as exc:
        detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        print(f"gcp-guard: {detail}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
