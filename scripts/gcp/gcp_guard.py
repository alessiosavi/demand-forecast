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
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path

MARKERS = ("STARTED", "DONE", "FAILED", "TIMEOUT", "PREEMPTED", "HEARTBEAT")
OPEN = {"reserved", "open", "needs_settlement"}
HERE = Path(__file__).resolve().parent
CENT = Decimal("0.000001")


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


def committed(ledger):
    return sum(
        (
            number(s["reservation_usd"] if s["settled_cost_usd"] is None else s["settled_cost_usd"])
            for s in ledger["sessions"]
        ),
        sum((number(h["usd"]) for h in ledger["holds"].values()), Decimal(0)),
    )


def validate(ledger):
    cfg = ledger["config"]
    if ledger["schema_version"] != 1 or cfg["currency"] != "USD":
        raise GuardError("unsupported ledger schema or currency")
    if not 0 < number(cfg["enforce_usd"]) <= number(cfg["cap_usd"]):
        raise GuardError("require 0 < enforce <= cap")
    if number(cfg["boot_disk_usd_per_gb_month"]) <= 0:
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

    def cloud(self, *args, mutate=False, absent=False, timeout=300):
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
                "boot_disk_usd_per_gb_month": str(number(args.boot_disk_usd_per_gb_month)),
                "boot_disk_type": args.boot_disk_type,
                "token": os.urandom(12).hex(),
                "created_at": stamp(now()),
            },
            "prices": {},
            "sessions": [],
            "holds": {},
        }
        self.save()
        self.report(args)

    def price(self, args):
        parts = args.key.split("|")
        if (
            len(parts) != 3
            or not re.fullmatch(r"[a-z][a-z0-9-]+", parts[0])
            or not re.fullmatch(r"[a-z]+-[a-z]+[0-9]+", parts[2])
        ):
            raise GuardError("price key must be machine_type|provisioning|region")
        parts[1] = model(parts[1])
        date.fromisoformat(args.fetched_on)
        if number(args.usd) <= 0 or not args.source.strip():
            raise GuardError("price and source are required and price must be positive")
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

    def inventory(self):
        # Include tokenized boot disks left before the post-create label call.
        resources = {}
        for kind in ("instances", "disks"):
            items = self.cloud("compute", kind, "list")
            if not isinstance(items, list):
                raise GuardError(f"invalid {kind} list; absence unverified")
            resources[kind] = [
                item
                for item in items
                if item.get("labels", {}).get("guard-ledger") == self.config["token"]
                or (
                    kind == "disks"
                    and item.get("name", "").startswith(f"gcp-{self.config['token']}-")
                )
            ]
        return resources

    def markers(self, session):
        return {
            marker: self.cloud(
                "storage", "objects", "describe", session["state_uri"] + "/" + marker, absent=True
            )
            for marker in MARKERS
        }

    def labels(self, session):
        return f"guard-ledger={self.config['token']},guard-session={session['label']}"

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
        wait_seconds = request_seconds(request_valid_for or "0s")
        key = f"{args.machine_type}|{args.provisioning}|{region}"
        if key not in self.data["prices"]:
            raise GuardError(f"no stored price for {key}")
        state_uri = args.state_uri or f"gs://{self.config['bucket']}/runs/{args.label}/state"
        if state_uri.split("/")[2] != self.config["bucket"]:
            raise GuardError("state URI must use the configured bucket")
        price = self.data["prices"][key]
        disk_rate = (
            Decimal(args.boot_disk_size) * number(self.config["boot_disk_usd_per_gb_month"]) / 730
        )
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
            self.cloud("storage", "objects", "describe", state_uri + "/DONE", absent=True)
            is not None
        ):
            raise GuardError("state prefix is DONE; refusing launch/resume")
        session = {
            "label": args.label,
            "state_uri": state_uri,
            "status": "reserved",
            "settled_cost_usd": None,
            "reservation_usd": reservation,
            "hourly_usd": price["hourly_usd"],
            "price": dict(price),
            "disk_hourly_usd": str(disk_rate),
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
            f"size={args.boot_disk_size}GB,type={self.config['boot_disk_type']},"
            f"image-family={args.image_family},image-project={args.image_project}",
        ]
        if args.provisioning == "FLEX_START":
            command.append(f"--request-valid-for-duration={request_valid_for or '0s'}")
        if args.accelerator:
            command.append(f"--accelerator={args.accelerator}")
        command.extend("--local-ssd=interface=NVME" for _ in range(args.local_ssd))
        self.data["sessions"].append(session)
        self.save()  # Lock covers admission and this durable reservation, before create.
        try:
            result = self.cloud(*command, mutate=True, timeout=wait_seconds + 300)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            session["error"] = str(getattr(exc, "stderr", "") or exc)
            self.save()
            # A rejected request with no resources is free. A transport timeout or
            # unknown error may have created resources; never silently free it.
            definitive = isinstance(exc, subprocess.CalledProcessError) and re.search(
                r"ZONE_RESOURCE_POOL_EXHAUSTED|QUOTA_EXCEEDED|Quota ['\"][^'\"]+['\"] exceeded|stock-out|INVALID_ARGUMENT|Invalid value",
                session["error"],
                re.I,
            )
            if definitive:
                remaining = self.inventory()
                if not any(v.get("name") == args.label for v in remaining["instances"]) and not any(
                    d.get("name") == session["boot_disk"] for d in remaining["disks"]
                ):
                    session.update(status="cancelled", settled_cost_usd="0.000000")
                    self.save()
            raise GuardError("create failed; " + session["error"]) from exc
        if not self.dry_run:
            if not isinstance(result, list) or len(result) != 1:
                raise GuardError("unexpected create response; reservation retained; run sweep")
            vm = result[0]
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

    def status(self, args):
        inventory = self.inventory()
        sessions = [self.session(args.label)] if args.label else self.data["sessions"]
        print(
            json.dumps(
                [
                    {
                        "session": session,
                        "instances": [
                            v for v in inventory["instances"] if v.get("name") == session["label"]
                        ],
                        "markers": self.markers(session),
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
                if item.get("labels", {}).get("guard-session") == session["label"]
                or item.get("name") in {session["label"], session["boot_disk"]}
            ]
            for kind, items in inventory.items()
        }

    def delete_resource(self, kind, item):
        location = "region" if item.get("region") else "zone"
        value = item.get(location, "").rsplit("/", 1)[-1]
        if not value:
            raise GuardError("resource location missing; refusing deletion")
        self.cloud("compute", kind, "delete", item["name"], f"--{location}={value}", mutate=True)

    def operations_evidence(self, session):
        """Recover a single boot's lifetime from retained zonal control-plane history."""
        zone = session["launch"]["zone"]
        operations = self.cloud(
            "compute",
            "operations",
            "list",
            f"--zones={zone}",
            f"--filter=targetId={session['instance_id']}",
        )
        if not isinstance(operations, list):
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
            or op.get("zone", "").rsplit("/", 1)[-1] != zone
            or not op.get("targetLink", "").endswith(target)
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
        ends = [
            op
            for op in operations
            if op.get("operationType") in {"compute.instances.preempted", "stop", "delete"}
        ]
        deletes = [op for op in ends if op["operationType"] == "delete"]
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

    def close(self, args):
        session = self.session(args.label)
        if session["status"] in {"closed", "cancelled"}:
            raise GuardError("session is already settled")
        if (args.settled_usd is None) != (args.evidence is None) or (
            args.evidence is not None and not args.evidence.strip()
        ):
            raise GuardError("manual settlement requires both --settled-usd and --evidence")
        resources = self.matches(self.inventory(), session)
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
            if (
                vm
                and vm.get("lastStartTimestamp")
                and vm.get("lastStopTimestamp")
                and vm.get("status") == "TERMINATED"
            ):
                start, stop = instant(vm["lastStartTimestamp"]), instant(vm["lastStopTimestamp"])
                if stop < start or stop > now() + timedelta(seconds=30):
                    raise GuardError("invalid runtime timestamps; reservation retained")
                # Repeated starts can hide earlier runtime. Only this guard's
                # first observed boot can be automatically settled.
                if not session.get("last_start") or start != instant(session["last_start"]):
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
        remaining = self.matches(self.inventory(), session)
        if any(remaining.values()):
            raise GuardError("instance or disk remains; reservation retained; use sweep")
        evidence = session.get("runtime_evidence")
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
            reason = evidence["source"] + "; boot disk through verified deletion (conservative)"
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
                if known is None:
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
                    findings.append({"finding": reason, "kind": kind, "name": item["name"]})
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
    price.add_argument("key")
    price.add_argument("--usd", required=True)
    price.add_argument("--source", required=True)
    price.add_argument("--fetched-on", required=True)
    launch = commands.add_parser("launch")
    launch.add_argument("label", type=name)
    for key in ("machine-type", "zone", "image-family", "image-project"):
        launch.add_argument("--" + key, required=True, type=name)
    launch.add_argument("--provisioning", required=True, type=model)
    launch.add_argument("--hours", required=True)
    launch.add_argument("--boot-disk-size", required=True, type=int)
    launch.add_argument("--job-uri", required=True, type=uri)
    launch.add_argument("--state-uri", type=uri)
    launch.add_argument("--accelerator")
    launch.add_argument("--local-ssd", type=int, default=0, choices=range(25))
    launch.add_argument(
        "--request-valid-for", help="Flex-start capacity wait: 0s (default), or 90s–2h"
    )
    launch.add_argument("--dry-run", action="store_true")
    status = commands.add_parser("status")
    status.add_argument("label", nargs="?", type=name)
    resume = commands.add_parser("resume")
    resume.add_argument("old_label", type=name)
    resume.add_argument("label", type=name)
    resume.add_argument("--hours")
    resume.add_argument("--dry-run", action="store_true")
    close = commands.add_parser("close")
    close.add_argument("label", type=name)
    close.add_argument("--settled-usd")
    close.add_argument("--evidence")
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


def main():
    args = parser().parse_args()
    try:
        dry_run = getattr(args, "dry_run", False)
        path = args.ledger.resolve()
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
