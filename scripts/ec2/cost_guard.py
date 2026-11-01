#!/usr/bin/env python3
"""Operator-side, stdlib-only EC2 commitment ledger.

Run from a supervised terminal. The $3 margin does not bound simultaneous loss
of the operator host and guest grace between launch and scheduler creation.
Reconcile at least every 24 hours while experiment resources exist.

Transfer manifests are JSON: {"bytes": N} or {"files": [{"bytes": N, ...}]}.
All rates are ledger configuration; actual AWS billing is reconciled by record.
"""

import argparse
import fcntl
import ipaddress
import json
import math
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

AWS_TIMEOUT_S = 300
AWS_WAIT_TIMEOUT_S = 600


# Accounting. A pending transfer has a hold and zero settled cost, never two
# reservations. Historical settled sessions and manual adjustments remain here.
def committed(ledger):
    return math.fsum(
        [
            s["reservation_usd"] if s.get("settled_cost_usd") is None else s["settled_cost_usd"]
            for s in ledger["sessions"]
        ]
        + [h["usd"] for h in ledger["holds"].values()]
        + [t["cost_usd"] for t in ledger["transfers"]]
        + [item["usd"] for item in ledger.get("line_items", [])]
    )


def reservation_usd(minutes, rate):
    return (minutes + 20) / 60 * rate


def session_cost_usd(elapsed_seconds, rate):
    return max(60, elapsed_seconds) / 3600 * rate


def ebs_hold_usd(volume_gb, usd_per_gb_month, hours):
    value = Decimal(str(volume_gb)) * Decimal(str(usd_per_gb_month)) / 720 * Decimal(str(hours))
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_CEILING))


def egress_usd(byte_count, usd_per_gb):
    return byte_count / 1e9 * usd_per_gb


class GuardError(Exception):
    def __init__(self, message, code=2):
        super().__init__(message)
        self.code = code


def utc_now():
    return datetime.now(timezone.utc)


def stamp(value):
    if value.tzinfo is None:
        raise GuardError("clock must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def instant(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise GuardError("timestamps must specify UTC (Z or +00:00)")
    return parsed


def finite(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("expected a finite, nonnegative number")
    return result


def label(value):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,24}", value):
        raise argparse.ArgumentTypeError("label must be 1–24 letters, digits, '_' or '-'")
    return value


def run_command(cmd, timeout=None):
    env = dict(os.environ, AWS_DEFAULT_OUTPUT="json", AWS_PAGER="")
    return subprocess.run(
        cmd, check=True, capture_output=True, text=True, env=env, timeout=timeout
    ).stdout


def missing(exc):
    # A timeout, AccessDenied, or malformed response is never proof of absence.
    return isinstance(exc, subprocess.CalledProcessError) and any(
        code in (exc.stderr or "")
        for code in (
            "ResourceNotFoundException",
            "InvalidInstanceID.NotFound",
            "InvalidVolume.NotFound",
        )
    )


def ssh_failure(exc):
    """Return failure kind, detail and retryability without hiding guest replies."""
    output = (exc.stdout or "").strip()
    error = (exc.stderr or "").strip()
    detail = "\n".join(part for part in (output, error) if part) or f"SSH exit {exc.returncode}"
    lowered = error.lower()
    # arm-deadline.sh prints its mismatch to stdout and then exits nonzero.
    if "DEADLINE_MISMATCH" in detail:
        return "DEADLINE_MISMATCH", detail, False
    if re.fullmatch(r"DEADLINE_NOT_READY [^\r\n]+", output):
        return "BOOT_NOT_READY", detail, True
    if output:
        return "BAD_OUTPUT", detail, False
    if exc.returncode == 127 or (
        "arm-deadline.sh" in lowered
        and ("no such file" in lowered or "command not found" in lowered)
    ):
        return "BOOTSTRAP_NOT_FINISHED", detail, True
    if "host key verification failed" in lowered or "host identification has changed" in lowered:
        return "HOST_KEY_REFUSED", detail, True
    if "timed out" in lowered or "timeout" in lowered:
        return "CONNECTION_TIMEOUT", detail, True
    if "connection refused" in lowered:
        return "CONNECTION_REFUSED", detail, True
    if exc.returncode == 255:
        return "CONNECTION_ERROR", detail, True
    return "BAD_OUTPUT", detail, False


def deadline_reply_error(output, requested):
    if output.startswith("DEADLINE_MISMATCH"):
        return "DEADLINE_MISMATCH", output.strip()
    match = re.fullmatch(r"DEADLINE_SET ([^\r\n]+)\n?", output)
    if not match:
        return "BAD_OUTPUT", "expected exactly one DEADLINE_SET line"
    try:
        delta = (instant(match[1]) - requested).total_seconds()
    except (ValueError, GuardError) as exc:
        return "BAD_OUTPUT", str(exc)
    if not -60 <= delta <= 0:
        return "DEADLINE_MISMATCH", "armed deadline outside [-60, 0] seconds"
    return None


class Guard:
    def __init__(self, ledger_path, run=run_command, now=utc_now, sleep=time.sleep):
        self.ledger_path = Path(ledger_path)
        self.run = run
        self.now = now
        self.sleep = sleep
        self.ledger = None
        self.txn = None
        self.command = None

    @property
    def config(self):
        return self.ledger["config"]

    @property
    def rate(self):
        return self.config["compute_usd_per_h"] + self.config["ipv4_usd_per_h"]

    def validate(self):
        data = self.ledger
        if data["schema_version"] != 1:
            raise GuardError("unsupported ledger schema")
        amounts = [committed(data), self.config["cap_usd"], self.config["enforce_usd"]]
        amounts += [s["reservation_usd"] for s in data["sessions"]]
        amounts += [
            s["settled_cost_usd"] for s in data["sessions"] if s["settled_cost_usd"] is not None
        ]
        amounts += [h["usd"] for h in data["holds"].values()]
        amounts += [t["cost_usd"] for t in data["transfers"]]
        amounts += [item["usd"] for item in data["line_items"]]
        amounts += [
            self.config[k]
            for k in (
                "compute_usd_per_h",
                "ipv4_usd_per_h",
                "ebs_usd_per_gb_month",
                "egress_usd_per_gb",
                "volume_gb",
            )
        ]
        if any(not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in amounts):
            raise GuardError("invalid nonnegative ledger amounts")
        if self.config["enforce_usd"] > self.config["cap_usd"]:
            raise GuardError("enforcement limit exceeds cap")
        active = [s for s in data["sessions"] if s["status"] in ("pending", "open")]
        if len(active) > 1:
            raise GuardError("multiple active sessions: inspect ledger before proceeding")
        for session in data["sessions"]:
            if session["status"] in ("closed", "cancelled") and session["settled_cost_usd"] is None:
                raise GuardError("settled session has no cost")
        for transfer in data["transfers"]:
            if transfer["status"] == "pending" and transfer["hold"] not in data["holds"]:
                raise GuardError("pending transfer has no reservation")

    def save(self):
        self.validate()
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.ledger_path.parent, delete=False
            ) as stream:
                temporary = stream.name
                json.dump(self.ledger, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.ledger_path)
            directory = os.open(self.ledger_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def stage(self, name, txn=None):
        txn = txn or self.txn
        utc = stamp(self.now())
        entry = {"utc": utc, "command": self.command, "stage": name}
        if txn:
            txn["stage"] = name
            txn["stages"].append({"stage": name, "utc": utc})
            entry["txn_id"] = txn["id"]
        self.ledger["events"].append(entry)
        self.save()

    def transaction(self):
        self.txn = {
            "id": os.urandom(8).hex(),
            "command": self.command,
            "status": "pending",
            "stages": [],
        }
        self.ledger["transactions"].append(self.txn)
        return self.txn

    def finish(self):
        self.txn["status"] = "closed"
        self.stage("complete")

    def aws(self, service, operation, *args, stage=None):
        self.stage(stage or f"aws:{service}:{operation}")
        timeout = AWS_WAIT_TIMEOUT_S if (service, operation) == ("ec2", "wait") else AWS_TIMEOUT_S
        result = self.run(
            ["aws", service, operation, *args, "--region", self.config["region"]], timeout=timeout
        )
        return json.loads(result) if result.strip() else {}

    def admit(self):
        if self.ledger["flags"].get("EXPIRED_SESSION"):
            raise GuardError("EXPIRED_SESSION: reconcile before new commitments", 7)
        self.validate()
        if committed(self.ledger) > self.config["enforce_usd"]:
            raise GuardError("commitment would exceed enforcement limit")

    def active(self):
        return [s for s in self.ledger["sessions"] if s["status"] in ("open", "pending")]

    def obsolete_schedules(self):
        return [
            s
            for s in self.ledger["sessions"]
            if s["status"] in ("closed", "cancelled") and not s.get("schedule_deleted", False)
        ]

    def init(self, args):
        if args.market == "spot":
            raise GuardError("spot is not authorised for this attempt", 3)
        cleanup = instant(args.cleanup_by)
        if cleanup <= self.now():
            raise GuardError("cleanup-by must be in the future")
        if self.ledger is None:
            token = os.urandom(6).hex()
            self.ledger = {
                "schema_version": 1,
                "config": {
                    "cap_usd": args.cap,
                    "enforce_usd": args.enforce,
                    "compute_usd_per_h": args.price,
                    "ipv4_usd_per_h": args.ipv4_usd_per_h,
                    "ebs_usd_per_gb_month": args.ebs_usd_per_gb_month,
                    "egress_usd_per_gb": args.egress_usd_per_gb,
                    "volume_gb": args.volume_gb,
                    "type": args.type,
                    "market": args.market,
                    "region": args.region,
                    "token": token,
                    "experiment": "pretrain-" + token,
                    "role": "pretrain-cleanup-scheduler-" + token,
                    "cleanup_by": stamp(cleanup),
                    "created": stamp(self.now()),
                    "cleanup_schedule_name": f"pretrain-{token}-cleanup",
                    "known_hosts_path": str(self.ledger_path.parent.resolve() / "known_hosts"),
                    "ready": False,
                },
                "sessions": [],
                "holds": {},
                "transfers": [],
                "transactions": [],
                "line_items": [],
                "copyback_verified": None,
                "flags": {"EXPIRED_SESSION": False, "OVERDUE_STORAGE": False},
                "events": [],
            }
            self.ledger["holds"]["ebs"] = {"usd": self.storage_hold(cleanup), "kind": "ebs"}
            self.admit()
        elif self.config["ready"] or self.ledger["sessions"]:
            raise GuardError("ledger already initialised; cannot reset commitments")
        self.transaction()
        self.stage("reserved")
        role_name = self.config["role"]
        if args.create_role:
            trust = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "scheduler.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            }
            try:
                self.aws("iam", "get-role", "--role-name", role_name)
            except subprocess.CalledProcessError as exc:
                if "NoSuchEntity" not in (exc.stderr or ""):
                    raise
                self.aws(
                    "iam",
                    "create-role",
                    "--role-name",
                    role_name,
                    "--assume-role-policy-document",
                    json.dumps(trust),
                )
            policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["ec2:StopInstances", "ec2:TerminateInstances"],
                        "Resource": "arn:aws:ec2:" + self.config["region"] + ":*:instance/*",
                        "Condition": {
                            "StringEquals": {
                                "aws:ResourceTag/experiment": self.config["experiment"]
                            }
                        },
                    }
                ],
            }
            self.aws(
                "iam",
                "put-role-policy",
                "--role-name",
                role_name,
                "--policy-name",
                "pretrain-cleanup",
                "--policy-document",
                json.dumps(policy),
            )
        role = self.aws("iam", "get-role", "--role-name", role_name)["Role"]
        self.config["role_arn"] = role["Arn"]
        self.config["ready"] = True
        self.finish()

    def storage_hold(self, cleanup):
        hours = (
            cleanup + timedelta(hours=24) - instant(self.config["created"])
        ).total_seconds() / 3600
        return ebs_hold_usd(self.config["volume_gb"], self.config["ebs_usd_per_gb_month"], hours)

    def schedule(self, name, when, instance_id, terminate=False):
        return {
            "Name": name,
            "State": "ENABLED",
            "ScheduleExpression": "at(" + stamp(when)[:-1] + ")",
            "ScheduleExpressionTimezone": "UTC",
            "FlexibleTimeWindow": {"Mode": "OFF"},
            "Target": {
                "Arn": "arn:aws:scheduler:::aws-sdk:ec2:"
                + ("terminateInstances" if terminate else "stopInstances"),
                "RoleArn": self.config["role_arn"],
                "Input": json.dumps({"InstanceIds": [instance_id], "Force": True}),
                "RetryPolicy": {"MaximumRetryAttempts": 10, "MaximumEventAgeInSeconds": 3600},
            },
        }

    def session_schedule(self, session):
        return self.schedule(
            session["stop_schedule_name"],
            instant(session["deadline"]) + timedelta(minutes=20),
            session["instance_id"],
        )

    def cleanup_schedule(self, instance_id):
        return self.schedule(
            self.config["cleanup_schedule_name"],
            instant(self.config["cleanup_by"]),
            instance_id,
            terminate=True,
        )

    def read_schedule(self, expected):
        actual = self.aws("scheduler", "get-schedule", "--name", expected["Name"])
        for key in (
            "State",
            "ScheduleExpression",
            "ScheduleExpressionTimezone",
            "FlexibleTimeWindow",
        ):
            if actual.get(key) != expected[key]:
                raise GuardError(f"schedule {expected['Name']}: {key} mismatch")
        target = actual.get("Target", {})
        for key in ("Arn", "RoleArn", "RetryPolicy"):
            if target.get(key) != expected["Target"][key]:
                raise GuardError(f"schedule {expected['Name']}: target {key} mismatch")
        if json.loads(target.get("Input", "null")) != json.loads(expected["Target"]["Input"]):
            raise GuardError(f"schedule {expected['Name']}: target Input mismatch")
        return actual

    def write_schedule(self, operation, config):
        # Full UpdateSchedule writable configuration (excluding read-only ARN,
        # CreationDate and LastModificationDate); preserve nested Target intact.
        fields = {
            "ActionAfterCompletion": "action-after-completion",
            "ClientToken": "client-token",
            "Description": "description",
            "EndDate": "end-date",
            "FlexibleTimeWindow": "flexible-time-window",
            "GroupName": "group-name",
            "KmsKeyArn": "kms-key-arn",
            "Name": "name",
            "ScheduleExpression": "schedule-expression",
            "ScheduleExpressionTimezone": "schedule-expression-timezone",
            "StartDate": "start-date",
            "State": "state",
            "Target": "target",
        }
        argv = []
        for key, flag in fields.items():
            if key in config:
                value = config[key]
                argv.extend(
                    [
                        "--" + flag,
                        json.dumps(value) if isinstance(value, (list, dict)) else str(value),
                    ]
                )
        self.aws("scheduler", operation, *argv)

    def create_schedule(self, expected):
        self.write_schedule("create-schedule", expected)
        return self.read_schedule(expected)

    def delete_schedule(self, name):
        try:
            self.aws("scheduler", "delete-schedule", "--name", name)
        except subprocess.CalledProcessError as exc:
            if not missing(exc):
                raise

    def delete_session_schedule(self, session):
        self.delete_schedule(session["stop_schedule_name"])
        session["schedule_deleted"] = True
        self.stage("schedule_deleted")

    def discover(self, session=None):
        if session and session.get("instance_id"):
            args = ["--instance-ids", session["instance_id"]]
        elif session and self.session_txn(session)["command"] == "provision":
            args = ["--filters", "Name=client-token,Values=" + self.config["token"]]
        elif self.ledger.get("instance_id"):
            args = ["--instance-ids", self.ledger["instance_id"]]
        else:
            args = ["--filters", "Name=client-token,Values=" + self.config["token"]]
        try:
            response = self.aws("ec2", "describe-instances", *args)
        except subprocess.CalledProcessError as exc:
            if missing(exc):
                return []
            raise
        return [i for r in response["Reservations"] for i in r["Instances"]]

    def session_txn(self, session):
        return next(t for t in self.ledger["transactions"] if t["id"] == session["txn_id"])

    def remember_instance(self, session, instance):
        session["instance_id"] = instance["InstanceId"]
        session["public_ip"] = instance.get("PublicIpAddress")
        self.ledger["instance_id"] = instance["InstanceId"]
        self.ledger["volume_ids"] = list(
            dict.fromkeys(
                self.ledger.get("volume_ids", [])
                + [
                    b["Ebs"]["VolumeId"]
                    for b in instance.get("BlockDeviceMappings", [])
                    if "Ebs" in b
                ]
            )
        )

    def volumes(self):
        response = self.aws(
            "ec2",
            "describe-volumes",
            "--filters",
            "Name=tag:experiment,Values=" + self.config["experiment"],
        )
        return [v for v in response["Volumes"] if v["State"] != "deleted"]

    def arm(self, session):
        try:
            requested = instant(session["deadline"])
            txn = self.session_txn(session)
            called = instant(txn["aws_called"])
            address = str(ipaddress.IPv4Address(session["public_ip"]))
        except (KeyError, TypeError, ValueError, GuardError) as exc:
            raise GuardError(f"deadline arming failed: {exc}", 4) from exc
        cutoff = min(requested, called + timedelta(minutes=15))
        known_hosts = self.config.setdefault(
            "known_hosts_path", str(self.ledger_path.parent.resolve() / "known_hosts")
        )
        cmd = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "UserKnownHostsFile=" + known_hosts,
        ]
        if self.config.get("key_path"):
            cmd += ["-i", self.config["key_path"]]
        cmd += [
            "ubuntu@" + address,
            "sudo",
            "/usr/local/sbin/arm-deadline.sh",
            "set",
            session["deadline"],
        ]
        attempts = session.setdefault("arm_attempts", {"count": 0, "last_error": None, "utc": None})
        while self.now() < cutoff:
            attempts["count"] += 1
            attempts["utc"] = stamp(self.now())
            # One transaction stage; every attempt durably updates its counter
            # and timestamp before SSH, including after crash recovery.
            if not any(s["stage"] == "deadline_armed" for s in txn["stages"]):
                self.stage("deadline_armed", txn)
            else:
                self.save()
            try:
                remaining = (cutoff - self.now()).total_seconds()
                if remaining <= 0:
                    break
                output = self.run(cmd, timeout=min(120, remaining))
            except subprocess.TimeoutExpired as exc:
                kind, detail, retry = "COMMAND_TIMEOUT", str(exc), True
            except subprocess.CalledProcessError as exc:
                kind, detail, retry = ssh_failure(exc)
            except OSError as exc:
                kind, detail, retry = "SSH_ERROR", str(exc), False
            else:
                failure = deadline_reply_error(output, requested)
                if failure is None and self.now() >= cutoff:
                    failure = "READINESS_EXPIRED", "SSH reply arrived after the arming cutoff"
                if failure is None:
                    session["armed_at"] = stamp(self.now())
                    session["status"] = "open"
                    txn["status"] = "open"
                    self.stage("open", txn)
                    return
                kind, detail = failure
                retry = False
            attempts["last_error"] = {"kind": kind, "detail": detail}
            self.save()
            print(f"arming attempt {attempts['count']}: {kind}: {detail}")
            if not retry:
                raise GuardError(f"deadline arming failed: {kind}", 4)
            remaining = (cutoff - self.now()).total_seconds()
            if remaining > 0:
                self.sleep(min(20, remaining))
        raise GuardError(f"deadline arming exhausted at {stamp(cutoff)}", 4)

    def prepare_launch(self, args):
        required = ("OPERATOR_IP", "VPC_ID", "SUBNET_ID", "AMI_ID")
        for name in required:
            if not os.environ.get(name):
                raise GuardError(f"missing environment variable {name}")
        operator_ip = str(ipaddress.IPv4Address(os.environ["OPERATOR_IP"]))
        user_data = Path(args.user_data).resolve()
        if not user_data.is_file():
            raise GuardError(f"missing Task 13 user data: {user_data}")
        key_name = "pretrain-" + self.config["token"]
        key_path = self.ledger_path.parent.resolve() / (key_name + ".pem")
        keys = self.aws(
            "ec2", "describe-key-pairs", "--filters", "Name=key-name,Values=" + key_name
        )["KeyPairs"]
        if keys:
            if not key_path.is_file() or not key_path.stat().st_size:
                raise GuardError(f"existing key pair requires its private key at {key_path}")
        else:
            # Exclusive creation avoids overwriting another operator's private key.
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                key = self.aws("ec2", "create-key-pair", "--key-name", key_name)["KeyMaterial"]
                stream.write(key)
                stream.flush()
                os.fsync(stream.fileno())
        os.chmod(key_path, 0o600)
        self.config["key_path"] = str(key_path)
        groups = self.aws(
            "ec2",
            "describe-security-groups",
            "--filters",
            "Name=group-name,Values=" + key_name,
            "Name=vpc-id,Values=" + os.environ["VPC_ID"],
        )["SecurityGroups"]
        if groups:
            if len(groups) != 1:
                raise GuardError("ambiguous security group")
            group = groups[0]
            permissions = group.get("IpPermissions", [])
            if len(permissions) != 1:
                raise GuardError("existing security group must allow only operator SSH")
            permission = permissions[0]
            if (
                (permission.get("IpProtocol"), permission.get("FromPort"), permission.get("ToPort"))
                != ("tcp", 22, 22)
                or [r["CidrIp"] for r in permission.get("IpRanges", [])] != [operator_ip + "/32"]
                or permission.get("Ipv6Ranges")
                or permission.get("UserIdGroupPairs")
                or permission.get("PrefixListIds")
            ):
                raise GuardError("existing security group has unexpected ingress")
            group_id = group["GroupId"]
        else:
            group_id = self.aws(
                "ec2",
                "create-security-group",
                "--group-name",
                key_name,
                "--description",
                "pretrain operator SSH only",
                "--vpc-id",
                os.environ["VPC_ID"],
            )["GroupId"]
            self.aws(
                "ec2",
                "authorize-security-group-ingress",
                "--group-id",
                group_id,
                "--protocol",
                "tcp",
                "--port",
                "22",
                "--cidr",
                operator_ip + "/32",
            )
        self.config["security_group_id"] = group_id
        image = self.aws("ec2", "describe-images", "--image-ids", os.environ["AMI_ID"])["Images"]
        if len(image) != 1 or not image[0].get("RootDeviceName"):
            raise GuardError("AMI has no unambiguous root device")
        disk = [
            {
                "DeviceName": image[0]["RootDeviceName"],
                "Ebs": {
                    "VolumeSize": self.config["volume_gb"],
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ]
        tags = [
            {
                "ResourceType": resource,
                "Tags": [{"Key": "experiment", "Value": self.config["experiment"]}],
            }
            for resource in ("instance", "volume")
        ]
        return [
            "--image-id",
            os.environ["AMI_ID"],
            "--instance-type",
            self.config["type"],
            "--count",
            "1",
            "--client-token",
            self.config["token"],
            "--key-name",
            key_name,
            "--security-group-ids",
            group_id,
            "--subnet-id",
            os.environ["SUBNET_ID"],
            "--associate-public-ip-address",
            "--tag-specifications",
            json.dumps(tags),
            "--block-device-mappings",
            json.dumps(disk),
            "--user-data",
            "file://" + str(user_data),
            "--instance-initiated-shutdown-behavior",
            "stop",
        ]

    def begin_session(self, args):
        if self.active():
            raise GuardError("an open or pending session already exists", 6)
        if self.obsolete_schedules():
            raise GuardError("obsolete stop schedule pending; run reconcile")
        if not self.config["ready"]:
            raise GuardError("init must confirm the scheduler role first")
        if args.minutes <= 0:
            raise GuardError("minutes must be positive")
        if any(s["label"] == args.label for s in self.ledger["sessions"]):
            raise GuardError("session labels cannot be reused")
        deadline = self.now() + timedelta(minutes=args.minutes)
        if deadline + timedelta(minutes=20) >= instant(self.config["cleanup_by"]):
            raise GuardError("session backstop must precede cleanup-by")
        if self.command == "provision" and (
            self.ledger.get("instance_id")
            or any(s.get("aws_called") for s in self.ledger["sessions"])
        ):
            raise GuardError("launch already attempted; use reconcile, then start")
        if self.command == "start" and not self.ledger.get("instance_id"):
            raise GuardError("no known instance; provision or reconcile first")
        consume = getattr(args, "consume_hold", None)
        if consume:
            self.release_hold(consume)
        self.transaction()
        session = {
            "label": args.label,
            "txn_id": self.txn["id"],
            "status": "pending",
            "reservation_usd": reservation_usd(args.minutes, self.rate),
            "settled_cost_usd": None,
            "instance_id": self.ledger.get("instance_id"),
            "public_ip": None,
            "aws_called": None,
            "deadline": stamp(deadline),
            "stop_schedule_name": f"pretrain-{self.config['token']}-stop-{args.label}",
            "stop_schedule_verified": False,
        }
        self.ledger["sessions"].append(session)
        self.admit()
        self.stage("reserved")
        try:
            if self.command == "provision":
                argv = self.prepare_launch(args)
                session["aws_called"] = stamp(self.now())
                self.txn["aws_called"] = session["aws_called"]
                response = self.aws("ec2", "run-instances", *argv, stage="aws_called")
                if len(response["Instances"]) != 1:
                    raise GuardError("launch returned an ambiguous instance set; reconcile")
                self.remember_instance(session, response["Instances"][0])
                self.stage("instance_discovered")
                self.create_schedule(self.session_schedule(session))
                session["stop_schedule_verified"] = True
                self.stage("stop_schedule_verified")
                self.create_schedule(self.cleanup_schedule(session["instance_id"]))
                self.stage("schedules_created")
            else:
                instances = self.discover(session)
                if len(instances) != 1 or instances[0]["State"]["Name"] != "stopped":
                    raise GuardError("start requires a confirmed stopped instance; reconcile")
                session["confirmed_stopped_before_start"] = True
                self.remember_instance(session, instances[0])
                self.create_schedule(self.session_schedule(session))
                session["stop_schedule_verified"] = True
                self.read_schedule(self.cleanup_schedule(session["instance_id"]))
                self.stage("schedules_created")
                session["aws_called"] = stamp(self.now())
                self.txn["aws_called"] = session["aws_called"]
                self.aws(
                    "ec2",
                    "start-instances",
                    "--instance-ids",
                    session["instance_id"],
                    stage="aws_called",
                )
            self.aws("ec2", "wait", "instance-running", "--instance-ids", session["instance_id"])
            for _ in range(12):
                instances = self.discover(session)
                if len(instances) == 1 and instances[0].get("PublicIpAddress"):
                    self.remember_instance(session, instances[0])
                    break
            else:
                raise GuardError("public IP did not become available", 4)
            self.arm(session)
        except (
            GuardError,
            OSError,
            ValueError,
            KeyError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            # BaseException intentionally escapes: a dead process cannot promise a
            # stop. The durable reservation then belongs to reconcile / shell trap.
            if session.get("instance_id"):
                try:
                    if not session.get("aws_called") and not session.get(
                        "confirmed_stopped_before_start"
                    ):
                        # A failed pre-start discovery must not make an external
                        # boot's runtime free. Use the last known stopped time.
                        session["aws_called"] = self.unreserved_origin()
                        self.txn["aws_called"] = session["aws_called"]
                    self.stop_session(session, cancelled=True)
                except (
                    GuardError,
                    OSError,
                    ValueError,
                    KeyError,
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                ) as stop_error:
                    print(f"stop unconfirmed; reservation retained: {stop_error}")
            raise exc

    def stop_time(self, session, instance):
        """Only a stopped-state transition dates a stop; termination is not evidence."""
        state = instance["State"]["Name"]
        reason = (instance.get("StateTransitionReason") or "") if state == "stopped" else ""
        match = re.search(r"\((\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) GMT\)", reason)
        called = self.session_txn(session).get("aws_called", session.get("aws_called"))
        evidenced = None
        if state == "stopped" and match:
            evidenced = match[1].replace(" ", "T") + "Z"
        elif state == "terminated" and session.get("stop_time_estimated") is False:
            evidenced = session.get("stopped_at")
        if evidenced:
            try:
                stopped = instant(evidenced)
                if stopped <= self.now() and (not called or stopped >= instant(called)):
                    if state == "stopped":
                        session["stop_time_evidence"] = reason
                    return stopped, False
            except ValueError:
                pass
        # Stop at deadline+20, up to 60 minutes of delivery retries, then 5 minutes to stop.
        return min(self.now(), instant(session["deadline"]) + timedelta(minutes=85)), True

    def update_stop_time_flag(self):
        self.ledger["flags"]["STOP_TIME_ESTIMATED"] = any(
            s.get("stop_time_estimated", False) for s in self.ledger["sessions"]
        )

    def settle_session(self, session, cancelled=False, instance=None):
        if session["settled_cost_usd"] is None:
            called = self.session_txn(session).get("aws_called", session.get("aws_called"))
            stopped, estimated = (
                self.stop_time(session, instance) if instance is not None else (self.now(), False)
            )
            session["settled_cost_usd"] = (
                session_cost_usd((stopped - instant(called)).total_seconds(), self.rate)
                if called
                else 0.0
            )
            session["stopped_at"] = stamp(stopped)
            session["stop_time_estimated"] = estimated and bool(called)
            if session["stop_time_estimated"]:
                session["observed_at"] = stamp(self.now())
            self.update_stop_time_flag()
            session["status"] = "cancelled" if cancelled else "closed"
            session["schedule_deleted"] = False
            txn = self.session_txn(session)
            txn["status"] = session["status"]
            self.stage(session["status"], txn)

    def stop_session(self, session, cancelled=False, instance=None):
        if instance is None:
            try:
                instances = self.discover(session)
            except (
                OSError,
                ValueError,
                KeyError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ):
                # Failed accounting discovery must not prevent stopping a known
                # instance. Keep its reservation until stop-time evidence can
                # be recovered; the waiter alone cannot date an earlier stop.
                if session.get("instance_id"):
                    self.aws(
                        "ec2", "stop-instances", "--instance-ids", session["instance_id"], "--force"
                    )
                    self.aws(
                        "ec2", "wait", "instance-stopped", "--instance-ids", session["instance_id"]
                    )
                raise
            if len(instances) != 1:
                print("instance absent or ambiguous; pending reservation retained")
                return
            instance = instances[0]
        self.remember_instance(session, instance)
        if instance["State"]["Name"] in ("stopped", "stopping", "terminated"):
            if instance["State"]["Name"] == "stopping":
                self.aws(
                    "ec2", "wait", "instance-stopped", "--instance-ids", session["instance_id"]
                )
                instances = self.discover(session)
                if len(instances) != 1 or instances[0]["State"]["Name"] not in (
                    "stopped",
                    "terminated",
                ):
                    raise GuardError("post-wait stop state unconfirmed; reservation retained")
                instance = instances[0]
                self.remember_instance(session, instance)
            self.settle_session(session, cancelled=cancelled, instance=instance)
        else:
            self.aws("ec2", "stop-instances", "--instance-ids", session["instance_id"], "--force")
            self.aws("ec2", "wait", "instance-stopped", "--instance-ids", session["instance_id"])
            self.settle_session(session, cancelled=cancelled)
        self.delete_session_schedule(session)

    def settle(self, args):
        if not args.evidence.strip():
            raise GuardError("settle requires console evidence")
        session = next((s for s in self.ledger["sessions"] if s["txn_id"] == args.txn), None)
        if (
            session is None
            or session["status"] not in ("closed", "cancelled")
            or not session.get("stop_time_estimated")
        ):
            raise GuardError("settle requires a session with an estimated stop time")
        txn = self.session_txn(session)
        called = instant(txn.get("aws_called", session["aws_called"]))
        stopped = instant(args.stopped_at)
        observed = instant(session.get("observed_at", session["stopped_at"]))
        if not called <= stopped <= observed:
            raise GuardError("stopped-at must be between aws_called and observed_at")
        txn["stop_time_correction"] = {
            "estimated_stopped_at": session["stopped_at"],
            "stopped_at": stamp(stopped),
            "evidence": args.evidence,
        }
        txn["evidence"] = args.evidence
        session["stopped_at"] = stamp(stopped)
        session["settled_cost_usd"] = session_cost_usd(
            (stopped - called).total_seconds(), self.rate
        )
        session["stop_time_estimated"] = False
        self.update_stop_time_flag()
        self.stage("stop_time_corrected", txn)

    def stop(self, _args):
        self.transaction()
        self.stage("reserved")
        active = self.active()
        if active:
            self.stop_session(active[0])
        else:
            # A manual start outside this tool still needs stopping and accounting.
            self.reconcile_instances()
        self.finish()

    def cancel(self, args):
        if not args.evidence.strip():
            raise GuardError("cancel requires console evidence")
        txn = next((t for t in self.ledger["transactions"] if t["id"] == args.txn), None)
        if txn is None:
            raise GuardError("unknown transaction")
        self.txn = txn
        txn["evidence"] = args.evidence
        session = next((s for s in self.ledger["sessions"] if s["txn_id"] == args.txn), None)
        if session:
            self.settle_session(session, cancelled=True)
        txn["status"] = "cancelled"
        self.stage("cancelled")

    def launch_trap(self, _args):
        self.transaction()
        self.stage("reserved")
        for session in self.active():
            if self.session_txn(session)["command"] != "provision" or session.get(
                "stop_schedule_verified"
            ):
                continue
            # Rediscovery covers the interval after AWS replied but before the id
            # reached the ledger. Absence leaves the reservation pending.
            try:
                self.stop_session(session, cancelled=True)
            except (
                GuardError,
                OSError,
                ValueError,
                KeyError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ) as exc:
                print(f"stop unconfirmed; reservation retained: {exc}")
                raise
            if session["status"] == "cancelled":
                self.session_txn(session)["evidence"] = "trap"
                self.stage("cancelled", self.session_txn(session))
        self.finish()

    def release_hold(self, name):
        if name == "ebs" or name.startswith("transfer:"):
            raise GuardError("storage and transfer holds require confirmed settlement")
        if name not in self.ledger["holds"]:
            raise GuardError("unknown hold")
        del self.ledger["holds"][name]

    def hold(self, args):
        if args.label in self.ledger["holds"] or args.label == "ebs":
            raise GuardError("hold label already exists or is reserved")
        self.ledger["holds"][args.label] = {"usd": args.usd, "kind": "allowance"}
        self.admit()
        self.stage("reserved")

    def release(self, args):
        self.release_hold(args.label)
        self.stage("released")

    def record(self, args):
        self.ledger["line_items"].append(
            {"kind": args.kind, "usd": args.usd, "note": args.note, "utc": stamp(self.now())}
        )
        self.stage("recorded")

    def transfer_begin(self, args):
        manifest = Path(args.manifest).resolve()
        if args.attempt < 1:
            raise GuardError("attempt must be positive")
        if any(
            t["manifest"] == str(manifest) and t["attempt"] == args.attempt
            for t in self.ledger["transfers"]
        ):
            raise GuardError("attempt already reserved; retries require a new attempt")
        contents = json.loads(manifest.read_text())
        sizes = (
            [contents["bytes"]] if "bytes" in contents else [f["bytes"] for f in contents["files"]]
        )
        if any(type(n) is not int or n < 0 for n in sizes):
            raise GuardError("manifest bytes must be nonnegative integers")
        total = sum(sizes)
        name = "transfer:" + os.urandom(8).hex()
        self.ledger["holds"][name] = {
            "usd": egress_usd(total, self.config["egress_usd_per_gb"]),
            "kind": "transfer",
        }
        self.ledger["transfers"].append(
            {
                "manifest": str(manifest),
                "attempt": args.attempt,
                "bytes_manifest": total,
                "hold": name,
                "cost_usd": 0,
                "status": "pending",
            }
        )
        self.admit()
        self.stage("reserved")

    def transfer_end(self, args):
        if args.bytes < 0:
            raise GuardError("bytes must be nonnegative")
        manifest = str(Path(args.manifest).resolve())
        transfer = next(
            (
                t
                for t in self.ledger["transfers"]
                if t["manifest"] == manifest and t["attempt"] == args.attempt
            ),
            None,
        )
        if transfer is None:
            raise GuardError("transfer-end requires a prior transfer-begin for this attempt")
        if transfer["status"] == "settled":
            if transfer["bytes"] != args.bytes:
                raise GuardError(
                    "attempt already settled with different statistics; record a correction"
                )
            return
        transfer.update(
            bytes=args.bytes,
            cost_usd=egress_usd(args.bytes, self.config["egress_usd_per_gb"]),
            status="settled",
        )
        del self.ledger["holds"][transfer["hold"]]
        self.stage("settled")

    def copyback_verified(self, args):
        manifest, report = Path(args.manifest).resolve(), Path(args.sha_report).resolve()
        if not manifest.is_file() or not report.is_file():
            raise GuardError("manifest and SHA report must both exist")
        self.ledger["copyback_verified"] = {
            "manifest": str(manifest),
            "sha_report": str(report),
            "utc": stamp(self.now()),
        }
        self.stage("copyback_verified")

    def extend_cleanup(self, args):
        if any(
            t["command"] == "extend-cleanup" and t["status"] == "pending"
            for t in self.ledger["transactions"]
        ):
            raise GuardError("cleanup extension is pending; reconcile it before extending again")
        cleanup = instant(args.to)
        if cleanup <= instant(self.config["cleanup_by"]) or cleanup <= self.now():
            raise GuardError("extension must move cleanup-by forward into the future")
        if not self.ledger.get("instance_id") or "ebs" not in self.ledger["holds"]:
            raise GuardError("no provisioned storage to extend")
        old = self.cleanup_schedule(self.ledger["instance_id"])
        self.config["cleanup_by"] = stamp(cleanup)
        self.ledger["holds"]["ebs"]["usd"] = self.storage_hold(cleanup)
        self.admit()
        self.transaction()
        self.stage("reserved")  # enlarged hold and new deadline precede ANY AWS call
        actual = self.read_schedule(old)
        expected = self.cleanup_schedule(self.ledger["instance_id"])
        actual["ScheduleExpression"] = expected["ScheduleExpression"]
        self.txn["schedule_update"] = actual
        self.write_schedule("update-schedule", actual)
        self.read_schedule(expected)
        self.finish()

    def reconcile_instances(self):
        active = self.active()
        instances = self.discover(active[0] if active else None)
        expired = False
        for instance in instances:
            instance_id = instance["InstanceId"]
            session = next(
                (s for s in self.active() if s.get("instance_id") in (None, instance_id)), None
            )
            state = instance["State"]["Name"]
            if session:
                self.remember_instance(session, instance)
                expired = state == "running" and self.now() > instant(
                    session["deadline"]
                ) + timedelta(minutes=20)
                if expired:
                    self.ledger["flags"]["EXPIRED_SESSION"] = True
                    self.stage("expired_session")
                if state in ("stopped", "stopping", "terminated"):
                    self.stop_session(session, instance=instance)
                elif state == "running" and not expired and session.get("aws_called"):
                    try:
                        self.read_schedule(self.session_schedule(session))
                        self.read_schedule(self.cleanup_schedule(instance_id))
                        session["stop_schedule_verified"] = True
                        if session["status"] == "pending":
                            self.arm(session)
                    except subprocess.TimeoutExpired:
                        self.stop_session(session)
                        raise
                    except (
                        GuardError,
                        OSError,
                        ValueError,
                        KeyError,
                        subprocess.CalledProcessError,
                    ):
                        self.stop_session(session)
                else:
                    if not session.get("aws_called") and state == "running":
                        # A start never sent by this transaction cannot explain
                        # this runtime. Conservatively charge the unknown gap.
                        origin = self.unreserved_origin()
                        session["aws_called"] = origin
                        self.session_txn(session)["aws_called"] = origin
                    self.stop_session(session)
            elif state in ("running", "pending", "stopping"):
                origin = self.unreserved_origin()
                # A separate recovery transaction owns this interval; it never
                # borrows the original EC2 LaunchTime or a prior session's start.
                recovery = {
                    "id": os.urandom(8).hex(),
                    "command": "reconcile",
                    "status": "pending",
                    "stages": [],
                    "aws_called": origin,
                }
                self.ledger["transactions"].append(recovery)
                session = {
                    "label": "reconcile-" + recovery["id"],
                    "txn_id": recovery["id"],
                    "status": "pending",
                    "reservation_usd": session_cost_usd(
                        (self.now() - instant(origin)).total_seconds(), self.rate
                    ),
                    "settled_cost_usd": None,
                    "aws_called": origin,
                    "deadline": stamp(self.now()),
                    "stop_schedule_name": f"pretrain-{self.config['token']}-stop-reconcile",
                    "stop_schedule_verified": False,
                }
                self.ledger["sessions"].append(session)
                self.remember_instance(session, instance)
                self.stage("unreserved_runtime", recovery)
                self.stop_session(session)
        # No result is not definitive for unresolved launches. In particular do
        # not clear an expired flag while an expired session is still pending.
        unresolved_expired = self.ledger["flags"].get("EXPIRED_SESSION", False) and any(
            self.now() > instant(s["deadline"]) + timedelta(minutes=20) for s in self.active()
        )
        self.ledger["flags"]["EXPIRED_SESSION"] = expired or unresolved_expired

    def unreserved_origin(self):
        stops = [s["stopped_at"] for s in self.ledger["sessions"] if s.get("stopped_at")]
        return self.ledger.get("last_transition_at") or max(stops, default=self.config["created"])

    def storage_status(self):
        exists = bool(self.volumes())
        self.ledger["flags"]["OVERDUE_STORAGE"] = exists and self.now() > instant(
            self.config["cleanup_by"]
        )
        return exists

    def reconcile(self, _args):
        self.transaction()
        self.stage("reserved")
        # Stop unsafe runtime first: a scheduler outage must never block an EC2
        # stop. A pending extension may conservatively cause an open session to
        # stop until its cleanup schedule has been repaired and verified.
        self.reconcile_instances()
        for session in self.obsolete_schedules():
            self.delete_session_schedule(session)
        # An extension can die before or during update-schedule. Its enlarged
        # reservation remains in place and its full desired configuration is
        # replayable once captured. A crash before get-schedule needs discovery.
        for txn in self.ledger["transactions"]:
            if txn is self.txn or txn["command"] != "extend-cleanup" or txn["status"] != "pending":
                continue
            desired = txn.get("schedule_update")
            if desired is None:
                desired = self.aws(
                    "scheduler", "get-schedule", "--name", self.config["cleanup_schedule_name"]
                )
                desired["ScheduleExpression"] = self.cleanup_schedule(self.ledger["instance_id"])[
                    "ScheduleExpression"
                ]
            self.write_schedule("update-schedule", desired)
            self.read_schedule(self.cleanup_schedule(self.ledger["instance_id"]))
            txn["status"] = "closed"
            self.stage("extension_recovered", txn)
        self.storage_status()
        self.finish()

    def status(self, _args):
        self.transaction()
        self.update_stop_time_flag()
        self.stage("reserved")
        self.storage_status()
        readings = {}
        expected = []
        if self.ledger.get("instance_id") and not self.ledger.get("terminated_at"):
            expected.append(self.cleanup_schedule(self.ledger["instance_id"]))
        expected.extend(self.session_schedule(s) for s in self.active() if s.get("instance_id"))
        for schedule in expected:
            try:
                self.read_schedule(schedule)
                readings[schedule["Name"]] = "verified"
            except (
                GuardError,
                OSError,
                ValueError,
                KeyError,
                subprocess.CalledProcessError,
            ) as exc:
                readings[schedule["Name"]] = "UNVERIFIED: " + str(exc)
        self.ledger["schedule_readback"] = readings
        self.finish()
        print(
            json.dumps(
                {
                    "committed": committed(self.ledger),
                    "enforce": self.config["enforce_usd"],
                    "cap": self.config["cap_usd"],
                    "holds": self.ledger["holds"],
                    "sessions": self.ledger["sessions"],
                    "schedules": readings,
                },
                indent=2,
            )
        )

    def terminate(self, args):
        if self.active():
            raise GuardError("stop or reconcile all open/pending sessions before termination")
        if not args.abandon and not self.ledger["copyback_verified"]:
            raise GuardError("copyback-verified is required (or explicit --abandon)")
        if not self.ledger.get("instance_id"):
            raise GuardError("no discovered instance to terminate; reconcile first")
        self.transaction()
        self.stage("reserved")
        instance_id = self.ledger["instance_id"]
        instances = self.discover()
        if len(instances) > 1:
            raise GuardError("instance discovery is ambiguous; reconcile in the console")
        if instances:
            if instances[0]["State"]["Name"] in ("running", "pending", "stopping"):
                self.reconcile_instances()
            if instances[0]["State"]["Name"] != "terminated":
                self.aws("ec2", "terminate-instances", "--instance-ids", instance_id, "--force")
            self.aws("ec2", "wait", "instance-terminated", "--instance-ids", instance_id)
        # Wait on each known root volume AND search the tag: a replacement or an
        # attached surviving volume must not silently release the storage hold.
        for volume_id in self.ledger.get("volume_ids", []):
            if instances:
                self.aws("ec2", "wait", "volume-deleted", "--volume-ids", volume_id)
            else:
                try:
                    volumes = self.aws("ec2", "describe-volumes", "--volume-ids", volume_id)[
                        "Volumes"
                    ]
                except subprocess.CalledProcessError as exc:
                    if "InvalidVolume.NotFound" not in (exc.stderr or ""):
                        raise
                else:
                    if (
                        len(volumes) != 1
                        or volumes[0].get("VolumeId") != volume_id
                        or volumes[0].get("State") != "deleted"
                    ):
                        raise GuardError("recorded volume deletion unconfirmed; EBS hold retained")
        if self.storage_status():
            raise GuardError("storage still exists; EBS hold retained")
        if not instances:
            self.txn["evidence"] = "instance no longer listed; recorded volumes confirmed deleted"
            self.stage("termination_confirmed_from_evidence")
        if "ebs" in self.ledger["holds"]:
            hours = (self.now() - instant(self.config["created"])).total_seconds() / 3600
            usd = ebs_hold_usd(
                self.config["volume_gb"], self.config["ebs_usd_per_gb_month"], max(0, hours)
            )
            self.ledger["line_items"].append(
                {
                    "kind": "ebs",
                    "usd": usd,
                    "note": "storage through confirmed deletion",
                    "utc": stamp(self.now()),
                }
            )
            del self.ledger["holds"]["ebs"]
            self.stage("storage_deleted")
        for session in self.ledger["sessions"]:
            self.delete_session_schedule(session)
        self.delete_schedule(self.config["cleanup_schedule_name"])
        self.ledger["terminated_at"] = stamp(self.now())
        self.finish()
        self.status(args)

    def dispatch(self, args):
        self.command = args.command
        if self.ledger_path.exists():
            self.ledger = json.loads(self.ledger_path.read_text())
            self.validate()
        elif args.command != "init":
            raise GuardError("ledger does not exist; init first")
        commitments = {"provision", "start", "hold", "transfer-begin", "extend-cleanup"}
        if self.command in commitments and self.ledger["flags"].get("EXPIRED_SESSION"):
            raise GuardError("EXPIRED_SESSION: reconcile before new commitments", 7)
        method = (
            "begin_session"
            if self.command in ("start", "provision")
            else self.command.replace("-", "_")
        )
        getattr(self, method)(args)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--ledger", default="ops/cost-ledger.json")
    commands = result.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--cap", type=finite, default=50.0)
    init.add_argument("--enforce", type=finite, default=47.0)
    init.add_argument("--cleanup-by", required=True)
    init.add_argument("--type", default="g6.xlarge")
    init.add_argument("--market", choices=("on-demand", "spot"), default="on-demand")
    init.add_argument("--price", "--compute-usd-per-h", type=finite, default=1.0064)
    init.add_argument("--ipv4-usd-per-h", type=finite, default=0.005)
    init.add_argument("--ebs-usd-per-gb-month", type=finite, default=0.0952)
    init.add_argument("--egress-usd-per-gb", type=finite, default=0.09)
    init.add_argument("--volume-gb", type=int, default=200)
    init.add_argument("--region", default="eu-central-1")
    init.add_argument("--create-role", action="store_true")
    for name in ("provision", "start"):
        command = commands.add_parser(name)
        command.add_argument("--label", type=label, required=True)
        command.add_argument("--minutes", type=finite, required=True)
        if name == "start":
            command.add_argument("--consume-hold", type=label)
        else:
            command.add_argument(
                "--user-data", default=str(Path(__file__).with_name("pretrain-user-data.sh"))
            )
    for name in ("stop", "reconcile", "status", "launch-trap"):
        commands.add_parser(name)
    cancel = commands.add_parser("cancel")
    cancel.add_argument("--txn", required=True)
    cancel.add_argument("--evidence", required=True)
    settle = commands.add_parser("settle")
    settle.add_argument("--txn", required=True)
    settle.add_argument("--stopped-at", required=True)
    settle.add_argument("--evidence", required=True)
    hold = commands.add_parser("hold")
    hold.add_argument("--label", type=label, required=True)
    hold.add_argument("--usd", type=finite, required=True)
    release = commands.add_parser("release")
    release.add_argument("--label", type=label, required=True)
    extend = commands.add_parser("extend-cleanup")
    extend.add_argument("--to", required=True)
    for name in ("transfer-begin", "transfer-end"):
        transfer = commands.add_parser(name)
        transfer.add_argument("--manifest", required=True)
        transfer.add_argument("--attempt", type=int, required=True)
        if name == "transfer-end":
            transfer.add_argument("--bytes", type=int, required=True)
    terminate = commands.add_parser("terminate")
    terminate.add_argument("--abandon", action="store_true")
    record = commands.add_parser("record")
    record.add_argument("--kind", choices=("ebs", "transfer", "other"), required=True)
    record.add_argument("--usd", type=finite, required=True)
    record.add_argument("--note", required=True)
    copyback = commands.add_parser("copyback-verified")
    copyback.add_argument("--manifest", required=True)
    copyback.add_argument("--sha-report", required=True)
    watch = commands.add_parser("watch")
    watch.add_argument("--interval", type=finite, default=300)
    return result


def main(argv=None, *, run=run_command, now=utc_now, sleep=time.sleep):
    args = parser().parse_args(argv)
    if args.command == "watch":
        if args.interval <= 0:
            print("watch interval must be positive")
            return 2
        try:
            while True:
                code = main(["--ledger", args.ledger, "reconcile"], run=run, now=now, sleep=sleep)
                if code:
                    print(f"reconcile failed ({code}); reservation retained; retrying")
                sleep(args.interval)
        except KeyboardInterrupt:
            return 0
    guard = Guard(args.ledger, run=run, now=now, sleep=sleep)
    try:
        guard.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with guard.ledger_path.with_name(guard.ledger_path.name + ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise GuardError("another operator command is using this ledger") from exc
            guard.dispatch(args)
        return 0
    except GuardError as exc:
        print(str(exc))
        return exc.code
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(f"operation failed; inspect ledger and reconcile: {exc}")
        return 2
    finally:
        # Report persisted reality, never a rejected in-memory candidate.
        if guard.ledger_path.is_file():
            try:
                data = json.loads(guard.ledger_path.read_text())
                if committed(data) > data["config"]["enforce_usd"]:
                    print("OVER CAP")
                for flag, enabled in data["flags"].items():
                    if enabled:
                        print(flag)
            except (OSError, ValueError, KeyError, TypeError):
                pass


if __name__ == "__main__":
    raise SystemExit(main())
