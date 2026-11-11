"""Ledger-aware session recovery; AWS is invoked only by operator commands."""

import ipaddress
import json
import math
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from process_control import cleanup_signals, run_process

HERE = Path(__file__).resolve().parent
CAPACITY_ERRORS = {
    "InsufficientInstanceCapacity",
    "InsufficientHostCapacity",
    "InsufficientReservedInstanceCapacity",
}


def ledger():
    return json.loads(Path(os.environ["LEDGER"]).read_text())


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def epoch(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def limit(name, default):
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def validate_limits():
    for name in (
        "AWS_COMMAND_TIMEOUT_SECONDS",
        "GUARD_TIMEOUT_SECONDS",
        "STOP_TIMEOUT_SECONDS",
        "SSH_TIMEOUT_SECONDS",
        "CHECKOUT_TIMEOUT_SECONDS",
        "PUSH_TIMEOUT_SECONDS",
        "DISPATCH_TIMEOUT_SECONDS",
        "COPY_TIMEOUT_SECONDS",
        "COPY_BUDGET_SECONDS",
        "START_TIMEOUT_SECONDS",
        "COMMAND_TIMEOUT_SECONDS",
    ):
        limit(name, 180)
    if os.environ.get("LATEST_DEADLINE"):
        epoch(os.environ["LATEST_DEADLINE"])


def execute(argv, seconds, capture=False):
    result = run_process(
        argv, seconds, text=True, stdout=subprocess.PIPE if capture else sys.stderr
    )
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, argv, output=result.stdout)
    return result.stdout or ""


def aws(service, operation, *args):
    return json.loads(
        execute(
            [
                "aws",
                service,
                operation,
                *args,
                "--profile",
                os.environ["AWS_PROFILE"],
                "--region",
                ledger()["config"]["region"],
                "--output",
                "json",
            ],
            limit("AWS_COMMAND_TIMEOUT_SECONDS", 30),
            capture=True,
        )
    )


def guard(*args, seconds=None):
    return execute(
        [
            os.environ["PYTHON"],
            os.environ["LOCAL_CODE"] + "/scripts/ec2/cost_guard.py",
            "--ledger",
            os.environ["LEDGER"],
            *args,
        ],
        seconds or limit("GUARD_TIMEOUT_SECONDS", 180),
    )


def credentials():
    try:
        return aws("sts", "get-caller-identity")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise ValueError(
            f"AWS credentials unavailable: run aws sso login --profile {os.environ['AWS_PROFILE']}"
        ) from exc


def ingress():
    credentials()
    # Re-read the public address, rather than trusting a previously exported IP.
    address = execute(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--max-time",
            "8",
            "https://checkip.amazonaws.com",
        ],
        10,
        capture=True,
    ).strip()
    cidr = str(ipaddress.IPv4Address(address)) + "/32"
    group = ledger()["config"]["security_group_id"]
    response = aws("ec2", "describe-security-groups", "--group-ids", group)

    def present(groups):
        return any(
            p.get("IpProtocol") in ("tcp", "6")
            and p.get("FromPort") == 22
            and p.get("ToPort") == 22
            and any(r.get("CidrIp") == cidr for r in p.get("IpRanges", []))
            for g in groups["SecurityGroups"]
            if g["GroupId"] == group
            for p in g.get("IpPermissions", [])
        )

    if not present(response):
        try:
            aws(
                "ec2",
                "authorize-security-group-ingress",
                "--group-id",
                group,
                "--protocol",
                "tcp",
                "--port",
                "22",
                "--cidr",
                cidr,
            )
        except subprocess.CalledProcessError:
            # A concurrent operator may have added the same rule; verify it below.
            pass
        if not present(aws("ec2", "describe-security-groups", "--group-ids", group)):
            raise ValueError(f"Could not confirm tcp/22 ingress for {cidr} in {group}")
    message = f"{utc()} ensured tcp/22 ingress group={group} cidr={cidr}; old rules retained\n"
    with (Path(os.environ["EXPERIMENT"]) / "ops/sg-changes.log").open("a") as out:
        out.write(message)
    print(message, end="", file=sys.stderr)
    return address


def unresolved(data):
    return [
        s
        for s in data["sessions"]
        if s["status"] in ("open", "pending")
        or s.get("stop_time_estimated")
        or (s["status"] in ("closed", "cancelled") and not s.get("stopped_at"))
    ]


def recovery_commands(data):
    prefix = [
        os.environ["PYTHON"],
        os.environ["LOCAL_CODE"] + "/scripts/ec2/cost_guard.py",
        "--ledger",
        os.environ["LEDGER"],
    ]
    for s in unresolved(data):
        if s["status"] in ("open", "pending"):
            print(
                "Stop unconfirmed; recover AWS evidence, then run: "
                + shlex.join(prefix + ["reconcile"]),
                file=sys.stderr,
            )
        # Placeholders are deliberately explicit: SSH loss cannot date an EC2 stop.
        print(
            f"Unresolved session label={s['label']} txn={s['txn_id']}; after confirmed closure run: "
            + shlex.join(
                prefix
                + [
                    "settle",
                    "--txn",
                    s["txn_id"],
                    "--stopped-at",
                    "<confirmed-stop-UTC>",
                    "--evidence",
                    "<AWS/console evidence>",
                ]
            ),
            file=sys.stderr,
        )


def owned_session(data, owned):
    active = [s for s in data["sessions"] if s["status"] in ("open", "pending")]
    if owned is not None and any(s["txn_id"] not in owned for s in active):
        raise ValueError("Another transaction owns the active session; refusing to stop it")


def stop(owned=None):
    # Interrupt protection covers only this finite cleanup, never copy-back.
    with cleanup_signals():
        owned_session(ledger(), owned)
        return _stop(owned)


def _stop(owned):
    end = time.monotonic() + limit("STOP_TIMEOUT_SECONDS", 480)
    connection = None
    try:
        output = execute(
            ["bash", str(HERE / "session.sh"), "host", "--fallback"],
            min(10, (end - time.monotonic()) / 10),
            capture=True,
        )
        connection = output.rstrip("\n").split("\t")
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    rc = 0

    def phase(operation):
        nonlocal rc
        owned_session(ledger(), owned)
        try:
            guard(
                operation,
                seconds=min(
                    limit("GUARD_TIMEOUT_SECONDS", 180), max(0.01, (end - time.monotonic()) / 2)
                ),
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            print(f"guard {operation} failed: {exc}", file=sys.stderr)
            rc = 2

    phase("stop")
    phase("reconcile")
    data = ledger()
    # A successful guard exit can leave pending discovery or estimated stop times.
    if (rc or unresolved(data)) and (not connection or len(connection) != 7):
        reason = (
            f"connection record has {len(connection)} fields; expected 7"
            if connection
            else "missing connection record"
        )
        print(f"SSH shutdown fallback skipped: {reason}", file=sys.stderr)
    if (rc or unresolved(data)) and connection and len(connection) == 7:
        key, address, known, _deadline, label, _instance, txn = connection
        if owned is None or txn in owned:
            print("Stop unresolved; attempting bounded SSH shutdown", file=sys.stderr)
            ssh = [
                "ssh",
                "-i",
                key,
                "-o",
                f"UserKnownHostsFile={known}",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                f"{os.environ['GUEST_USER']}@{address}",
            ]
            at, failed, shutdown_rc = utc(), "", 124
            try:
                # Refresh also remains inside the cleanup budget.
                execute(
                    [os.environ["PYTHON"], str(HERE / "ops_control.py"), "ingress"],
                    min(40, max(0.01, (end - time.monotonic()) / 3)),
                )
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                print(f"Ingress refresh unavailable; trying pinned host: {exc}", file=sys.stderr)
            try:
                reply = run_process(
                    ssh + ["sudo shutdown -h now"],
                    min(limit("SSH_TIMEOUT_SECONDS", 30), max(0.01, (end - time.monotonic()) / 2)),
                )
                shutdown_rc = reply.returncode
                probe_at = utc()
                probe = run_process(ssh + ["true"], min(15, max(0.01, end - time.monotonic())))
                if probe.returncode:
                    failed = probe_at
            except subprocess.TimeoutExpired:
                pass
            except OSError as exc:
                print(f"SSH shutdown failed: {exc}", file=sys.stderr)
            evidence = dict(
                label=label,
                txn=txn,
                shutdown_at=at,
                shutdown_ssh_exit=shutdown_rc,
                first_failed_ssh_at=failed or None,
                note="SSH loss is not confirmation of an EC2 stop",
            )
            path = Path(os.environ["EXPERIMENT"]) / "ops" / f"stop.{label}.json"
            if path.exists():
                previous = json.loads(path.read_text())
                evidence["first_failed_ssh_at"] = (
                    previous.get("first_failed_ssh_at") or failed or None
                )
            path.write_text(json.dumps(evidence, indent=2) + "\n")
            print(f"SSH stop evidence: {path}", file=sys.stderr)
    data = ledger()
    if unresolved(data) or data.get("flags", {}).get("STOP_TIME_ESTIMATED"):
        recovery_commands(data)
        return 2
    return rc


def capacity_recovery(session):
    """Retry only an evidenced, rejected capacity request, never an ambiguous boot."""
    data = ledger()
    txn = next(t for t in data["transactions"] if t["id"] == session["txn_id"])
    called = txn.get("aws_called") or session.get("aws_called")
    if not called or txn["command"] not in ("start", "provision"):
        raise ValueError("No admitted AWS start call; automatic retry refused")
    stages = [s["stage"] for s in txn.get("stages", [])]
    after_call = stages[stages.index("aws_called") + 1 :] if "aws_called" in stages else []
    if any(
        s in after_call for s in ("aws:ec2:wait", "deadline_armed", "open", "instance_discovered")
    ):
        raise ValueError("Start progressed beyond the AWS request; capacity retry refused")
    identity = credentials()
    event_name = "StartInstances" if txn["command"] == "start" else "RunInstances"
    response = aws(
        "cloudtrail",
        "lookup-events",
        "--lookup-attributes",
        f"AttributeKey=EventName,AttributeValue={event_name}",
        "--start-time",
        called,
        "--end-time",
        utc(),
    )
    events = []
    for wrapper in response.get("Events", []):
        event = json.loads(wrapper["CloudTrailEvent"])
        if event.get("eventName") != event_name or event.get("eventSource") != "ec2.amazonaws.com":
            continue
        caller = event.get("userIdentity") or {}
        if not identity.get("Arn") or caller.get("arn") != identity["Arn"]:
            continue
        if event.get("awsRegion") != data["config"]["region"] or epoch(event["eventTime"]) < epoch(
            called
        ):
            continue
        request = event.get("requestParameters") or {}
        if event_name == "StartInstances":
            ids = [i.get("instanceId") for i in request.get("instancesSet", {}).get("items", [])]
            matches = ids == [session.get("instance_id")]
        else:
            matches = request.get("clientToken") == data["config"]["token"]
        if matches:
            events.append(event)
    if len(events) != 1 or events[0].get("errorCode") not in CAPACITY_ERRORS:
        raise ValueError(
            "Missing, non-capacity or ambiguous CloudTrail evidence; automatic retry refused"
        )
    event = events[0]
    used = [t.get("evidence", "") for t in data["transactions"] if t["id"] != txn["id"]]
    event_id = event["eventID"]
    if any(event_id in evidence for evidence in used):
        raise ValueError("CloudTrail event already used by another transaction")
    if txn.get("stage") not in ("aws_called", "cancelled", "schedule_deleted"):
        raise ValueError("Failure occurred after the AWS start stage; automatic retry refused")
    # The guard must have confirmed stopped/absent state before releasing anything.
    instances = (
        aws("ec2", "describe-instances", "--instance-ids", session["instance_id"])
        if session.get("instance_id")
        else aws(
            "ec2",
            "describe-instances",
            "--filters",
            "Name=client-token,Values=" + data["config"]["token"],
        )
    )
    discovered = [i for r in instances["Reservations"] for i in r["Instances"]]
    if session.get("instance_id"):
        safe = (
            len(discovered) == 1
            and discovered[0]["InstanceId"] == session["instance_id"]
            and discovered[0]["State"]["Name"] == "stopped"
        )
    else:
        safe = not discovered
    if not safe:
        raise ValueError("Capacity request has no confirmed stopped/absent instance; retry refused")
    path = Path(os.environ["EXPERIMENT"]) / "ops" / f"capacity.{session['label']}.json"
    path.write_text(json.dumps(event, indent=2) + "\n")
    evidence = f"CloudTrail {event_id} {event_name} {event['errorCode']} at {event['eventTime']}; request rejected; {path}"
    if session["status"] in ("open", "pending"):
        guard("cancel", "--txn", txn["id"], "--evidence", evidence)
    if next(s for s in ledger()["sessions"] if s["txn_id"] == txn["id"]).get("stop_time_estimated"):
        # A rejected start incurred no runtime; the attempted-call timestamp is
        # the conservative zero-runtime boundary (guard still charges its minimum).
        guard("settle", "--txn", txn["id"], "--stopped-at", called, "--evidence", evidence)
    elif session["status"] in ("closed", "cancelled"):
        guard("cancel", "--txn", txn["id"], "--evidence", evidence)
    if unresolved(ledger()):
        raise ValueError("Capacity transaction remains unresolved; retry refused")


def start_check(label, minutes):
    duration = float(minutes)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("minutes must be positive and finite")
    data = ledger()
    if unresolved(data) or data.get("flags", {}).get("STOP_TIME_ESTIMATED"):
        raise ValueError("Open/pending or estimated session exists; stop/reconcile/settle first")
    if any(s["label"] == label for s in data["sessions"]):
        raise ValueError("Label was already admitted; inspect its transaction and use a new label")


def start(label, minutes, retries="24", owned=None):
    import re

    if (
        not re.fullmatch(r"[a-zA-Z0-9_-]{1,24}", label)
        or not str(retries).isdigit()
        or not 0 <= int(retries) <= 24
    ):
        raise ValueError("Invalid label or retry count (0–24)")
    validate_limits()
    start_check(label, minutes)
    standalone = owned is None
    owned = set() if standalone else owned
    initial = {s["txn_id"] for s in ledger()["sessions"]}
    success = False
    try:
        for attempt in range(int(retries) + 1):
            suffix = f"-r{attempt + 1}" if attempt else ""
            current_label = label[: 24 - len(suffix)] + suffix
            duration = float(minutes)
            if os.environ.get("LATEST_DEADLINE"):
                duration = min(duration, (epoch(os.environ["LATEST_DEADLINE"]) - time.time()) / 60)
            start_check(current_label, str(duration))
            ingress()
            argv = [
                "bash",
                os.environ["LOCAL_CODE"] + "/scripts/ec2/pretrain_launch.sh",
                "start",
                "--label",
                current_label,
                "--minutes",
                str(duration),
                "--ledger",
                os.environ["LEDGER"],
            ]
            if os.environ.get("LATEST_DEADLINE"):
                argv += ["--latest-deadline", os.environ["LATEST_DEADLINE"]]
            try:
                execute(argv, limit("START_TIMEOUT_SECONDS", 1200))
            except subprocess.CalledProcessError:
                candidates = [
                    s
                    for s in ledger()["sessions"]
                    if s["label"] == current_label and s["txn_id"] not in initial
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        "Start failed without an owned admitted transaction; retry refused"
                    ) from None
                owned.add(candidates[0]["txn_id"])
                capacity_recovery(candidates[0])
                if attempt == int(retries):
                    raise ValueError("Capacity attempts exhausted") from None
                delay = 300
                if os.environ.get("LATEST_DEADLINE"):
                    delay = min(delay, max(0, epoch(os.environ["LATEST_DEADLINE"]) - time.time()))
                print(
                    f"Capacity confirmed; retry with fresh label in {delay:g} seconds",
                    file=sys.stderr,
                )
                execute(["sleep", str(delay)], delay + 5)
                continue
            finally:
                owned.update(
                    s["txn_id"]
                    for s in ledger()["sessions"]
                    if s["txn_id"] not in initial and s["label"] == current_label
                )
            current = next(
                s
                for s in ledger()["sessions"]
                if s["label"] == current_label and s["txn_id"] in owned
            )
            if current["status"] != "open":
                raise ValueError("Start did not open its admitted transaction")
            execute(["bash", str(HERE / "session.sh"), "host"], 10, capture=True)
            print(
                f"OPEN {current_label} {current['instance_id']} {current['public_ip']} {current['deadline']}"
            )
            success = True
            return current
    finally:
        if standalone and not success and owned:
            if any(s["txn_id"] in owned for s in unresolved(ledger())):
                stop(owned)
