"""Offline operator guard checks: real ledger writes, fake AWS/SSH and UTC clock."""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ec2/cost_guard.py"


@pytest.fixture
def cg():
    assert SCRIPT.is_file(), "operator cost guard has not been implemented"
    spec = importlib.util.spec_from_file_location("cost_guard", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Crash(BaseException):
    """Process loss, deliberately bypassing ordinary error recovery."""


class Clock:
    def __init__(self):
        self.value = datetime(2026, 11, 2, 12, tzinfo=timezone.utc)
        self.sleeps = []

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += timedelta(seconds=seconds)

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.advance(seconds)


def option(cmd, name):
    return cmd[cmd.index(name) + 1]


class Cloud:
    def __init__(self, path, clock):
        self.path, self.clock = path, clock
        self.calls, self.snapshots = [], []
        self.schedules = {}
        self.instance = None
        self.volumes = []
        self.fail = None
        self.crash = None
        self.armed_delta = 0
        self.armed_output = None
        self.ssh_errors = []
        self.corrupt = None
        self.role_exists = True
        self.key = False
        self.sg = False

    def __call__(self, cmd):
        self.calls.append(cmd)
        self.snapshots.append(json.loads(self.path.read_text()))
        if self.crash and self.crash(cmd):
            raise Crash()
        if self.fail and self.fail(cmd):
            raise subprocess.CalledProcessError(1, cmd, stderr="simulated failure")
        if cmd[0] == "ssh":
            if self.ssh_errors:
                code, output, error = self.ssh_errors.pop(0)
                raise subprocess.CalledProcessError(code, cmd, output=output, stderr=error)
            if self.armed_output is not None:
                return self.armed_output
            requested = datetime.fromisoformat(cmd[-1].replace("Z", "+00:00"))
            scheduled = requested + timedelta(seconds=self.armed_delta)
            return "DEADLINE_SET " + scheduled.isoformat().replace("+00:00", "Z") + "\n"
        assert cmd[0] == "aws"
        service, action = cmd[1:3]
        assert option(cmd, "--region") == "eu-central-1"
        if service == "iam":
            if action == "get-role" and not self.role_exists:
                raise subprocess.CalledProcessError(1, cmd, stderr="NoSuchEntity")
            if action == "create-role":
                self.role_exists = True
            return json.dumps(
                {"Role": {"Arn": "arn:aws:iam::123456789012:role/" + option(cmd, "--role-name")}}
            )
        if service == "scheduler":
            name = option(cmd, "--name")
            if action in ("create-schedule", "update-schedule"):
                fields = {
                    "Name": name,
                    "State": option(cmd, "--state"),
                    "ScheduleExpression": option(cmd, "--schedule-expression"),
                    "ScheduleExpressionTimezone": option(cmd, "--schedule-expression-timezone"),
                    "FlexibleTimeWindow": json.loads(option(cmd, "--flexible-time-window")),
                    "Target": json.loads(option(cmd, "--target")),
                }
                for key, flag in [
                    ("Description", "--description"),
                    ("KmsKeyArn", "--kms-key-arn"),
                    ("GroupName", "--group-name"),
                    ("ActionAfterCompletion", "--action-after-completion"),
                    ("StartDate", "--start-date"),
                    ("EndDate", "--end-date"),
                ]:
                    if flag in cmd:
                        fields[key] = option(cmd, flag)
                self.schedules[name] = fields
                return "{}"
            if name not in self.schedules:
                raise subprocess.CalledProcessError(1, cmd, stderr="ResourceNotFoundException")
            if action == "get-schedule":
                result = json.loads(json.dumps(self.schedules[name]))
                if self.corrupt:
                    self.corrupt(result)
                return json.dumps(result)
            if action == "delete-schedule":
                del self.schedules[name]
                return "{}"
        if service == "ec2":
            if action == "describe-key-pairs":
                return json.dumps({"KeyPairs": [{"KeyName": "key"}] if self.key else []})
            if action == "create-key-pair":
                self.key = True
                return json.dumps({"KeyMaterial": "test-private-key"})
            if action == "describe-security-groups":
                return json.dumps(
                    {
                        "SecurityGroups": [
                            {
                                "GroupId": "sg-test",
                                "IpPermissions": [
                                    {
                                        "IpProtocol": "tcp",
                                        "FromPort": 22,
                                        "ToPort": 22,
                                        "IpRanges": [{"CidrIp": "203.0.113.5/32"}],
                                    }
                                ],
                            }
                        ]
                        if self.sg
                        else []
                    }
                )
            if action == "create-security-group":
                self.sg = True
                return '{"GroupId":"sg-test"}'
            if action == "authorize-security-group-ingress":
                assert option(cmd, "--cidr") == "203.0.113.5/32"
                return "{}"
            if action == "describe-images":
                return '{"Images":[{"RootDeviceName":"/dev/sda1"}]}'
            if action == "run-instances":
                self.instance = {
                    "InstanceId": "i-test",
                    "State": {"Name": "running"},
                    "PublicIpAddress": "203.0.113.10",
                    "LaunchTime": "2020-01-01T00:00:00Z",
                    "BlockDeviceMappings": [{"Ebs": {"VolumeId": "vol-test"}}],
                }
                self.volumes = [{"VolumeId": "vol-test", "State": "in-use"}]
                return json.dumps({"Instances": [self.instance]})
            if action == "describe-instances":
                return json.dumps(
                    {"Reservations": [{"Instances": [self.instance] if self.instance else []}]}
                )
            if action == "describe-volumes":
                return json.dumps({"Volumes": self.volumes})
            if action in ("start-instances", "stop-instances", "terminate-instances"):
                assert option(cmd, "--instance-ids") == "i-test"
                self.instance["State"]["Name"] = {
                    "start-instances": "running",
                    "stop-instances": "stopped",
                    "terminate-instances": "terminated",
                }[action]
                if action == "terminate-instances":
                    self.volumes = []
                return "{}"
            if action == "wait":
                self.clock.advance(5)
                return ""
        raise AssertionError(f"unexpected external operation: {cmd}")


@pytest.fixture
def env(cg, tmp_path, monkeypatch):
    clock = Clock()
    path = tmp_path / "ops/ledger.json"
    cloud = Cloud(path, clock)
    for key, value in {
        "OPERATOR_IP": "203.0.113.5",
        "VPC_ID": "vpc-test",
        "SUBNET_ID": "subnet-test",
        "AMI_ID": "ami-test",
    }.items():
        monkeypatch.setenv(key, value)
    user_data = tmp_path / "guest.sh"
    user_data.write_text("#!/bin/bash\n")

    def run(*args):
        return cg.main(
            ["--ledger", str(path), *map(str, args)], run=cloud, now=clock, sleep=clock.sleep
        )

    def ledger():
        return json.loads(path.read_text())

    def edit(fn):
        data = ledger()
        fn(data)
        path.write_text(json.dumps(data))

    def provision(label="setup", minutes=180):
        return run("provision", "--label", label, "--minutes", minutes, "--user-data", user_data)

    assert run("init", "--cleanup-by", "2026-11-08T12:00:00Z") == 0
    return type(
        "Env",
        (),
        {
            "run": staticmethod(run),
            "ledger": staticmethod(ledger),
            "edit": staticmethod(edit),
            "provision": staticmethod(provision),
            "cloud": cloud,
            "clock": clock,
            "path": path,
            "tmp": tmp_path,
            "cg": cg,
        },
    )()


def test_accounting_units_and_minimum(cg):
    assert cg.reservation_usd(180, 1.0114) == pytest.approx(3.3713333333333333)
    assert cg.session_cost_usd(25 * 3600, 1.0114) == pytest.approx(25.285)
    assert cg.session_cost_usd(61, 1.0114) == pytest.approx(0.01713761111111111)
    assert cg.session_cost_usd(0, 1.0114) == pytest.approx(0.016856666666666667)
    assert cg.ebs_hold_usd(200, 0.0952, 144) == 3.81
    assert cg.egress_usd(1_000_000_000, 0.09) == 0.09


def test_one_formula_keeps_settled_and_cancelled_cost(cg):
    ledger = {
        "sessions": [
            {"settled_cost_usd": 3, "reservation_usd": 10},
            {"settled_cost_usd": 2, "reservation_usd": 8, "status": "cancelled"},
            {"settled_cost_usd": None, "reservation_usd": 4},
        ],
        "holds": {"ebs": {"usd": 1}, "eval": {"usd": 2}, "transfer:x": {"usd": 3}},
        "transfers": [{"cost_usd": 0.1}, {"cost_usd": 0.2}],
        "line_items": [{"usd": 0.4}],
    }
    assert cg.committed(ledger) == pytest.approx(15.7)


def test_init_defaults_and_iam_fail_closed(cg, tmp_path):
    clock = Clock()
    path = tmp_path / "ledger.json"
    cloud = Cloud(path, clock)
    cloud.role_exists = False
    args = ["--ledger", str(path), "init", "--cleanup-by", "2026-11-08T12:00:00Z"]
    assert cg.main(args + ["--market", "spot"], run=cloud, now=clock) == 3
    assert not path.exists()
    assert cg.main(args, run=cloud, now=clock) == 2
    data = json.loads(path.read_text())
    config = data["config"]
    assert re.fullmatch(r"[0-9a-f]{12}", config["token"])
    assert re.fullmatch(r"pretrain-[0-9a-f]{12}", config["experiment"])
    assert re.fullmatch(r"pretrain-[0-9a-f]{12}-cleanup", config["cleanup_schedule_name"])
    assert re.fullmatch(r"pretrain-cleanup-scheduler-[0-9a-f]{12}", config["role"])
    assert config["compute_usd_per_h"] == 1.0064
    assert config["ipv4_usd_per_h"] == 0.005
    assert config["ebs_usd_per_gb_month"] == 0.0952
    assert config["egress_usd_per_gb"] == 0.09
    assert config["volume_gb"] == 200
    assert config["type"] == "g6.xlarge"
    assert config["region"] == "eu-central-1"
    assert config["cap_usd"] == 50 and config["enforce_usd"] == 47
    assert data["holds"]["ebs"]["usd"] == 4.45  # six days + 24 h
    assert cg.main(args + ["--create-role"], run=cloud, now=clock) == 0
    assert any(c[2] == "put-role-policy" for c in cloud.calls)


@pytest.mark.parametrize(
    "command", ["provision", "start", "hold", "transfer-begin", "extend-cleanup"]
)
def test_commitments_refuse_cap_without_cloud_side_effects(env, command):
    if command in ("start", "extend-cleanup"):
        assert env.provision() == 0
        assert env.run("stop") == 0
    manifest = env.tmp / "manifest.json"
    manifest.write_text('{"bytes":1000000000}')
    env.edit(lambda d: d["line_items"].append({"usd": 47, "kind": "other"}))
    args = {
        "provision": ["--label", "x", "--minutes", "10"],
        "start": ["--label", "x", "--minutes", "10"],
        "hold": ["--label", "eval", "--usd", "1"],
        "transfer-begin": ["--manifest", str(manifest), "--attempt", "1"],
        "extend-cleanup": ["--to", "2026-11-10T12:00:00Z"],
    }
    count = len(env.cloud.calls)
    before = env.path.read_bytes()
    assert env.run(command, *args[command]) == 2
    assert len(env.cloud.calls) == count
    assert env.path.read_bytes() == before


def test_provision_backstops_deadline_tags_and_timestamps(env):
    assert env.provision() == 0
    data = env.ledger()
    session = data["sessions"][0]
    assert session["status"] == "open"
    assert session["instance_id"] == "i-test"
    assert session["public_ip"] == "203.0.113.10"
    assert session["aws_called"] == "2026-11-02T12:00:00Z"
    assert session["deadline"] == "2026-11-02T15:00:00Z"
    assert re.fullmatch(r"pretrain-[0-9a-f]{12}-stop-setup", session["stop_schedule_name"])
    stop = env.cloud.schedules[session["stop_schedule_name"]]
    assert stop["ScheduleExpression"] == "at(2026-11-02T15:20:00)"
    assert stop["Target"]["RetryPolicy"] == {
        "MaximumRetryAttempts": 10,
        "MaximumEventAgeInSeconds": 3600,
    }
    assert json.loads(stop["Target"]["Input"]) == {"InstanceIds": ["i-test"], "Force": True}
    call = next(c for c in env.cloud.calls if c[2] == "run-instances")
    assert re.fullmatch(r"[0-9a-f]{12}", option(call, "--client-token"))
    assert option(call, "--client-token") == data["config"]["token"]
    resource_name = "pretrain-" + data["config"]["token"]
    assert option(call, "--key-name") == resource_name
    for action, flag in (
        ("create-key-pair", "--key-name"),
        ("create-security-group", "--group-name"),
    ):
        creation = next(c for c in env.cloud.calls if c[2] == action)
        assert option(creation, flag) == resource_name
    assert "--instance-market-options" not in call
    tags = json.loads(option(call, "--tag-specifications"))
    assert {t["ResourceType"] for t in tags} == {"instance", "volume"}
    assert all(t["Tags"] == [{"Key": "experiment", "Value": resource_name}] for t in tags)
    disk = json.loads(option(call, "--block-device-mappings"))[0]["Ebs"]
    assert disk == {"VolumeSize": 200, "VolumeType": "gp3", "DeleteOnTermination": True}
    for cmd, snapshot in zip(env.cloud.calls, env.cloud.snapshots, strict=True):
        assert snapshot["events"][-1]["utc"]
        assert snapshot["events"][-1]["stage"]
        if cmd[0] == "aws" and cmd[2] == "run-instances":
            assert snapshot["sessions"][0]["status"] == "pending"
            assert snapshot["sessions"][0]["aws_called"] == session["aws_called"]


def test_start_order_and_hold_consumption_are_atomic(env):
    assert env.provision() == 0
    assert env.run("stop") == 0
    assert env.run("hold", "--label", "eval", "--usd", "5") == 0
    first = len(env.cloud.calls)
    assert env.run("start", "--label", "eval", "--minutes", "60", "--consume-hold", "eval") == 0
    calls = env.cloud.calls[first:]
    start = next(i for i, c in enumerate(calls) if c[2] == "start-instances")
    create = next(i for i, c in enumerate(calls) if c[2] == "create-schedule")
    read = next(i for i, c in enumerate(calls) if c[2] == "get-schedule")
    assert create < read < start
    snapshot = env.cloud.snapshots[first + create]
    assert "eval" not in snapshot["holds"]
    assert snapshot["sessions"][-1]["reservation_usd"] > 1


@pytest.mark.parametrize("status", ["open", "pending"])
def test_start_refuses_existing_session(env, status):
    assert env.provision() == 0
    env.edit(lambda d: d["sessions"][0].update(status=status))
    assert env.run("start", "--label", "x", "--minutes", "10") == 6


def test_stop_settles_total_days_and_never_refuses_over_cap(env, capsys):
    assert env.provision() == 0
    env.clock.advance(25 * 3600)
    env.edit(lambda d: d["line_items"].append({"usd": 47}))
    assert env.run("stop") == 0
    data = env.ledger()
    assert data["sessions"][0]["settled_cost_usd"] >= 25.285
    assert data["sessions"][0]["status"] == "closed"
    assert data["sessions"][0]["stop_schedule_name"] not in env.cloud.schedules
    assert "OVER CAP" in capsys.readouterr().out


def test_reserved_crash_absence_retains_reservation_until_evidence(env):
    env.cloud.crash = lambda c: c[2] == "describe-key-pairs"
    with pytest.raises(Crash):
        env.provision()
    env.cloud.crash = None
    before = env.cg.committed(env.ledger())
    assert env.run("reconcile") == 0
    assert env.ledger()["sessions"][0]["status"] == "pending"
    assert env.cg.committed(env.ledger()) == before
    assert any("Name=client-token,Values=" in " ".join(c) for c in env.cloud.calls)
    txn = env.ledger()["sessions"][0]["txn_id"]
    assert env.run("cancel", "--txn", txn, "--evidence", "console: no instance") == 0
    assert env.ledger()["sessions"][0]["settled_cost_usd"] == 0


@pytest.mark.parametrize("after_schedules", [False, True])
def test_launch_crash_recovery_never_adopts_without_both_backstops(env, after_schedules):
    env.cloud.crash = lambda c: c[0] == "ssh" if after_schedules else c[2] == "create-schedule"
    with pytest.raises(Crash):
        env.provision()
    env.cloud.crash = None
    env.clock.advance(120)
    assert env.run("reconcile") == 0
    s = env.ledger()["sessions"][0]
    if after_schedules:
        assert s["status"] == "open"
    else:
        assert s["status"] == "closed"
        assert 0.03 < s["settled_cost_usd"] < 0.04
        assert env.cloud.instance["State"]["Name"] == "stopped"


def test_restart_crash_uses_its_own_aws_called_not_prior_launch(env):
    assert env.provision() == 0
    assert env.run("stop") == 0
    env.clock.advance(24 * 3600)
    env.cloud.crash = lambda c: c[0] == "ssh"
    with pytest.raises(Crash):
        env.run("start", "--label", "next", "--minutes", "60")
    env.cloud.crash = None
    env.cloud.schedules = {}
    env.clock.advance(120)
    assert env.run("reconcile") == 0
    assert 0.03 < env.ledger()["sessions"][-1]["settled_cost_usd"] < 0.04


def test_expiration_stops_flags_blocks_every_commitment_then_clears(env):
    assert env.provision(minutes=10) == 0
    env.clock.advance(31 * 60)
    assert env.run("reconcile") == 0
    assert env.ledger()["flags"]["EXPIRED_SESSION"] is True
    for command in [
        ["start", "--label", "x", "--minutes", "5"],
        ["provision", "--label", "x", "--minutes", "5"],
        ["hold", "--label", "eval", "--usd", "1"],
        ["transfer-begin", "--manifest", "/missing", "--attempt", "1"],
        ["extend-cleanup", "--to", "2026-11-10T12:00:00Z"],
    ]:
        assert env.run(*command) == 7
    assert env.run("reconcile") == 0
    assert not env.ledger()["flags"]["EXPIRED_SESSION"]


def test_unreserved_running_instance_is_stopped_and_costed(env):
    assert env.provision() == 0
    assert env.run("stop") == 0
    previous_stop = env.ledger()["sessions"][0]["stopped_at"]
    env.cloud.instance["State"]["Name"] = "running"
    env.clock.advance(3600)
    assert env.run("reconcile") == 0
    s = env.ledger()["sessions"][-1]
    assert s["label"].startswith("reconcile")
    assert s["aws_called"] == previous_stop
    assert s["settled_cost_usd"] > 1.0114


def test_launch_error_stops_and_cancels_after_schedule_creation_failure(env):
    env.cloud.fail = lambda c: c[2] == "create-schedule"
    assert env.provision() == 2
    assert env.cloud.instance["State"]["Name"] == "stopped"
    assert env.ledger()["sessions"][0]["status"] == "cancelled"
    assert env.ledger()["sessions"][0]["settled_cost_usd"] > 0
    stop = next(c for c in env.cloud.calls if c[2] == "stop-instances")
    assert "--force" in stop


def test_failed_trap_stop_retains_pending_reservation(env):
    env.cloud.fail = lambda c: c[2] in ("create-schedule", "stop-instances")
    assert env.provision() == 2
    assert env.ledger()["sessions"][0]["status"] == "pending"
    assert env.ledger()["sessions"][0]["settled_cost_usd"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("ScheduleExpressionTimezone", "Europe/Rome"),
        ("FlexibleTimeWindow", {"Mode": "FLEXIBLE", "MaximumWindowInMinutes": 2}),
        ("MaximumRetryAttempts", 9),
        ("MaximumEventAgeInSeconds", 3599),
        ("State", "DISABLED"),
    ],
)
def test_schedule_readback_mismatches_fail_closed(env, field, value):
    assert env.provision() == 0
    assert env.run("stop") == 0

    def corrupt(schedule):
        if field.startswith("Maximum"):
            schedule["Target"]["RetryPolicy"][field] = value
        else:
            schedule[field] = value

    env.cloud.corrupt = corrupt
    count = len(env.cloud.calls)
    assert env.run("start", "--label", "bad", "--minutes", "60") == 2
    assert not any(c[2] == "start-instances" for c in env.cloud.calls[count:])


@pytest.mark.parametrize(
    "delta,output", [(1, None), (-61, None), (0, "noise\nDEADLINE_SET 2026-11-02T15:00:00Z\n")]
)
def test_deadline_mismatch_stops_and_returns_four(env, delta, output):
    env.cloud.armed_delta = delta
    env.cloud.armed_output = output
    assert env.provision() == 4
    assert env.cloud.instance["State"]["Name"] == "stopped"
    assert env.ledger()["sessions"][0]["settled_cost_usd"] > 0


def test_cleanup_extension_is_ledger_first_and_preserves_configuration(env):
    assert env.provision() == 0
    cleanup = next(n for n in env.cloud.schedules if n.endswith("-cleanup"))
    env.cloud.schedules[cleanup].update(
        Description="keep me",
        KmsKeyArn="arn:key",
        GroupName="default",
        ActionAfterCompletion="NONE",
        StartDate="2026-11-01T00:00:00Z",
        EndDate="2027-01-01T00:00:00Z",
    )
    assert env.run("extend-cleanup", "--to", "2026-11-09T12:00:00Z") == 0
    index = next(i for i, c in enumerate(env.cloud.calls) if c[2] == "update-schedule")
    saved = env.cloud.snapshots[index]
    assert saved["config"]["cleanup_by"] == "2026-11-09T12:00:00Z"
    assert saved["holds"]["ebs"]["usd"] == 5.08
    schedule = env.cloud.schedules[cleanup]
    assert schedule["Description"] == "keep me" and schedule["KmsKeyArn"] == "arn:key"
    assert schedule["EndDate"] == "2027-01-01T00:00:00Z"
    assert schedule["ScheduleExpression"] == "at(2026-11-09T12:00:00)"


@pytest.mark.parametrize("command", ["status", "reconcile"])
def test_overdue_storage_flag(env, command, capsys):
    assert env.provision() == 0
    env.clock.advance(7 * 86400)
    assert env.run(command) == 0
    assert env.ledger()["flags"]["OVERDUE_STORAGE"]
    assert "OVERDUE_STORAGE" in capsys.readouterr().out


def test_transfer_attempts_partial_end_and_actual_overage(env, capsys):
    manifest = env.tmp / "manifest.json"
    manifest.write_text('{"files":[{"path":"weights", "bytes":1000000000}]}')
    assert env.run("transfer-begin", "--manifest", manifest, "--attempt", 1) == 0
    base = env.cg.committed(env.ledger())
    assert env.run("transfer-begin", "--manifest", manifest, "--attempt", 1) == 2
    assert env.run("transfer-end", "--manifest", manifest, "--attempt", 2, "--bytes", 10) == 2
    assert (
        env.run("transfer-end", "--manifest", manifest, "--attempt", 1, "--bytes", 500000000) == 0
    )
    assert env.cg.committed(env.ledger()) == pytest.approx(base - 0.045)
    assert env.run("transfer-begin", "--manifest", manifest, "--attempt", 2) == 0
    assert (
        env.run(
            "transfer-end", "--manifest", manifest, "--attempt", 2, "--bytes", 1_000_000_000_000
        )
        == 0
    )
    assert "OVER CAP" in capsys.readouterr().out


def test_record_never_refuses_and_explicit_cancel_keeps_incurred_cost(env, capsys):
    assert env.provision() == 0
    assert env.run("record", "--kind", "other", "--usd", "100", "--note", "invoice") == 0
    txn = env.ledger()["sessions"][0]["txn_id"]
    assert env.run("cancel", "--txn", txn, "--evidence", "console confirmed stopped") == 0
    assert env.ledger()["sessions"][0]["settled_cost_usd"] > 0
    assert env.cg.committed(env.ledger()) > 100
    assert "OVER CAP" in capsys.readouterr().out


def test_copyback_files_and_termination_storage_confirmation(env, capsys):
    assert env.provision() == 0
    assert env.run("terminate", "--abandon") == 2
    assert env.run("stop") == 0
    assert env.run("terminate") == 2
    manifest, report = env.tmp / "manifest", env.tmp / "sha.txt"
    manifest.write_text("{}")
    assert env.run("copyback-verified", "--manifest", manifest, "--sha-report", report) == 2
    report.write_text("verified\n")
    assert env.run("copyback-verified", "--manifest", manifest, "--sha-report", report) == 0
    assert env.ledger()["copyback_verified"]["utc"]
    env.edit(lambda d: d["line_items"].append({"usd": 100}))
    assert env.run("terminate") == 0
    assert env.cloud.schedules == {}
    assert "ebs" not in env.ledger()["holds"]
    assert any(item.get("kind") == "ebs" for item in env.ledger()["line_items"])
    assert "OVER CAP" in capsys.readouterr().out


def test_bash_syntax_and_missing_environment(tmp_path):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is not installed")
    script = ROOT / "scripts/ec2/pretrain_launch.sh"
    assert script.is_file()
    subprocess.run([bash, "-n", str(script)], check=True)
    clean = {
        k: v
        for k, v in os.environ.items()
        if k not in ("OPERATOR_IP", "VPC_ID", "SUBNET_ID", "AMI_ID")
    }
    result = subprocess.run(
        [bash, str(script), "provision"], env=clean, capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "OPERATOR_IP" in result.stderr


def test_wrapper_crash_trap_runs_real_guard_stop_and_cancel(env):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is not installed")
    shim = env.tmp / "offline-python"
    state = env.tmp / "cloud.json"
    calls = env.tmp / "calls.jsonl"
    # A real separate process dies at schedule creation. The wrapper must invoke
    # recovery in another process; all cloud actions still use the fake runner.
    shim.write_text(
        f"#!{sys.executable}\n"
        "import importlib.util, json, os, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from tests.test_bench.test_cost_guard import Clock, Cloud\n"
        f"spec = importlib.util.spec_from_file_location('guard', {str(SCRIPT)!r})\n"
        "guard = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(guard)\n"
        f"path, state, calls = Path({str(env.path)!r}), Path({str(state)!r}), Path({str(calls)!r})\n"
        "clock = Clock()\n"
        "cloud = Cloud(path, clock)\n"
        "if state.exists():\n"
        "    for key, value in json.loads(state.read_text()).items(): setattr(cloud, key, value)\n"
        "def run(cmd):\n"
        "    with calls.open('a') as stream: stream.write(json.dumps(cmd) + '\\n')\n"
        "    if cmd[2] == 'create-schedule': os._exit(9)\n"
        "    result = cloud(cmd)\n"
        "    state.write_text(json.dumps({k: getattr(cloud, k) for k in ('instance', 'volumes', 'schedules', 'key', 'sg')}))\n"
        "    return result\n"
        "raise SystemExit(guard.main(sys.argv[2:], run=run, now=clock, sleep=clock.sleep))\n"
    )
    shim.chmod(0o700)
    result = subprocess.run(
        [
            bash,
            str(ROOT / "scripts/ec2/pretrain_launch.sh"),
            "provision",
            "--ledger",
            str(env.path),
            "--label",
            "shell",
            "--minutes",
            "180",
            "--user-data",
            str(env.tmp / "guest.sh"),
        ],
        env=dict(os.environ, PYTHON=str(shim)),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 9, result.stdout + result.stderr
    actions = [json.loads(line)[2] for line in calls.read_text().splitlines()]
    assert (
        actions.index("run-instances")
        < actions.index("create-schedule")
        < actions.index("stop-instances")
    )
    session = env.ledger()["sessions"][0]
    assert session["status"] == "cancelled" and session["settled_cost_usd"] > 0
    txn = next(t for t in env.ledger()["transactions"] if t["id"] == session["txn_id"])
    assert txn["evidence"] == "trap"


@pytest.mark.parametrize(
    "command", ["stop", "reconcile", "cancel", "transfer-end", "record", "terminate"]
)
def test_every_recording_command_runs_over_cap(env, capsys, command):
    assert env.provision() == 0
    manifest = env.tmp / "transfer.json"
    manifest.write_text('{"bytes":1000000000}')
    if command == "transfer-end":
        assert env.run("transfer-begin", "--manifest", manifest, "--attempt", 1) == 0
    if command == "terminate":
        assert env.run("stop") == 0
    txn = env.ledger()["sessions"][0]["txn_id"]
    env.edit(lambda d: d["line_items"].append({"usd": 100, "kind": "other"}))
    argv = {
        "stop": [],
        "reconcile": [],
        "cancel": ["--txn", txn, "--evidence", "console confirmation"],
        "transfer-end": ["--manifest", manifest, "--attempt", 1, "--bytes", 10],
        "record": ["--kind", "ebs", "--usd", 1, "--note", "invoice adjustment"],
        "terminate": ["--abandon"],
    }[command]
    assert env.run(command, *argv) == 0
    assert "OVER CAP" in capsys.readouterr().out


@pytest.mark.parametrize("failure_stage", ["get-schedule", "update-schedule"])
def test_interrupted_extension_keeps_hold_and_is_recoverable(env, failure_stage):
    assert env.provision() == 0
    assert env.run("stop") == 0
    env.cloud.crash = lambda c: c[2] == failure_stage
    with pytest.raises(Crash):
        env.run("extend-cleanup", "--to", "2026-11-09T12:00:00Z")
    assert env.ledger()["holds"]["ebs"]["usd"] == 5.08
    env.cloud.crash = None
    assert env.run("reconcile") == 0
    schedule = env.cloud.schedules[env.ledger()["config"]["cleanup_schedule_name"]]
    assert schedule["ScheduleExpression"] == "at(2026-11-09T12:00:00)"


def test_start_on_unreserved_running_instance_does_not_erase_runtime(env):
    assert env.provision() == 0
    assert env.run("stop") == 0
    env.clock.advance(3600)
    env.cloud.instance["State"]["Name"] = "running"
    assert env.run("start", "--label", "new", "--minutes", "60") == 2
    assert env.cloud.instance["State"]["Name"] == "stopped"
    assert sum(s["settled_cost_usd"] or 0 for s in env.ledger()["sessions"]) > 1.02


def test_terminated_status_does_not_require_deleted_schedules(env):
    assert env.provision() == 0
    assert env.run("stop") == 0
    assert env.run("terminate", "--abandon") == 0
    assert env.ledger()["schedule_readback"] == {}


def test_non_default_rates_are_read_from_ledger(env):
    env.edit(
        lambda d: d["config"].update(
            compute_usd_per_h=2.0,
            ipv4_usd_per_h=0.1,
            egress_usd_per_gb=0.2,
            ebs_usd_per_gb_month=0.2,
        )
    )
    assert env.provision(minutes=40) == 0
    assert env.ledger()["sessions"][0]["reservation_usd"] == pytest.approx(2.1)
    env.clock.advance(3600)
    assert env.run("stop") == 0
    assert env.ledger()["sessions"][0]["settled_cost_usd"] >= 2.1
    manifest = env.tmp / "manifest.json"
    manifest.write_text('{"bytes":1000000000}')
    assert env.run("transfer-begin", "--manifest", manifest, "--attempt", 1) == 0
    assert (
        env.run("transfer-end", "--manifest", manifest, "--attempt", 1, "--bytes", 1_000_000_000)
        == 0
    )
    assert env.ledger()["transfers"][0]["cost_usd"] == 0.2


def test_saving_failure_cannot_replace_previous_complete_json(env, monkeypatch):
    before = env.path.read_bytes()

    def fail_replace(*args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(env.cg.os, "replace", fail_replace)
    assert env.run("hold", "--label", "eval", "--usd", 1) == 2
    assert env.path.read_bytes() == before
    assert not list(env.path.parent.glob("tmp*"))


def test_operator_commands_serialize_cap_checks(env):
    import fcntl

    lock_path = env.path.with_name(env.path.name + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert env.run("hold", "--label", "eval", "--usd", 1) == 2
    assert "eval" not in env.ledger()["holds"]


def test_pending_cleanup_repair_cannot_block_expired_instance_stop(env):
    assert env.provision(minutes=10) == 0
    env.cloud.fail = lambda c: c[2] == "update-schedule"
    assert env.run("extend-cleanup", "--to", "2026-11-09T12:00:00Z") == 2
    env.clock.advance(31 * 60)
    assert env.run("reconcile") == 2  # scheduler remains unavailable
    assert env.cloud.instance["State"]["Name"] == "stopped"
    assert env.ledger()["flags"]["EXPIRED_SESSION"]


def test_cannot_stack_cleanup_extensions_before_recovery(env):
    assert env.provision() == 0
    env.cloud.fail = lambda c: c[2] == "update-schedule"
    assert env.run("extend-cleanup", "--to", "2026-11-09T12:00:00Z") == 2
    before = env.path.read_bytes()
    assert env.run("extend-cleanup", "--to", "2026-11-10T12:00:00Z") == 2
    assert env.path.read_bytes() == before


def test_unreserved_stop_failure_retains_at_least_accrued_cost(env):
    assert env.provision() == 0
    assert env.run("stop") == 0
    env.cloud.instance["State"]["Name"] = "running"
    env.clock.advance(7200)
    env.cloud.fail = lambda c: c[2] == "stop-instances"
    assert env.run("reconcile") == 2
    session = env.ledger()["sessions"][-1]
    assert session["status"] == "pending"
    assert session["reservation_usd"] == pytest.approx(2.0228)


def test_aws_called_crash_before_response_keeps_unknown_launch_pending(env):
    env.cloud.crash = lambda c: c[2] == "run-instances"
    with pytest.raises(Crash):
        env.provision()
    env.cloud.crash = None
    session = env.ledger()["sessions"][0]
    assert session["aws_called"] is not None
    assert env.run("reconcile") == 0
    assert env.ledger()["sessions"][0]["status"] == "pending"
    assert env.ledger()["sessions"][0]["reservation_usd"] == session["reservation_usd"]


def test_missing_cleanup_blocks_crash_adoption_even_with_stop_schedule(env):
    env.cloud.crash = lambda c: c[0] == "ssh"
    with pytest.raises(Crash):
        env.provision()
    env.cloud.crash = None
    del env.cloud.schedules[env.ledger()["config"]["cleanup_schedule_name"]]
    assert env.run("reconcile") == 0
    assert env.cloud.instance["State"]["Name"] == "stopped"
    assert env.ledger()["sessions"][0]["status"] == "closed"


def test_ssh_failure_stops_with_exit_four(env):
    env.cloud.fail = lambda c: c[0] == "ssh"
    assert env.provision() == 4
    assert env.cloud.instance["State"]["Name"] == "stopped"


def test_create_role_trust_and_tag_scoped_policy(cg, tmp_path):
    path = tmp_path / "ledger.json"
    clock = Clock()
    cloud = Cloud(path, clock)
    cloud.role_exists = False
    assert (
        cg.main(
            [
                "--ledger",
                str(path),
                "init",
                "--cleanup-by",
                "2026-11-08T12:00:00Z",
                "--create-role",
            ],
            run=cloud,
            now=clock,
        )
        == 0
    )
    create = next(c for c in cloud.calls if c[2] == "create-role")
    policy = next(c for c in cloud.calls if c[2] == "put-role-policy")
    assert json.loads(option(create, "--assume-role-policy-document"))["Statement"][0][
        "Principal"
    ] == {"Service": "scheduler.amazonaws.com"}
    statement = json.loads(option(policy, "--policy-document"))["Statement"][0]
    assert statement["Action"] == ["ec2:StopInstances", "ec2:TerminateInstances"]
    assert statement["Condition"] == {
        "StringEquals": {
            "aws:ResourceTag/experiment": json.loads(path.read_text())["config"]["experiment"]
        }
    }


def test_cli_runner_forces_json_and_inherits_profile(cg, monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "operator-supplied")
    captured = {}

    def run(cmd, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, stdout="{}")

    monkeypatch.setattr(cg.subprocess, "run", run)
    assert cg.run_command(["aws", "ec2", "describe-instances"]) == "{}"
    assert captured["env"]["AWS_DEFAULT_OUTPUT"] == "json"
    assert captured["env"]["AWS_PROFILE"] == "operator-supplied"
    assert captured["check"] and captured["capture_output"]


@pytest.mark.parametrize(
    "failure,kind",
    [
        (
            (255, "", "ssh: connect to host 203.0.113.10 port 22: Connection refused"),
            "CONNECTION_REFUSED",
        ),
        (
            (255, "", "ssh: connect to host 203.0.113.10 port 22: Connection timed out"),
            "CONNECTION_TIMEOUT",
        ),
        (
            (127, "", "/usr/local/sbin/arm-deadline.sh: No such file or directory"),
            "BOOTSTRAP_NOT_FINISHED",
        ),
        (
            (1, "", "sudo: /usr/local/sbin/arm-deadline.sh: command not found"),
            "BOOTSTRAP_NOT_FINISHED",
        ),
    ],
)
def test_arming_retries_readiness_and_persists_each_attempt(env, capsys, failure, kind):
    env.cloud.ssh_errors = [failure, failure]
    assert env.provision() == 0
    assert env.clock.sleeps == [20, 20]
    data = env.ledger()
    session = data["sessions"][0]
    assert session["status"] == "open"
    assert session["arm_attempts"]["count"] == 3
    assert session["arm_attempts"]["last_error"]["kind"] == kind
    assert session["arm_attempts"]["utc"] == "2026-11-02T12:00:45Z"
    snapshots = [
        s for c, s in zip(env.cloud.calls, env.cloud.snapshots, strict=True) if c[0] == "ssh"
    ]
    assert [s["sessions"][0]["arm_attempts"]["count"] for s in snapshots] == [1, 2, 3]
    assert snapshots[0]["sessions"][0]["arm_attempts"]["last_error"] is None
    assert snapshots[1]["sessions"][0]["arm_attempts"]["last_error"]["kind"] == kind
    assert [s["sessions"][0]["arm_attempts"]["utc"] for s in snapshots] == [
        "2026-11-02T12:00:05Z",
        "2026-11-02T12:00:25Z",
        "2026-11-02T12:00:45Z",
    ]
    txn = next(t for t in data["transactions"] if t["id"] == session["txn_id"])
    assert sum(s["stage"] == "deadline_armed" for s in txn["stages"]) == 1
    assert sum(e["stage"] == "deadline_armed" for e in data["events"]) == 1
    assert capsys.readouterr().out.count(kind) == 2


def test_arming_pins_first_host_key_next_to_ledger(env):
    expected = str(env.path.parent / "known_hosts")
    assert env.ledger()["config"]["known_hosts_path"] == expected
    assert env.provision() == 0
    cmd, snapshot = next(
        (c, s) for c, s in zip(env.cloud.calls, env.cloud.snapshots, strict=True) if c[0] == "ssh"
    )
    assert "StrictHostKeyChecking=accept-new" in cmd
    assert "UserKnownHostsFile=" + expected in cmd
    assert "BatchMode=yes" in cmd and "ConnectTimeout=20" in cmd
    assert snapshot["config"]["known_hosts_path"] == expected
    assert not any("StrictHostKeyChecking=no" in arg for arg in cmd)
    assert cmd[-4:] == ["sudo", "/usr/local/sbin/arm-deadline.sh", "set", "2026-11-02T15:00:00Z"]


@pytest.mark.parametrize(
    "message,kind",
    [
        ("Host key verification failed.", "HOST_KEY_REFUSED"),
        ("ssh: connect to host 203.0.113.10 port 22: Connection timed out", "CONNECTION_TIMEOUT"),
    ],
)
def test_arming_stops_at_fifteen_minutes_from_aws_call(env, capsys, message, kind):
    env.cloud.ssh_errors = [(255, "", message)] * 50
    assert env.provision() == 4
    data = env.ledger()
    session = data["sessions"][0]
    assert session["status"] == "cancelled"
    assert session["settled_cost_usd"] == pytest.approx(905 / 3600 * 1.0114)
    assert session["arm_attempts"]["count"] == 45
    assert session["arm_attempts"]["last_error"]["kind"] == kind
    assert env.clock.sleeps == [20] * 44 + [15]
    assert env.cloud.instance["State"]["Name"] == "stopped"
    stop_snapshot = next(
        s
        for c, s in zip(env.cloud.calls, env.cloud.snapshots, strict=True)
        if c[2] == "stop-instances"
    )
    assert stop_snapshot["events"][-1]["utc"] == "2026-11-02T12:15:00Z"
    assert kind in capsys.readouterr().out


def test_arming_short_deadline_takes_precedence_over_readiness_window(env):
    env.cloud.ssh_errors = [(255, "", "Connection refused")] * 10
    assert env.provision(minutes=1) == 4
    session = env.ledger()["sessions"][0]
    assert session["arm_attempts"]["count"] == 3
    assert env.clock.sleeps == [20, 20, 15]
    assert session["status"] == "cancelled"
    assert session["settled_cost_usd"] == pytest.approx(65 / 3600 * 1.0114)


@pytest.mark.parametrize(
    "output,code,kind",
    [
        ("DEADLINE_MISMATCH unknown 2026-11-02T15:00:00Z\n", 1, "DEADLINE_MISMATCH"),
        ("DEADLINE_MISMATCH unknown 2026-11-02T15:00:00Z\n", 0, "DEADLINE_MISMATCH"),
        ("DEADLINE_SET 2026-11-02T15:00:01Z\n", 0, "DEADLINE_MISMATCH"),
        ("DEADLINE_SET invalid-time\n", 0, "BAD_OUTPUT"),
        ("unexpected guest output\n", 0, "BAD_OUTPUT"),
    ],
)
def test_arming_reachable_guest_bad_reply_is_final(env, capsys, output, code, kind):
    if code:
        env.cloud.ssh_errors = [(code, output, "")]
    else:
        env.cloud.armed_output = output
    assert env.provision() == 4
    session = env.ledger()["sessions"][0]
    assert session["arm_attempts"]["count"] == 1
    assert session["arm_attempts"]["last_error"]["kind"] == kind
    assert env.clock.sleeps == []
    assert env.cloud.instance["State"]["Name"] == "stopped"
    assert kind in capsys.readouterr().out


def test_arming_restart_uses_new_aws_call_for_readiness_bound(env):
    assert env.provision() == 0
    assert env.run("stop") == 0
    env.clock.advance(24 * 3600)
    env.cloud.ssh_errors = [(255, "", "Connection refused")]
    assert env.run("start", "--label", "next", "--minutes", 60) == 0
    assert env.ledger()["sessions"][-1]["arm_attempts"]["count"] == 2
    assert env.ledger()["sessions"][-1]["status"] == "open"


def test_arming_crash_keeps_pre_call_attempt_record(env):
    env.cloud.crash = lambda c: c[0] == "ssh"
    with pytest.raises(Crash):
        env.provision()
    session = env.ledger()["sessions"][0]
    assert session["status"] == "pending"
    assert session["arm_attempts"] == {
        "count": 1,
        "last_error": None,
        "utc": "2026-11-02T12:00:05Z",
    }


def test_arming_recovery_does_not_reset_fifteen_minute_window(env):
    env.cloud.crash = lambda c: c[0] == "ssh"
    with pytest.raises(Crash):
        env.provision()
    env.cloud.crash = None
    env.clock.advance(15 * 60)
    previous_calls = len(env.cloud.calls)
    assert env.run("reconcile") == 0
    assert not any(c[0] == "ssh" for c in env.cloud.calls[previous_calls:])
    assert env.cloud.instance["State"]["Name"] == "stopped"
