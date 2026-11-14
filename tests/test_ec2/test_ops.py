"""Synthetic operator integration: no credentials, guest, network or ML imports."""

import ctypes
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from process_identity import kill_owned

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "scripts/ec2/ops"

# The real wrappers invoke executable stubs, including a fake guard interface.
# Existing cost_guard tests cover its ledger/AWS implementation independently.
STUB = r"""
import ctypes, json, os, shlex, signal, sys, time
from pathlib import Path
from process_identity import process_identity
name = Path(sys.argv[0]).name
args = sys.argv[1:]
state_path = Path(os.environ["STUB_STATE"])
state = json.loads(state_path.read_text())
with Path(os.environ["STUB_CALLS"]).open("a") as out:
    out.write(json.dumps([name, *args]) + "\n")
def save():
    temporary = state_path.with_name(state_path.name + "." + str(os.getpid()))
    temporary.write_text(json.dumps(state))
    temporary.replace(state_path)
def finish(rc=0):
    save()
    sys.exit(rc)
def stall():
    state["stalled_pid"] = os.getpid()
    state["stalled_ppid"] = os.getppid()
    state["stalled_pid_identity"] = process_identity(os.getpid())
    state["stalled_ppid_identity"] = process_identity(os.getppid())
    if state.get("suspend_supervisor"):
        os.kill(os.getppid(), signal.SIGSTOP)
    save()
    time.sleep(60)
def alive(pid):
    if sys.platform == "darwin":
        # Inspect zombie state directly, including exiting libproc misses.
        mib = (ctypes.c_int * 4)(1, 14, 1, pid)
        data = ctypes.create_string_buffer(4096)
        size = ctypes.c_size_t(len(data))
        query = ctypes.CDLL(None, use_errno=True).sysctl
        if query(mib, 4, data, ctypes.byref(size), None, 0):
            raise OSError(ctypes.get_errno(), "process-state inspection failed")
        if not size.value:
            return False
        # SDK extern_proc: four pointers, flags, then p_stat.
        return data[4 * ctypes.sizeof(ctypes.c_void_p) + 4] != b"\x05"
    try:
        os.kill(pid, 0)
        if sys.platform == "linux":
            return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (ProcessLookupError, FileNotFoundError):
        return False
    return True
if name == "curl":
    print(state.get("operator_ip", "192.0.2.1"))
    finish()
if name == "aws":
    if args[:2] == ["ec2", "describe-security-groups"]:
        print(json.dumps({"SecurityGroups": [{"GroupId": "sg-test", "IpPermissions": [
            {"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
             "IpRanges": [{"CidrIp": ip} for ip in state.get("cidrs", ["192.0.2.1/32"])]}]}]}))
        finish()
    if args[:2] == ["ec2", "authorize-security-group-ingress"]:
        state.setdefault("cidrs", ["192.0.2.1/32"]).append(args[args.index("--cidr")+1])
        print('{}')
        finish()
    if state.get("real_guard") and args[:2] != ["sts", "get-caller-identity"]:
        from datetime import datetime, timezone
        ledger = json.loads(Path(os.environ["LEDGER"]).read_text())
        config = ledger["config"]
        service, action = args[:2]
        def opt(key): return args[args.index(key)+1]
        if service == "scheduler":
            name = opt("--name")
            schedules = state.setdefault("schedules", {})
            if action in ("create-schedule", "update-schedule"):
                schedules[name] = dict(Name=name, State=opt("--state"), ScheduleExpression=opt("--schedule-expression"),
                    ScheduleExpressionTimezone=opt("--schedule-expression-timezone"),
                    FlexibleTimeWindow=json.loads(opt("--flexible-time-window")), Target=json.loads(opt("--target")))
                print('{}')
            elif action == "get-schedule":
                if name not in schedules:
                    print('ResourceNotFoundException', file=sys.stderr)
                    finish(2)
                print(json.dumps(schedules[name]))
            elif action == "delete-schedule":
                schedules.pop(name, None)
                print('{}')
            else: raise AssertionError(args)
            finish()
        if service == "cloudtrail":
            print(json.dumps({"Events": [{"CloudTrailEvent": json.dumps(e)} for e in state.get("events", [])]}))
            finish()
        if action == "describe-volumes":
            print('{"Volumes": []}')
            finish()
        if action == "describe-instances":
            items = [] if state.get("absent") else [dict(InstanceId="i-synthetic",
                State={"Name": state.get("instance_state", "stopped")}, PublicIpAddress="192.0.2.10",
                StateTransitionReason=state.get("stop_reason", ""))]
            print(json.dumps({"Reservations": [{"Instances": items}]}))
            finish()
        if action == "start-instances":
            if state.get("capacity_failures", 0) or state.get("start_error"):
                state["capacity_failures"] = max(0, state.get("capacity_failures", 0)-1)
                code = state.get("start_error", "InsufficientInstanceCapacity")
                event = dict(eventName="StartInstances", eventSource="ec2.amazonaws.com", awsRegion=config["region"],
                    eventTime=ledger["sessions"][-1]["aws_called"], eventID="event-"+str(len(state.get("events", []))),
                    requestParameters={"instancesSet": {"items": [{"instanceId": "i-synthetic"}]}},
                    userIdentity={"arn": "arn:aws:iam::123456789012:user/synthetic"}, errorCode=code)
                if not state.get("missing_event"):
                    state.setdefault("events", []).append(event)
                print(code, file=sys.stderr)
                finish(2)
            state["instance_state"] = "running"
            print('{}')
            finish()
        if action == "stop-instances":
            if state.get("suspend_supervisor"):
                state["leaf_alive_at_stop"] = alive(state["stalled_pid"])
            state["instance_state"] = "stopped"
            state["stop_reason"] = "User initiated ("+datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")+" GMT)"
            print('{}')
            finish()
        if action == "wait":
            print('{}')
            finish()
        raise AssertionError(args)
    assert args[:2] == ["sts", "get-caller-identity"]
    assert "--output" in args and args[args.index("--output")+1] == "json"
    print('{"Account":"123456789012","Arn":"arn:aws:iam::123456789012:user/synthetic"}')
    finish(state.get("token_rc", 0))
if name == "sleep":
    finish()
if name == "cost_guard.py":
    ledger_path = Path(os.environ["LEDGER"])
    ledger = json.loads(ledger_path.read_text())
    args = [a for a in args if a not in ("--ledger", str(ledger_path))]
    operation = args[0]
    if operation == "start":
        if state.get("start_failures", 0):
            state["start_failures"] -= 1
            print("STOP_TIME_ESTIMATED unresolved: settle with evidence first")
            finish(2)
        label = args[args.index("--label")+1]
        if state.get("foreign_start"):
            label = "another-operator"
        session = dict(label=label, status="open", public_ip="192.0.2.10",
                       instance_id="i-synthetic", deadline=os.environ["TEST_DEADLINE"], txn_id="txn-1")
        ledger["sessions"].append(session)
        ledger_path.write_text(json.dumps(ledger))
        if state.get("foreign_start"):
            finish(6)
    elif operation == "status":
        print(json.dumps(dict(committed=1.23, sessions=ledger["sessions"], schedules={})))
        print("OVER CAP trailing diagnostic")
    elif operation in ("stop", "reconcile"):
        rc = state.get(operation + "_rc", 0)
        if not rc and not state.get("stop_rc"):
            for session in ledger["sessions"]:
                session.update(status="closed", stopped_at=os.environ["TEST_DEADLINE"], stop_time_estimated=False)
            ledger_path.write_text(json.dumps(ledger))
        finish(rc)
    elif operation == "transfer-begin":
        manifest = args[args.index("--manifest")+1]
        state.setdefault("reserved", []).append(json.loads(Path(manifest).read_text())["bytes"])
        finish(state.get("begin_rc", 0))
    elif operation == "transfer-end":
        state.setdefault("received", []).append(int(args[args.index("--bytes")+1]))
        finish(state.get("end_rc", 0))
    finish()
if name == "ssh":
    cmd = args[-1]
    if state.get("stall_contains") and state["stall_contains"] in cmd:
        stall()
    if state.get("ssh_failures", 0):
        state["ssh_failures"] -= 1
        finish(255)
    if cmd.startswith("sudo /usr/local/sbin/arm-deadline.sh set "):
        state["stop_fallback"] = True
        print(state.get("helper_reply", "DEADLINE_SET") + " " + shlex.split(cmd)[-1])
        finish(state.get("helper_rc", 0))
    if any("arm-deadline.sh" in a for a in args):
        print("DEADLINE_SET " + args[-1])
        finish()
    if cmd == "true" and state.get("stop_fallback"):
        finish(state.get("probe_rc", 255))
    if "wait-job.sh" in cmd:
        codes = state.get("wait_codes", [])
        finish(codes.pop(0) if codes else 0)
    if "dispatch-job.sh" in cmd:
        argv = shlex.split(cmd)
        label = argv[argv.index("/home/ubuntu/dispatch-job.sh")+1]
        if state.get("dispatch_bad"):
            print("unrecognised dispatch output")
        else:
            print("/home/ubuntu/fev/host-logs/" + label + ".ABC12345")
        finish(state.get("dispatch_rc", 0))
    if "du -sb" in cmd:
        print(state.get("size", 1024))
        finish(state.get("du_rc", 0))
    if "git checkout" in cmd:
        time.sleep(state.get("checkout_delay", 0))
        finish(state.get("checkout_rc", 0))
    if "SETUP_VERIFIED" in cmd:
        finish(state.get("marker_rc", 0))
    finish()
if name == "scp":
    if state.get("stall_scp"):
        stall()
    finish(state.get("scp_rc", 0))
if name == "rsync":
    if state.get("stall_rsync"):
        stall()
    if not state.get("no_stats"):
        if state.get("openrsync"):
            print("Total received: 1,234 B")
            print("sent 200 bytes  received 1,345 bytes  200 bytes/sec")
        else:
            print("Total bytes received: 1,234")
    finish(state.get("rsync_rc", 0))
raise RuntimeError(name)
"""


class Operator:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.bin = tmp / "bin"
        self.bin.mkdir()
        self.experiment = tmp / "experiment with spaces"
        (self.experiment / "ops").mkdir(parents=True)
        self.code = tmp / "code"
        ec2 = self.code / "scripts/ec2"
        ec2.mkdir(parents=True)
        self.state_path = tmp / "state.json"
        self.state_path.write_text("{}")
        self.calls_path = tmp / "calls.jsonl"
        self.ledger_path = self.experiment / "ops/cost-ledger.json"
        self.key = self.experiment / "test key.pem"
        self.key.write_text("synthetic private key")
        self.known = self.experiment / "known hosts"
        self.known.write_text("192.0.2.10 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEsynthetic\n")
        self.deadline = (datetime.now(timezone.utc) + timedelta(hours=2)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        self.env = dict(os.environ)
        for key in list(self.env):
            if key.startswith("AWS_") or key in ("OPERATOR_ENV", "BASH_ENV"):
                self.env.pop(key)
        self.env.update(
            PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
            LOCAL_CODE=str(self.code),
            EXPERIMENT=str(self.experiment),
            PYTHON=sys.executable,
            LEDGER=str(self.ledger_path),
            AWS_PROFILE="synthetic-profile",
            AWS_REGION="eu-central-1",
            VPC_ID="vpc-test",
            AMI_ID="ami-test",
            OPERATOR_IP="192.0.2.1",
            STUB_STATE=str(self.state_path),
            STUB_CALLS=str(self.calls_path),
            TEST_DEADLINE=self.deadline,
            PYTHONPATH=os.pathsep.join(
                (str(OPS), str(Path(__file__).parent), self.env.get("PYTHONPATH", ""))
            ),
        )
        for name in ("aws", "ssh", "scp", "rsync", "sleep", "curl"):
            self.executable(self.bin / name, f"#!{sys.executable}\n" + STUB)
        self.executable(ec2 / "cost_guard.py", STUB)
        self.executable(
            ec2 / "pretrain_launch.sh",
            '#!/bin/bash\nset -u\nexec "$PYTHON" "$LOCAL_CODE/scripts/ec2/cost_guard.py" "$@"\n',
        )
        self.reset_ledger(opened=True)

    @staticmethod
    def executable(path, text):
        path.write_text(text)
        path.chmod(0o755)

    def reset_ledger(self, *, opened):
        session = dict(
            label="test",
            txn_id="txn-1",
            status="open",
            public_ip="192.0.2.10",
            instance_id="i-synthetic",
            deadline=self.deadline,
        )
        self.ledger_path.write_text(
            json.dumps(
                dict(
                    config=dict(
                        key_path=str(self.key),
                        known_hosts_path=str(self.known),
                        region="eu-central-1",
                        security_group_id="sg-test",
                    ),
                    sessions=[session] if opened else [],
                )
            )
        )

    def state(self, **values):
        current = json.loads(self.state_path.read_text())
        current.update(values)
        self.state_path.write_text(json.dumps(current))

    def calls(self, name):
        if not self.calls_path.exists():
            return []
        return [
            call[1:]
            for line in self.calls_path.read_text().splitlines()
            if (call := json.loads(line))[0] == name
        ]

    def run(self, script, *args):
        return subprocess.run(
            ["/bin/bash", str(OPS / script), *map(str, args)],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=20,
        )

    def guard_operations(self):
        return [
            next(a for a in call if not a.startswith("--") and a != str(self.ledger_path))
            for call in self.calls("cost_guard.py")
        ]

    def chain(self, **state):
        self.reset_ledger(opened=False)
        script = self.tmp / "payload.sh"
        script.write_text("#!/bin/bash\nset -u\ntrue\n")
        self.env.update(
            LABEL="chain",
            MINUTES="60",
            SHA="a" * 40,
            GUEST_SCRIPT=str(script),
            GUEST_ARGS='["argument with spaces"]',
            PUSH="[]",
            PULL=json.dumps([dict(remote_dir="/results", local_dir=str(self.tmp / "rows"))]),
            EXPECTED_EXIT="0",
        )
        self.state(**state)
        return self.run("run-guest-job.sh")


@pytest.fixture
def operator(tmp_path):
    return Operator(tmp_path)


def test_start_refuses_expired_token(operator):
    operator.reset_ledger(opened=False)
    operator.state(token_rc=255)
    result = operator.run("session.sh", "start", "token", "60", "0")
    assert result.returncode == 2
    assert "aws sso login --profile synthetic-profile" in result.stderr
    assert not operator.calls("cost_guard.py")


def test_start_does_not_retry_unadmitted_failure(operator):
    operator.reset_ledger(opened=False)
    operator.state(start_failures=1)
    result = operator.run("session.sh", "start", "retry", "60", "1")
    assert result.returncode == 2
    assert "retry refused" in result.stderr
    assert not operator.calls("sleep")
    assert operator.guard_operations() == ["start"]


def test_host_preserves_spaces_in_ledger_paths(operator):
    result = operator.run("session.sh", "host")
    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip("\n").split("\t") == [
        str(operator.key),
        "192.0.2.10",
        str(operator.known),
        operator.deadline,
    ]


def test_remote_refuses_unpinned_host(operator):
    operator.known.write_text("192.0.2.99 ssh-ed25519 AAAA\n")
    result = operator.run("remote.sh", "ssh", "true")
    assert result.returncode != 0
    assert not operator.calls("ssh")


@pytest.mark.parametrize("reconcile_rc", [0, 2])
def test_stop_deadline_backstop_and_reconcile(operator, reconcile_rc):
    operator.state(stop_rc=2, reconcile_rc=reconcile_rc)
    result = operator.run("session.sh", "stop")
    assert result.returncode == 2
    assert operator.guard_operations() == ["stop", "reconcile"]
    calls = operator.calls("ssh")
    assert calls[0][-1].startswith("sudo /usr/local/sbin/arm-deadline.sh set ")
    assert calls[1][-1] == "true"
    assert "StrictHostKeyChecking=yes" in calls[0]
    evidence = json.loads((operator.experiment / "ops/stop.test.json").read_text())
    assert evidence["shutdown_ssh_exit"] == 0
    assert evidence["deadline_reply"] == "DEADLINE_SET " + evidence["requested_deadline"]
    assert evidence["first_failed_ssh_at"].endswith("Z")
    datetime.strptime(evidence["first_failed_ssh_at"], "%Y-%m-%dT%H:%M:%SZ")


@pytest.mark.parametrize(
    "now, expected",
    [
        ("2026-11-11T12:30:00Z", "2026-11-11T12:31:00Z"),
        ("2026-11-11T12:30:00.001Z", "2026-11-11T12:32:00Z"),
        ("2026-11-11T23:59:59.999Z", "2026-11-12T00:01:00Z"),
    ],
)
def test_stop_fallback_uses_earliest_safe_helper_deadline(operator, monkeypatch, now, expected):
    monkeypatch.syspath_prepend(str(OPS))
    import session_control

    for name, value in operator.env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("GUEST_USER", "ubuntu")
    current = datetime.fromisoformat(now.replace("Z", "+00:00")).timestamp()
    monkeypatch.setattr(session_control.time, "time", lambda: current)
    operator.state(stop_rc=2, probe_rc=0)

    assert session_control.stop() == 2
    command = operator.calls("ssh")[0][-1]
    assert shlex.split(command) == ["sudo", "/usr/local/sbin/arm-deadline.sh", "set", expected]
    assert "shutdown" not in command
    requested = datetime.strptime(expected, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    assert 60 <= requested.timestamp() - current < 120
    evidence = json.loads((operator.experiment / "ops/stop.test.json").read_text())
    assert evidence["requested_deadline"] == expected
    assert evidence["deadline_reply"] == "DEADLINE_SET " + expected
    assert evidence["shutdown_ssh_exit"] == 0
    assert evidence["first_failed_ssh_at"] is None
    assert json.loads(operator.ledger_path.read_text())["sessions"][0]["status"] == "open"


@pytest.mark.parametrize(
    "helper_rc, helper_reply",
    [(3, "DEADLINE_NOT_READY"), (1, "DEADLINE_MISMATCH"), (0, "UNEXPECTED_REPLY")],
)
def test_stop_reports_unconfirmed_helper_reply(operator, helper_rc, helper_reply):
    operator.state(stop_rc=2, helper_rc=helper_rc, helper_reply=helper_reply)
    result = operator.run("session.sh", "stop")
    assert result.returncode == 2
    evidence = json.loads((operator.experiment / "ops/stop.test.json").read_text())
    assert evidence["shutdown_ssh_exit"] == helper_rc
    assert evidence["deadline_reply"] == helper_reply + " " + evidence["requested_deadline"]
    assert evidence["deadline_reply"] in result.stderr
    assert f"exit={helper_rc}" in result.stderr
    assert json.loads(operator.ledger_path.read_text())["sessions"][0]["status"] == "open"


def test_stop_reconciles_when_no_usable_ssh_key(operator):
    operator.key.unlink()
    operator.state(stop_rc=2)
    result = operator.run("session.sh", "stop")
    assert result.returncode == 2
    assert operator.guard_operations() == ["stop", "reconcile"]
    assert "SSH stop fallback skipped: missing connection record" in result.stderr
    assert not operator.calls("ssh")


def test_stop_reports_incomplete_connection_record(operator, monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(OPS))
    import session_control

    for name in ("PYTHON", "LOCAL_CODE", "LEDGER"):
        monkeypatch.setenv(name, operator.env[name])
    monkeypatch.setattr(
        session_control, "execute", lambda *args, **kwargs: "key\thost\tknown\tdeadline\n"
    )
    monkeypatch.setattr(session_control, "guard", lambda *args, **kwargs: None)
    assert session_control.stop() == 2
    assert (
        "SSH stop fallback skipped: connection record has 4 fields; expected 7"
        in capsys.readouterr().err
    )


def test_stop_reconciles_with_errexit_in_operator_env(operator):
    env = operator.tmp / "operator-env.sh"
    env.write_text("set -e\n")
    operator.env["OPERATOR_ENV"] = str(env)
    operator.state(stop_rc=2)
    result = operator.run("session.sh", "stop")
    assert result.returncode == 2
    assert operator.guard_operations() == ["stop", "reconcile"]


def test_status_trailing_text_is_valid_json(operator):
    result = operator.run("session.sh", "status")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["committed"] == 1.23


@pytest.mark.parametrize("codes, expected_rc", [([255, 124, 0], 0), ([255, 1], 1)])
def test_wait_repolls_without_redispatch(operator, codes, expected_rc):
    operator.state(wait_codes=codes)
    result = operator.run("remote.sh", "wait", "/saved/job", "3", int(time.time()) + 2000)
    assert result.returncode == expected_rc, result.stderr
    calls = operator.calls("ssh")
    assert len(calls) == len(codes)
    for call in calls:
        argv = shlex.split(call[-1])
        assert argv[:4] == ["bash", "/home/ubuntu/wait-job.sh", "/saved/job", "3"]
        assert 0 < int(argv[-1]) <= 600
        assert "dispatch-job.sh" not in call[-1]


def test_wait_past_deadline_never_polls(operator):
    result = operator.run("remote.sh", "wait", "/saved/job", "0", int(time.time()) - 1)
    assert result.returncode == 124
    assert not operator.calls("ssh")


@pytest.mark.parametrize("openrsync, expected", [(False, 1234), (True, 1345)])
def test_pull_accounts_real_stats_and_excludes_large_artifacts(operator, openrsync, expected):
    operator.state(openrsync=openrsync)
    result = operator.run(
        "remote.sh", "pull", "/guest/results", operator.tmp / "rows", "--exclude", "*.npz"
    )
    assert result.returncode == 0, result.stderr
    assert operator.guard_operations() == ["transfer-begin", "transfer-end"]
    state = json.loads(operator.state_path.read_text())
    assert state["received"] == [expected]
    assert state["reserved"] == [1024 + 4 * 1024 * 1024]
    argv = operator.calls("rsync")[0]
    assert all(pattern in argv for pattern in ("bundle/", "checkpoints/", "*.npz"))
    assert "StrictHostKeyChecking=yes" in argv[argv.index("-e") + 1]


def test_pull_failure_still_settles_observed_bytes(operator):
    operator.state(rsync_rc=23)
    result = operator.run("remote.sh", "pull", "/guest/results", operator.tmp / "rows")
    assert result.returncode == 23
    assert operator.guard_operations() == ["transfer-begin", "transfer-end"]


def test_pull_without_statistics_retains_hold(operator):
    operator.state(no_stats=True)
    result = operator.run("remote.sh", "pull", "/guest/results", operator.tmp / "rows")
    assert result.returncode != 0
    assert operator.guard_operations() == ["transfer-begin"]


def test_pull_no_transfer_when_reservation_fails(operator):
    operator.state(begin_rc=2)
    result = operator.run("remote.sh", "pull", "/guest/results", operator.tmp / "rows")
    assert result.returncode != 0
    assert not operator.calls("rsync")


def test_push_preserves_local_and_guest_spaces(operator):
    source = operator.tmp / "input with spaces.json"
    source.write_text("{}")
    result = operator.run("remote.sh", "push", source, "/guest/with spaces")
    assert result.returncode == 0, result.stderr
    assert operator.calls("scp")[0][-2:] == [str(source), "ubuntu@192.0.2.10:/guest/with spaces/"]


def test_chain_full_sequence_and_saved_outer_job(operator):
    result = operator.chain()
    assert result.returncode == 0, result.stderr
    saved = operator.experiment / "ops/outer-job.chain.txt"
    assert saved.read_text().strip() == "/home/ubuntu/fev/host-logs/chain.ABC12345"
    ops = operator.guard_operations()
    assert ops[0] == "start" and ops[-2:] == ["stop", "reconcile"]
    assert ops.count("transfer-begin") == 3  # results, setup evidence, job evidence
    ssh = [call[-1] for call in operator.calls("ssh")]
    checkout = next(i for i, cmd in enumerate(ssh) if "git checkout" in cmd)
    setup = next(i for i, cmd in enumerate(ssh) if "gates-chain" in cmd and "dispatch-job" in cmd)
    outer = next(
        i for i, cmd in enumerate(ssh) if "dispatch-job" in cmd and "gates-chain" not in cmd
    )
    assert checkout < setup < outer
    assert "argument with spaces" in shlex.split(ssh[outer])[-1]


@pytest.mark.parametrize(
    "failure",
    [
        dict(checkout_rc=7),
        dict(scp_rc=9),
        dict(wait_codes=[1]),
        dict(marker_rc=1),
        dict(wait_codes=[0, 1]),
        dict(rsync_rc=23),
        dict(dispatch_rc=255),
        dict(dispatch_bad=True),
    ],
)
def test_chain_failures_always_stop_and_reconcile(operator, failure):
    result = operator.chain(**failure)
    assert result.returncode != 0
    assert operator.guard_operations()[-2:] == ["stop", "reconcile"], result.stderr
    if failure.get("dispatch_rc") or failure.get("dispatch_bad"):
        dispatches = [call for call in operator.calls("ssh") if "dispatch-job.sh" in call[-1]]
        assert len(dispatches) == 1


def test_chain_guest_failure_still_copies_results_and_stops(operator):
    result = operator.chain(wait_codes=[0, 1])
    assert result.returncode == 1, result.stderr
    assert any("ubuntu@192.0.2.10:/results/" in call for call in operator.calls("rsync"))
    assert operator.guard_operations()[-2:] == ["stop", "reconcile"]
    assert json.loads(operator.ledger_path.read_text())["sessions"][0]["status"] == "closed"


@pytest.mark.parametrize("initial_sha", [None, "b" * 40, "a" * 40])
def test_chain_checkout_env_sha_is_idempotent(operator, initial_sha):
    guest_code = operator.tmp / "guest code"
    (guest_code / ".git").mkdir(parents=True)
    guest_env = operator.tmp / "guest env.sh"
    original = "export OTHER=retained"
    if initial_sha:
        original += f" FEV_SHA={initial_sha}"
    original += "\n"
    guest_env.write_text(original)
    operator.env.update(GUEST_CODE=str(guest_code), GUEST_ENV=str(guest_env), FEV_SHA="a" * 40)
    operator.executable(
        operator.bin / "git",
        '#!/bin/bash\nif [[ $1 == rev-parse ]]; then printf "%s\\n" "$SHA"; fi\nexit 0\n',
    )
    for _ in range(2):
        result = operator.chain()
        assert result.returncode == 0, result.stderr
        checkout = [call[-1] for call in operator.calls("ssh") if "git checkout" in call[-1]][-1]
        result = subprocess.run(
            ["/bin/bash", "-c", checkout],
            env=operator.env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0, result.stderr
    contents = guest_env.read_text()
    assert contents.startswith(original)
    assert contents.count("FEV_SHA=" + "a" * 40) == 1


def test_chain_skip_setup_is_explicit(operator):
    operator.env["SKIP_SETUP"] = "1"
    result = operator.chain()
    assert result.returncode == 0, result.stderr
    assert not any("gates-chain" in call[-1] for call in operator.calls("ssh"))


def test_chain_token_expiry_during_stop_records_evidence_and_reconciles(operator):
    result = operator.chain(stop_rc=2, reconcile_rc=2)
    assert result.returncode != 0
    assert operator.guard_operations()[-2:] == ["stop", "reconcile"]
    evidence = operator.experiment / "ops/stop.chain.json"
    assert json.loads(evidence.read_text())["first_failed_ssh_at"]


def test_chain_failed_dispatch_preserves_reply_without_redispatch(operator):
    result = operator.chain(dispatch_rc=255)
    assert result.returncode != 0
    assert (
        (operator.experiment / "ops/setup-job.chain.txt")
        .read_text()
        .startswith("/home/ubuntu/fev/host-logs/gates-chain.")
    )
    assert len([call for call in operator.calls("ssh") if "dispatch-job.sh" in call[-1]]) == 1


def test_chain_does_not_stop_another_operators_session(operator):
    result = operator.chain(foreign_start=True)
    assert result.returncode != 0
    assert "refusing to stop" in result.stderr
    assert "stop" not in operator.guard_operations()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Total bytes received: 1,234\n", 1234),
        ("Total received: 1,234 B\nsent 10 bytes  received 1,345 bytes  10 bytes/sec\n", 1345),
        ("sent 0 bytes  received 0 bytes\n", 0),
    ],
)
def test_rsync_received_formats(tmp_path, text, expected):
    log = tmp_path / "rsync.log"
    log.write_text(text)
    result = subprocess.run(
        [sys.executable, str(OPS / "rsync_received.py"), str(log)], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert int(result.stdout) == expected


def test_guest_wait_expected_mismatch_and_unknown(tmp_path):
    script = str(OPS / "guest/wait-job.sh")
    status = tmp_path / "exit"
    status.write_text("3\n")
    assert (
        subprocess.run(
            ["/bin/bash", script, str(tmp_path), "3", "1"], capture_output=True
        ).returncode
        == 0
    )
    assert (
        subprocess.run(
            ["/bin/bash", script, str(tmp_path), "0", "1"], capture_output=True
        ).returncode
        == 1
    )
    status.unlink()
    assert (
        subprocess.run(
            ["/bin/bash", script, str(tmp_path), "0", "1"], capture_output=True
        ).returncode
        == 124
    )


def wins_env(tmp_path):
    guest_env = tmp_path / "env.sh"
    guest_env.write_text(
        f"export TOOLS={shlex.quote(str(ROOT))}\nexport OPS={shlex.quote(str(tmp_path / 'ops'))}\nexport PY={shlex.quote(sys.executable)}\n"
    )
    return dict(os.environ, GUEST_ENV=str(guest_env))


def run_wins(tmp_path, items):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(items))
    return subprocess.run(
        ["/bin/bash", str(OPS / "guest/wins-guest.sh"), "synthetic", str(manifest)],
        env=wins_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=15,
    )


@pytest.mark.parametrize(
    "commands, expected_rc, summary",
    [
        (["true", "true"], 0, "ok=2 failed=0"),
        (["true", "false", "true"], 1, "ok=2 failed=1"),
        (["false", "false"], 1, "ok=0 failed=2"),
    ],
)
def test_wins_true_false_summary_and_resume(tmp_path, commands, expected_rc, summary):
    items = [dict(name=f"item{i}", lane="light", cmd=cmd) for i, cmd in enumerate(commands)]
    result = run_wins(tmp_path, items)
    assert result.returncode == expected_rc, result.stderr
    assert result.stdout.splitlines()[-1] == f"WINS_DONE {summary}"
    resumed = run_wins(tmp_path, items)
    assert resumed.returncode == expected_rc
    assert resumed.stdout.count("WINS_START") == len(commands) + commands.count("false")
    assert len(list((tmp_path / "ops/wins/synthetic").glob("*.log"))) == len(items)


@pytest.fixture
def wins_probe(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text("""import fcntl, json, subprocess, sys, time
from pathlib import Path
events = Path(sys.argv[1])
def record(kind):
    with events.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps([kind, sys.argv[2]]) + "\\n")
record("start")
time.sleep(0.2)
rc = subprocess.run(sys.argv[3:]).returncode
record("end")
sys.exit(rc)
""")
    events = tmp_path / "events.jsonl"
    return [sys.executable, str(probe), str(events)], events


@pytest.mark.parametrize("light_lanes", [None, 1, 8])
def test_wins_lane_concurrency_and_heavy_barrier(tmp_path, monkeypatch, wins_probe, light_lanes):
    if light_lanes is None:
        monkeypatch.delenv("WINS_LANES", raising=False)
    else:
        monkeypatch.setenv("WINS_LANES", str(light_lanes))
    probe, events = wins_probe
    lanes = ["light"] * 4 + ["heavy"] + ["light"] * 3
    items = [
        dict(
            name=f"item{i}",
            lane=lane,
            cmd=[*probe, f"item{i}", "true" if i % 2 else "false"],
        )
        for i, lane in enumerate(lanes)
    ]
    result = run_wins(tmp_path, items)
    assert result.returncode == 1, result.stderr
    active, maximum = set(), 0
    started, retries = set(), []
    for event, name in map(json.loads, events.read_text().splitlines()):
        if event == "start":
            if name in started:
                assert len(started) == len(items)
                assert not active
                retries.append(name)
            started.add(name)
            active.add(name)
        else:
            active.remove(name)
        maximum = max(maximum, len(active))
        assert len(active) <= (light_lanes or 3)
        if retries:
            assert len(active) <= 1
        if "item4" in active:
            assert active == {"item4"}
    assert not active and maximum == min(light_lanes or 3, 4)
    assert retries == ["item0", "item2", "item4", "item6"]
    assert [line for line in result.stdout.splitlines() if line.startswith("WINS_RETRY")] == [
        f"WINS_RETRY {name}" for name in retries
    ]
    assert result.stdout.splitlines()[-1] == "WINS_DONE ok=4 failed=4"


@pytest.mark.parametrize("light_lanes", ["0", "x", "9"])
def test_wins_invalid_lanes_start_nothing(tmp_path, monkeypatch, wins_probe, light_lanes):
    monkeypatch.setenv("WINS_LANES", light_lanes)
    probe, events = wins_probe
    result = run_wins(tmp_path, [dict(name="item", lane="light", cmd=[*probe, "item", "true"])])
    assert result.returncode == 2
    assert "WINS_ERROR" in result.stderr
    assert "WINS_START" not in result.stdout
    assert not events.exists()


def test_wins_retry_recovers_and_keeps_both_attempts(tmp_path):
    flag = tmp_path / "first-attempt"
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; p = Path(sys.argv[1]); "
        "existed = p.exists(); p.touch(); sys.exit(0 if existed else 1)",
        str(flag),
    ]
    result = run_wins(tmp_path, [dict(name="item", lane="light", cmd=command)])
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "WINS_START item lane=light",
        "WINS_EXIT item exit=1",
        "WINS_RETRY item",
        "WINS_START item lane=light",
        "WINS_EXIT item exit=0",
        "WINS_DONE ok=1 failed=0",
    ]
    state = tmp_path / "ops/wins/synthetic"
    assert json.loads((state / "item.json").read_text()) == dict(argv=command, exit=0)
    assert (
        sum(line.startswith("COMMAND ") for line in (state / "item.log").read_text().splitlines())
        == 2
    )


@pytest.mark.parametrize(
    "command, retry_failed, attempts, expected_exit",
    [
        (["false"], "1", 2, 1),
        (["false"], "0", 1, 1),
        (["/nonexistent/wins-test-command"], "1", 1, 127),
        ([sys.executable, "-c", "raise SystemExit(127)"], "1", 2, 127),
    ],
)
def test_wins_retry_attempt_limit(
    tmp_path, monkeypatch, command, retry_failed, attempts, expected_exit
):
    monkeypatch.setenv("WINS_RETRY_FAILED", retry_failed)
    result = run_wins(tmp_path, [dict(name="item", lane="light", cmd=command)])
    assert result.returncode == 1, result.stderr
    assert result.stdout.count("WINS_START") == attempts
    assert result.stdout.count("WINS_RETRY") == attempts - 1
    assert result.stdout.count(f"WINS_EXIT item exit={expected_exit}") == attempts
    assert result.stdout.splitlines()[-1] == "WINS_DONE ok=0 failed=1"
    state = tmp_path / "ops/wins/synthetic"
    assert json.loads((state / "item.json").read_text())["exit"] == expected_exit
    assert (
        sum(line.startswith("COMMAND ") for line in (state / "item.log").read_text().splitlines())
        == attempts
    )


def test_wins_resume_reaches_cli_even_after_previous_success(tmp_path, monkeypatch):
    cli = tmp_path / "demand-forecast"
    calls = tmp_path / "cli.jsonl"
    cli.write_text(
        f'#!{sys.executable}\nimport json, sys\nwith open({str(calls)!r}, "a") as out:\n    out.write(json.dumps(sys.argv[1:]) + "\\n")\n'
    )
    cli.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    items = [
        dict(name=kind, lane="light", cmd=f"demand-forecast fev {kind} --out-dir /runs/{kind}")
        for kind in ("run", "tune")
    ]
    assert run_wins(tmp_path, items).returncode == 0
    assert run_wins(tmp_path, items).returncode == 0
    commands = list(map(json.loads, calls.read_text().splitlines()))
    assert len(commands) == 4
    assert all(argv.count("--resume") == 1 for argv in commands)


@pytest.mark.skipif(
    not shutil.which("setsid"), reason="Ubuntu-only setsid is absent on the macOS operator"
)
def test_guest_dispatch_detaches_and_atomically_publishes_exit(tmp_path):
    result = subprocess.run(
        ["/bin/bash", str(OPS / "guest/dispatch-job.sh"), "synthetic", "bash", "-c", "exit 3"],
        env=dict(os.environ, HOST_LOGS=str(tmp_path)),
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0
    job = Path(result.stdout.strip())
    waited = subprocess.run(
        ["/bin/bash", str(OPS / "guest/wait-job.sh"), str(job), "3", "5"], capture_output=True
    )
    assert waited.returncode == 0
    assert (job / "exit").read_text() == "3\n"
    assert not (job / "exit.tmp").exists()


def use_real_guard(operator):
    """Real persisted guard transactions; only aws/ssh/curl executables are fake."""
    import importlib.util

    shutil.copyfile(ROOT / "scripts/ec2/cost_guard.py", operator.code / "scripts/ec2/cost_guard.py")
    shutil.copyfile(
        ROOT / "scripts/ec2/pretrain_launch.sh", operator.code / "scripts/ec2/pretrain_launch.sh"
    )
    now = datetime.now(timezone.utc)
    config = dict(
        key_path=str(operator.key),
        known_hosts_path=str(operator.known),
        region="eu-central-1",
        security_group_id="sg-test",
        token="synthetic",
        experiment="synthetic",
        created=(now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        cleanup_by=(now + timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        cleanup_schedule_name="cleanup",
        role_arn="arn:aws:iam::123456789012:role/test",
        compute_usd_per_h=1.0064,
        ipv4_usd_per_h=0.005,
        ebs_usd_per_gb_month=0.0952,
        egress_usd_per_gb=0.09,
        volume_gb=200,
        cap_usd=50,
        enforce_usd=47,
        ready=True,
        type="g6.xlarge",
        market="on-demand",
    )
    data = dict(
        schema_version=1,
        config=config,
        sessions=[],
        transactions=[],
        events=[],
        flags={},
        holds={},
        transfers=[],
        line_items=[],
        instance_id="i-synthetic",
        volume_ids=[],
    )
    operator.ledger_path.write_text(json.dumps(data))
    spec = importlib.util.spec_from_file_location(
        "integration_cost_guard", ROOT / "scripts/ec2/cost_guard.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    guard = module.Guard(operator.ledger_path)
    guard.ledger = data
    operator.state(
        real_guard=True,
        instance_state="stopped",
        schedules={"cleanup": guard.cleanup_schedule("i-synthetic")},
    )
    return data


def test_real_guard_capacity_failure_settles_then_uses_fresh_label(operator):
    use_real_guard(operator)
    operator.state(capacity_failures=1)
    result = operator.run("session.sh", "start", "capacity", "60", "1")
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("OPEN capacity-r2 ")
    data = json.loads(operator.ledger_path.read_text())
    assert [(s["label"], s["status"]) for s in data["sessions"]] == [
        ("capacity", "cancelled"),
        ("capacity-r2", "open"),
    ]
    assert data["sessions"][0]["stop_time_estimated"] is False
    txn = next(t for t in data["transactions"] if t["id"] == data["sessions"][0]["txn_id"])
    assert "CloudTrail event-0" in txn["stop_time_correction"]["evidence"]
    assert operator.calls("sleep") == [["300"]]
    assert len([a for a in operator.calls("aws") if a[:2] == ["ec2", "start-instances"]]) == 2


@pytest.mark.parametrize(
    "state",
    [dict(start_error="UnauthorizedOperation"), dict(capacity_failures=1, missing_event=True)],
)
def test_real_guard_never_retries_noncapacity_or_missing_evidence(operator, state):
    use_real_guard(operator)
    operator.state(**state)
    result = operator.run("session.sh", "start", "refuse", "60", "2")
    assert result.returncode != 0
    assert not operator.calls("sleep")
    assert len([a for a in operator.calls("aws") if a[:2] == ["ec2", "start-instances"]]) == 1


def test_real_guard_estimated_stop_cannot_report_success(operator):
    use_real_guard(operator)
    assert operator.run("session.sh", "start", "estimate", "60", "0").returncode == 0
    operator.state(instance_state="stopped", stop_reason="")
    result = operator.run("session.sh", "stop")
    assert result.returncode == 2, result.stderr
    data = json.loads(operator.ledger_path.read_text())
    assert data["sessions"][0]["status"] == "closed"
    assert data["sessions"][0]["stop_time_estimated"] is True
    assert "'--ledger'" not in result.stderr  # shlex emits a copyable command
    assert "settle --txn " + data["sessions"][0]["txn_id"] in result.stderr
    assert "--stopped-at '<confirmed-stop-UTC>'" in result.stderr
    assert any(
        c[-1].startswith("sudo /usr/local/sbin/arm-deadline.sh set ") for c in operator.calls("ssh")
    )
    assert len([t for t in data["transactions"] if t["command"] == "stop"]) == 1
    assert len([t for t in data["transactions"] if t["command"] == "reconcile"]) == 1


def test_real_guard_empty_discovery_retains_pending_and_falls_back(operator):
    use_real_guard(operator)
    assert operator.run("session.sh", "start", "absent", "60", "0").returncode == 0
    operator.state(absent=True)
    result = operator.run("session.sh", "stop")
    assert result.returncode == 2, result.stderr
    assert json.loads(operator.ledger_path.read_text())["sessions"][0]["status"] == "open"
    assert "Stop unconfirmed" in result.stderr
    assert any(
        c[-1].startswith("sudo /usr/local/sbin/arm-deadline.sh set ") for c in operator.calls("ssh")
    )


def prepare_real_chain(operator, **state):
    # Prepare the input environment with the existing fixture, then replace its
    # fake guard. Do not run the fixture's chain/reset after this point.
    script = operator.tmp / "payload.sh"
    script.write_text("#!/bin/bash\ntrue\n")
    operator.env.update(
        LABEL="real-chain",
        MINUTES="60",
        SHA="a" * 40,
        GUEST_SCRIPT=str(script),
        GUEST_ARGS="[]",
        PUSH="[]",
        PULL="[]",
        SKIP_SETUP="1",
    )
    use_real_guard(operator)
    operator.state(**state)


def real_chain(operator, **state):
    prepare_real_chain(operator, **state)
    return operator.run("run-guest-job.sh")


@contextmanager
def running_real_chain(operator):
    child = subprocess.Popen(
        ["/bin/bash", str(OPS / "run-guest-job.sh")],
        env=operator.env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        yield child
    except subprocess.TimeoutExpired as exc:
        # communicate's TimeoutExpired message omits its captured stderr. Keep
        # the owner's periodic drain diagnostics in a hanging test's failure.
        def text(data):
            return data.decode(errors="replace") if isinstance(data, bytes) else data or ""

        pytest.fail(
            f"Local operator timed out after {exc.timeout}s\n"
            f"stdout:\n{text(exc.stdout)}\nstderr:\n{text(exc.stderr)}"
        )
    finally:
        # Also clean up the deliberately suspended supervisor/leaf on the old
        # implementation, so a failing regression never leaves fake transports.
        state = json.loads(operator.state_path.read_text())
        for field in ("stalled_pid", "stalled_ppid"):
            if pid := state.get(field):
                kill_owned(pid, state.get(field + "_identity"))
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=30)


def process_alive(pid):
    if sys.platform == "darwin":
        mib = (ctypes.c_int * 4)(1, 14, 1, pid)
        data = ctypes.create_string_buffer(4096)
        size = ctypes.c_size_t(len(data))
        query = ctypes.CDLL(None, use_errno=True).sysctl
        if query(mib, 4, data, ctypes.byref(size), None, 0):
            raise OSError(ctypes.get_errno(), "process-state inspection failed")
        if not size.value:
            return False
        return data[4 * ctypes.sizeof(ctypes.c_void_p) + 4] != b"\x05"
    try:
        os.kill(pid, 0)
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (ProcessLookupError, FileNotFoundError):
        return False


def timeout_after_stall(operator, operation):
    """Arm the real owner's deadline only after its transport has announced readiness."""
    hook = operator.tmp / "timeout-hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text("""import json, os, time
from pathlib import Path
import process_control as control
original = control.PinnedOutput.wait
def wait(output, seconds):
    argv = output.child.args
    target = json.loads(os.environ["TEST_TIMEOUT_OPERATION"])
    if any([Path(str(arg)).name, *argv[i+1:i+len(target)]] == target
           for i, arg in enumerate(argv)):
        end = time.monotonic() + 120
        while not json.loads(Path(os.environ["STUB_STATE"]).read_text()).get("stalled_pid"):
            if control.exited(output.child):
                return original(output, seconds)
            if time.monotonic() >= end:
                raise ValueError("Transport did not announce readiness")
            output.read_ready(0.05)
        # This is a real wait deadline; startup consumed none of its budget.
        seconds = 0.05
    return original(output, seconds)
control.PinnedOutput.wait = wait
if os.environ.get("EC2_CANCELLATION_GROUP_LOG"):
    import cancellation_registry_hook
""")
    operator.env["PYTHONPATH"] = str(hook) + os.pathsep + operator.env["PYTHONPATH"]
    operator.env["TEST_TIMEOUT_OPERATION"] = json.dumps(operation)


def wait_for_stall(operator, child):
    end = time.monotonic() + 120
    while not json.loads(operator.state_path.read_text()).get("stalled_pid"):
        if child.poll() is not None or time.monotonic() >= end:
            if child.poll() is None:
                child.kill()
            out, err = child.communicate(timeout=30)
            pytest.fail(
                f"Transport never became ready; exit={child.returncode}\nstdout:\n{out}\nstderr:\n{err}"
            )
        time.sleep(0.02)
    return json.loads(operator.state_path.read_text())


@pytest.mark.parametrize(
    "state",
    [
        dict(stall_contains="git checkout"),
        dict(stall_contains="dispatch-job.sh"),
        dict(stall_scp=True),
        dict(stall_rsync=True),
    ],
)
def test_real_guard_stalled_remote_reaches_single_cleanup(operator, state):
    # The nested transport timeout starts only once that leaf has been launched.
    # Healthy preparation retains the production budgets.
    prepare_real_chain(operator, **state)
    operation = (
        ["remote.sh", "dispatch"]
        if state.get("stall_contains") == "dispatch-job.sh"
        else ["remote.sh", "ssh"]
        if state.get("stall_contains")
        else ["remote.sh", "push"]
        if state.get("stall_scp")
        else ["remote.sh", "pull"]
    )
    timeout_after_stall(operator, operation)
    with running_real_chain(operator) as child:
        wait_for_stall(operator, child)
        out, err = child.communicate(timeout=60)
        assert child.returncode != 0, (out, err)
        state_after = json.loads(operator.state_path.read_text())
        assert not process_alive(state_after["stalled_pid"])
    data = json.loads(operator.ledger_path.read_text())
    assert len([t for t in data["transactions"] if t["command"] == "stop"]) == 1
    assert len([t for t in data["transactions"] if t["command"] == "reconcile"]) == 1
    assert data["sessions"][0]["status"] == "closed"
    if state.get("stall_contains") == "dispatch-job.sh":
        assert len([c for c in operator.calls("ssh") if "dispatch-job.sh" in c[-1]]) == 1
        assert (
            "UNKNOWN dispatch" in (operator.experiment / "ops/outer-job.real-chain.txt").read_text()
        )


def test_real_guard_poststart_host_failure_has_one_cleanup_owner(operator):
    operator.known.write_text("192.0.2.99 ssh-ed25519 AAAA\n")
    result = real_chain(operator)
    assert result.returncode != 0
    data = json.loads(operator.ledger_path.read_text())
    assert len([t for t in data["transactions"] if t["command"] == "stop"]) == 1
    assert len([t for t in data["transactions"] if t["command"] == "reconcile"]) == 1


def test_real_guard_changed_operator_ip_preserves_old_ingress(operator):
    use_real_guard(operator)
    operator.state(operator_ip="198.51.100.25")
    result = operator.run("session.sh", "start", "ip", "60", "0")
    assert result.returncode == 0, result.stderr
    state = json.loads(operator.state_path.read_text())
    assert state["cidrs"] == ["192.0.2.1/32", "198.51.100.25/32"]
    assert "198.51.100.25/32" in (operator.experiment / "ops/sg-changes.log").read_text()
    operator.state(operator_ip="198.51.100.26", ssh_failures=1, wait_codes=[0])
    result = operator.run("remote.sh", "wait", "/saved/job", "0", int(time.time()) + 60)
    assert result.returncode == 0, result.stderr
    state = json.loads(operator.state_path.read_text())
    assert state["cidrs"] == ["192.0.2.1/32", "198.51.100.25/32", "198.51.100.26/32"]
    assert not any("dispatch-job.sh" in c[-1] for c in operator.calls("ssh"))


def test_real_guard_latest_deadline_shortens_session(operator):
    use_real_guard(operator)
    operator.env["LATEST_DEADLINE"] = (datetime.now(timezone.utc) + timedelta(minutes=10)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    result = operator.run("session.sh", "start", "short", "60", "0")
    assert result.returncode == 0, result.stderr
    data = json.loads(operator.ledger_path.read_text())
    deadline = datetime.fromisoformat(data["sessions"][0]["deadline"].replace("Z", "+00:00"))
    latest = datetime.fromisoformat(operator.env["LATEST_DEADLINE"].replace("Z", "+00:00"))
    assert deadline <= latest


def test_rsync_real_parser_preserves_key_and_known_hosts_spaces(operator):
    parser = shutil.which("rsync")
    if not parser:
        pytest.skip("rsync parser not installed")
    # Run the installed parser; fake ssh records its actual argv and never connects.
    transport = subprocess.run(
        [
            sys.executable,
            str(OPS / "ops_control.py"),
            "rsync-transport",
            "ssh",
            "-i",
            str(operator.key),
            "-o",
            "UserKnownHostsFile=" + str(operator.known),
        ],
        env=operator.env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    subprocess.run(
        [parser, "-an", "-e", transport, "host:/unused/", str(operator.tmp / "out")],
        env=operator.env,
        capture_output=True,
        timeout=5,
    )
    calls = operator.calls("ssh")
    assert len(calls) == 1
    argv = calls[0]
    assert argv[argv.index("-i") + 1] == str(operator.key)
    assert argv[argv.index("-o") + 1] == "UserKnownHostsFile=" + str(operator.known)


def test_queue_lock_excludes_other_stages(tmp_path):
    import fcntl

    root = tmp_path / "ops/wins"
    root.mkdir(parents=True)
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run_wins(tmp_path, [dict(name="item", lane="light", cmd="true")])
        assert result.returncode == 2
        assert "Another queue is already running" in result.stderr
        assert "WINS_START" not in result.stdout


@pytest.mark.parametrize("unresolved_stop", [False, True])
def test_real_guard_interrupt_during_copy_advances_to_cleanup(operator, unresolved_stop):
    prepare_real_chain(operator, stall_rsync=True)
    with running_real_chain(operator) as child:
        # Wait for the fake transfer to enter its blocked operation, after its
        # durable state write. A call-log entry precedes that write and is not
        # sufficient readiness for killing this process.
        wait_for_stall(operator, child)
        if unresolved_stop:
            operator.state(instance_state="stopped", stop_reason="")
        child.send_signal(signal.SIGINT)
        _out, err = child.communicate(timeout=60)
    assert child.returncode == (2 if unresolved_stop else 130), err
    assert "Copy-back interrupted" in err
    if unresolved_stop:
        assert "Earlier failure exit=130 superseded by stop exit=2" in err
    else:
        assert "superseded by stop" not in err
    data = json.loads(operator.ledger_path.read_text())
    assert len([t for t in data["transactions"] if t["command"] == "stop"]) == 1
    assert len([t for t in data["transactions"] if t["command"] == "reconcile"]) == 1
    assert data["sessions"][0]["status"] == "closed", err


@pytest.mark.parametrize("copying", [False, True], ids=["checkout", "copy"])
@pytest.mark.parametrize("cancel", ["timeout", "interrupt"])
def test_real_guard_cancel_reaps_suspended_supervisor_and_leaf(operator, copying, cancel):
    # Giving the nested supervisor no opportunity to forward TERM pins the
    # orphaning interleaving, independently of scheduler speed.
    state = dict(stall_rsync=True) if copying else dict(stall_contains="git checkout")
    prepare_real_chain(operator, suspend_supervisor=True, **state)
    if cancel == "timeout":
        timeout_after_stall(operator, ["remote.sh", "pull" if copying else "ssh"])
    with running_real_chain(operator) as child:
        stalled = wait_for_stall(operator, child)
        if cancel == "interrupt":
            child.send_signal(signal.SIGINT)
        # communicate also requires EOF on both inherited pipes.
        out, err = child.communicate(timeout=60)
        for field in ("stalled_pid", "stalled_ppid"):
            assert not process_alive(stalled[field]), field
        assert child.returncode == (130 if copying and cancel == "interrupt" else 2), (out, err)
        assert json.loads(operator.state_path.read_text())["leaf_alive_at_stop"] is False
    data = json.loads(operator.ledger_path.read_text())
    assert len([t for t in data["transactions"] if t["command"] == "stop"]) == 1
    assert len([t for t in data["transactions"] if t["command"] == "reconcile"]) == 1
    assert data["sessions"][0]["status"] == "closed"


def test_real_guard_stalled_stop_fallback_is_bounded_and_unresolved(operator):
    use_real_guard(operator)
    assert operator.run("session.sh", "start", "shutdown", "60", "0").returncode == 0
    operator.state(
        instance_state="stopped",
        stall_contains="sudo /usr/local/sbin/arm-deadline.sh",
        stop_reason="",
    )
    operator.env.update(SSH_TIMEOUT_SECONDS="0.2", STOP_TIMEOUT_SECONDS="6")
    before = time.monotonic()
    result = operator.run("session.sh", "stop")
    assert time.monotonic() - before < 8
    assert result.returncode == 2
    assert "settle --txn" in result.stderr
    evidence = json.loads((operator.experiment / "ops/stop.shutdown.json").read_text())
    assert evidence["shutdown_ssh_exit"] == 124
    data = json.loads(operator.ledger_path.read_text())
    assert len([t for t in data["transactions"] if t["command"] == "stop"]) == 1
    assert len([t for t in data["transactions"] if t["command"] == "reconcile"]) == 1


def test_real_guard_capacity_attempt_limit(operator):
    use_real_guard(operator)
    operator.state(capacity_failures=3)
    result = operator.run("session.sh", "start", "bounded", "60", "0")
    assert result.returncode == 2
    assert "Capacity attempts exhausted" in result.stderr
    assert not operator.calls("sleep")
    data = json.loads(operator.ledger_path.read_text())
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["stop_time_estimated"] is False
