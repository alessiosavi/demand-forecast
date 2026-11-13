"""Real operator CLI against a fake gcloud; never uses an account or cloud API."""

import fcntl
import json
import os
import runpy
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / "scripts/gcp/gcp_guard.py"

FAKE = r"""
import fcntl, json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
assert '--format=json' in args, args
with Path(os.environ['CLOUD_CALLS']).open('a') as f:
    f.write(json.dumps(args) + '\n')
p = Path(os.environ['CLOUD_STATE'])
with p.open() as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    state = json.loads(p.read_text())
def save(): p.write_text(json.dumps(state))
def end(value=None, error=None):
    save()
    if error:
        print(error, file=sys.stderr)
        sys.exit(1)
    print(json.dumps(value))
    sys.exit(0)
def opt(name):
    return next((a.split('=', 1)[1] for a in args if a.startswith(name+'=')), None)
if args[:3] == ['storage', 'buckets', 'describe']:
    end(dict(location=state.get('region', 'EUROPE-WEST4'), location_type=state.get('location_type', 'region')))
if args[:3] == ['storage', 'objects', 'describe']:
    marker = args[3].rsplit('/', 1)[1]
    if state.get('marker_error'): end(error='403 Permission denied')
    if marker in state.get('markers', []): end(dict(name=marker, update_time='2026-11-13T00:00:00Z'))
    end(error='HTTPError 404: No such object')
if args[:2] == ['storage', 'cat']:
    marker = args[2].rsplit('/', 1)[1]
    if state.get('marker_read_error'): end(error='403 Permission denied')
    if marker not in state.get('markers', []): end(error='HTTPError 404: No such object')
    save()
    sys.stdout.write(state.get('marker_bodies', {}).get(marker, ''))
    sys.exit(0)
if args[:3] == ['compute', 'instances', 'create']:
    ledger = json.loads(Path(os.environ['LEDGER']).read_text())
    session = ledger['sessions'][-1]
    assert session['status'] == 'reserved' and session['settled_cost_usd'] is None
    assert Decimal(session['reservation_usd']) > 0
    if state.get('create_error'):
        state['create_failed'] = dict(instance=args[3], disk=session['boot_disk'])
        end(error=state['create_error'])
    if 'create_response' in state: end(state['create_response'])
    if '--async' in args:
        operation = dict(name='insert-'+args[3], operationType='insert',
                         zone='https://www.googleapis.com/compute/v1/projects/test-project/zones/'+opt('--zone'),
                         targetLink='https://www.googleapis.com/compute/v1/projects/test-project/zones/'+opt('--zone')+'/instances/'+args[3],
                         targetId=str(len(ledger['sessions'])), status='PENDING')
        state.setdefault('operations', []).append(operation)
        if state.get('async_pending'): end([operation])
    labels = dict(item.split('=', 1) for item in opt('--labels').split(','))
    assert opt('--boot-disk-name') is None, 'unsupported GA flag'
    disk = dict(item.split('=', 1) for item in opt('--create-disk').split(','))
    assert disk['boot'] == 'yes' and disk['auto-delete'] == 'yes'
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    vm = dict(name=args[3], id=str(len(ledger['sessions'])), zone=opt('--zone'), labels=labels,
              status='RUNNING', creationTimestamp=(now-timedelta(seconds=state.get('creation_age', 0))).isoformat(), lastStartTimestamp=(now-timedelta(seconds=state.get('start_age', 0))).isoformat(),
              scheduling=dict(instanceTerminationAction='DELETE', maxRunDuration=dict(seconds=opt('--max-run-duration')[:-1])),
              disks=[dict(type='PERSISTENT', source='projects/test/zones/'+opt('--zone')+'/disks/'+disk['name'], autoDelete=True)])
    if state.get('scratch_disk'):
        vm['disks'].append(dict(type='SCRATCH', interface='NVME', autoDelete=True))
    state.setdefault('instances', []).append(vm)
    state.setdefault('disks', []).append(dict(name=disk['name'], zone=opt('--zone'), labels={}))
    response = dict(vm)
    if state.get('omit_create_start'): response.pop('lastStartTimestamp')
    if state.get('omit_all_start'): vm.pop('lastStartTimestamp'); response.pop('lastStartTimestamp', None)
    if '--async' in args:
        operation['status'] = 'DONE'
        end([operation])
    end([response])
if args[:3] in (['compute', 'instances', 'list'], ['compute', 'disks', 'list']):
    kind = args[1]
    if state.get('create_failed'):
        index = state.get('failure_inventory_index', 0)
        reads = state.get('failure_inventory', [{}, {}])
        read = reads[min(index, len(reads)-1)]
        if kind == 'disks': state['failure_inventory_index'] = index + 1
        if read.get(kind+'_error'): end(error=read[kind+'_error'])
        if read.get(kind+'_warning'): print(read[kind+'_warning'], file=sys.stderr)
        items = read.get(kind, [])
        if isinstance(items, list):
            items = [dict(item, name=state['create_failed'][item['name']])
                     if item.get('name') in ('instance', 'disk') else item for item in items]
        end(items)
    end(state.get(kind, []))
if args[:3] == ['compute', 'instances', 'describe']:
    time.sleep(state.get('describe_delay', 0))
    matches = [v for v in state.get('instances', []) if v['name'] == args[3]
               and v['zone'].rsplit('/', 1)[-1] == opt('--zone')]
    if not matches: end(error='The resource was not found')
    if state.get('describe_updates'):
        matches[0].update(state['describe_updates'].pop(0))
    end(matches[0])
if args[:3] == ['compute', 'instances', 'stop']:
    from datetime import datetime, timezone
    for v in state['instances']:
        if v['name'] == args[3] and v['zone'].rsplit('/', 1)[-1] == opt('--zone'):
            if any(d['type'] == 'SCRATCH' for d in v['disks']) and opt('--discard-local-ssd') != 'true':
                end(error='Local SSD data must be explicitly discarded')
            v.update(status='TERMINATED', lastStopTimestamp=datetime.now(timezone.utc).isoformat())
    end({})
if args[:3] == ['compute', 'instances', 'delete']:
    if state.get('require_runtime_evidence'):
        ledger = json.loads(Path(os.environ['LEDGER']).read_text())
        assert ledger['sessions'][0]['runtime_evidence']['stop'] == state['require_runtime_evidence']
    if state.get('delete_error'): end(error='HTTP 503 delete failed')
    if state.get('delete_noop'): end({})
    state['instances'] = [v for v in state.get('instances', []) if v['name'] != args[3]
                          or v['zone'].rsplit('/', 1)[-1] != opt('--zone')]
    if not state.get('orphan_disk'): state['disks'] = []
    end({})
if args[:3] == ['compute', 'disks', 'add-labels']:
    if state.get('disk_label_error'): end(error='label update failed')
    for d in state.get('disks', []):
        if d['name'] == args[3]: d['labels'] = dict(x.split('=', 1) for x in opt('--labels').split(','))
    end({})
if args[:3] == ['compute', 'disks', 'delete']:
    if not any(d['name'] == args[3] for d in state.get('disks', [])):
        end(error='HTTPError 404: disk not found')
    state['disks'] = [d for d in state.get('disks', []) if d['name'] != args[3]]
    end({})
if args[:3] == ['compute', 'operations', 'list']:
    assert opt('--zones'), args
    target = opt('--filter')
    if state.get('operations_error'): end(error='403 Permission denied')
    if state.get('operations_warning'): print(state['operations_warning'], file=sys.stderr)
    if 'operations_response' in state: end(state['operations_response'])
    assert target, args
    if target.startswith('targetId='):
        end([o for o in state.get('operations', []) if str(o.get('targetId')) == target.split('=', 1)[1]
             and o.get('zone', '').rsplit('/', 1)[-1] == opt('--zones')])
    assert target.startswith("targetLink~'/projects/") and target.endswith("$' AND operationType=insert"), args
    link = target.removeprefix("targetLink~'").removesuffix("$' AND operationType=insert")
    end([o for o in state.get('operations', []) if o.get('targetLink', '').endswith(link)
         and o.get('operationType') == 'insert' and o.get('zone', '').rsplit('/', 1)[-1] == opt('--zones')])
raise AssertionError(args)
"""
# Decimal is imported separately to keep fake executable's source self-contained.
FAKE = "from decimal import Decimal\n" + FAKE


@pytest.fixture
def cloud(tmp_path):
    executable = tmp_path / "gcloud"
    executable.write_text(f"#!{sys.executable}\n" + FAKE)
    executable.chmod(0o755)
    state = tmp_path / "cloud.json"
    state.write_text("{}")
    ledger = tmp_path / "ledger.json"
    calls = tmp_path / "calls.jsonl"
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        CLOUD_STATE=str(state),
        CLOUD_CALLS=str(calls),
        LEDGER=str(ledger),
    )

    class Cloud:
        def command(self, *args):
            return [sys.executable, str(GUARD), "--ledger", str(ledger), *map(str, args)]

        def run(self, *args, ok=True):
            result = subprocess.run(
                self.command(*args), env=env, capture_output=True, text=True, timeout=20
            )
            if ok:
                assert result.returncode == 0, result.stdout + result.stderr
            return result

        def setup(self, cap="100"):
            self.run(
                "init",
                "--project",
                "test-project",
                "--bucket",
                "test-bucket",
                "--cap",
                cap,
                "--enforce",
                cap,
                "--boot-disk-usd-per-gb-month",
                "0.1",
            )
            self.price()

        def price(self, model="SPOT", rate="4"):
            self.run(
                "price",
                "set",
                f"g2-standard-8|{model}|europe-west4",
                "--usd",
                rate,
                "--source",
                "catalogue fixture",
                "--fetched-on",
                "2026-11-13",
            )

        def launch(self, name="run-a", *extra, ok=True, model="SPOT"):
            return self.run(
                "launch",
                name,
                "--machine-type",
                "g2-standard-8",
                "--zone",
                "europe-west4-a",
                "--provisioning",
                model,
                "--hours",
                "1",
                "--boot-disk-size",
                "100",
                "--image-family",
                "fixture-image",
                "--image-project",
                "fixture-images",
                "--job-uri",
                "gs://test-bucket/jobs/job.sh",
                *extra,
                ok=ok,
            )

        def read(self):
            return json.loads(ledger.read_text())

        def change(self, **changes):
            data = json.loads(state.read_text())
            data.update(changes)
            state.write_text(json.dumps(data))

        def calls(self):
            return [json.loads(s) for s in calls.read_text().splitlines()] if calls.exists() else []

    c = Cloud()
    c.ledger, c.state, c.env = ledger, state, env
    return c


def test_init_price_and_money_validation(cloud):
    cloud.setup()
    original = cloud.ledger.read_bytes()
    assert cloud.run("init", ok=False).returncode != 0
    assert cloud.ledger.read_bytes() == original
    report = json.loads(cloud.run("report").stdout)
    assert Decimal(report["committed_usd"]) == 0
    assert report["currency"] == "USD"
    assert len(cloud.read()["config"]["token"]) >= 24
    for invalid in ("-1", "NaN", "Infinity"):
        assert cloud.run("hold", "set", "storage", invalid, ok=False).returncode != 0
    cloud.run("hold", "set", "storage", "99")
    assert cloud.launch(ok=False).returncode != 0
    assert not cloud.read()["sessions"]


def test_failed_create_reserved_first_cancels_zero_and_burns_label(cloud):
    cloud.setup()
    cloud.change(create_error="ZONE_RESOURCE_POOL_EXHAUSTED: stock-out")
    result = cloud.launch(ok=False)
    assert result.returncode != 0
    session = cloud.read()["sessions"][0]
    assert session["status"] == "cancelled"
    assert Decimal(session["settled_cost_usd"]) == 0
    assert "stock-out" in session["error"]
    assert Decimal(json.loads(cloud.run("report").stdout)["committed_usd"]) == 0
    before = len(cloud.calls())
    assert cloud.launch(ok=False).returncode != 0
    assert not any(c[:3] == ["compute", "instances", "create"] for c in cloud.calls()[before:])


def test_launch_disk_type_and_nic_are_priced_and_retained_by_resume(cloud):
    cloud.setup()
    cloud.run(
        "price",
        "set",
        "hyperdisk-balanced",
        "--usd",
        "0.2",
        "--source",
        "disk catalogue fixture",
        "--fetched-on",
        "2026-11-13",
    )
    cloud.run(
        "price",
        "set",
        "g4-standard-48|SPOT|europe-west4",
        "--usd",
        "4",
        "--source",
        "machine fixture",
        "--fetched-on",
        "2026-11-13",
    )
    cloud.launch(
        "run-a",
        "--machine-type",
        "g4-standard-48",
        "--boot-disk-type",
        "hyperdisk-balanced",
        "--nic-type",
        "GVNIC",
    )
    session = cloud.read()["sessions"][0]
    assert session["reservation_usd"] == "5.034247"
    assert session["launch"]["boot_disk_type"] == "hyperdisk-balanced"
    assert session["launch"]["nic_type"] == "GVNIC"
    cloud.change(instances=[], disks=[])
    before = cloud.ledger.read_bytes()
    dry = cloud.run("resume", "run-a", "run-b", "--dry-run")
    assert cloud.ledger.read_bytes() == before
    assert "type=hyperdisk-balanced" in dry.stdout
    assert "--network-interface=nic-type=GVNIC" in dry.stdout
    cloud.run("resume", "run-a", "run-b")
    resumed = cloud.read()["sessions"][1]
    assert resumed["reservation_usd"] == "5.034247"
    creates = [c for c in cloud.calls() if c[:3] == ["compute", "instances", "create"]]
    assert len(creates) == 2
    for call in creates:
        assert any(",type=hyperdisk-balanced," in a for a in call)
        assert "--network-interface=nic-type=GVNIC" in call
        assert "--max-run-duration=3600s" in call
        assert "--instance-termination-action=DELETE" in call


def test_unpriced_disk_refused_before_reserving(cloud):
    cloud.setup()
    before = len(cloud.calls())
    result = cloud.launch("run-a", "--boot-disk-type", "hyperdisk-balanced", ok=False)
    assert result.returncode == 2 and "no stored" in result.stderr
    assert not cloud.read()["sessions"]
    assert len(cloud.calls()) == before


def test_legacy_disk_price_default_launch_and_resume(cloud):
    cloud.setup()
    ledger = cloud.read()
    ledger.pop("disk_prices", None)
    ledger["config"].update(boot_disk_type="pd-ssd", boot_disk_usd_per_gb_month="0.15")
    cloud.ledger.write_text(json.dumps(ledger))
    before = cloud.ledger.read_bytes()
    cloud.launch("dry", "--dry-run")
    assert cloud.ledger.read_bytes() == before
    cloud.launch()
    assert cloud.read()["sessions"][0]["reservation_usd"] == "5.025685"
    ledger = cloud.read()
    assert ledger["disk_prices"]["pd-ssd"]["usd_per_gb_month"] == "0.15"
    assert "boot_disk_usd_per_gb_month" not in ledger["config"]
    ledger["sessions"][0]["launch"].pop("boot_disk_type", None)
    ledger["sessions"][0]["launch"].pop("nic_type", None)
    cloud.ledger.write_text(json.dumps(ledger))
    cloud.change(instances=[], disks=[])
    cloud.run("resume", "run-a", "run-b")
    creates = [c for c in cloud.calls() if c[:3] == ["compute", "instances", "create"]]
    assert len(creates) == 2
    for call in creates:
        assert any(",type=pd-ssd," in a for a in call)
        assert not any(a.startswith("--network-interface=") for a in call)


def test_disk_price_cannot_be_zero_or_nonfinite(cloud):
    cloud.setup()
    before = cloud.ledger.read_bytes()
    for rate in ("0", "-1", "NaN", "Infinity"):
        result = cloud.run(
            "price",
            "set",
            "hyperdisk-balanced",
            "--usd",
            rate,
            "--source",
            "disk fixture",
            "--fetched-on",
            "2026-11-13",
            ok=False,
        )
        assert result.returncode == 2
        assert cloud.ledger.read_bytes() == before


@pytest.mark.parametrize("model", ["SPOT", "STANDARD", "FLEX_START"])
def test_every_create_and_resume_has_server_limit_and_delete(cloud, model):
    cloud.setup()
    cloud.price(model)
    cloud.launch(model=model)
    session = cloud.read()["sessions"][0]
    assert Decimal(session["reservation_usd"]) >= Decimal("5.017")
    cloud.change(instances=[], disks=[], markers=["PREEMPTED"])
    cloud.run("resume", "run-a", "run-b")
    sessions = cloud.read()["sessions"]
    assert sessions[1]["state_uri"] == sessions[0]["state_uri"]
    creates = [c for c in cloud.calls() if c[:3] == ["compute", "instances", "create"]]
    assert len(creates) == 2
    for call in creates:
        assert "--max-run-duration=3600s" in call
        assert "--instance-termination-action=DELETE" in call
        assert "--maintenance-policy=TERMINATE" in call
        assert "--no-restart-on-failure" in call
        assert any("startup-script=" in a and "guest-run.sh" in a for a in call)
        assert any("shutdown-script=" in a for a in call)
        if model == "FLEX_START":
            assert "--request-valid-for-duration=0s" in call
    assert cloud.read()["sessions"][0]["settled_cost_usd"] is None


def test_concurrent_launches_cannot_both_pass_cap(cloud):
    cloud.setup(cap="6")
    base = cloud.command(
        "launch",
        "first",
        "--machine-type",
        "g2-standard-8",
        "--zone",
        "europe-west4-a",
        "--provisioning",
        "SPOT",
        "--hours",
        "1",
        "--boot-disk-size",
        "100",
        "--image-family",
        "fixture",
        "--image-project",
        "fixture",
        "--job-uri",
        "gs://test-bucket/job.sh",
    )
    announce_lock = """
import fcntl, os, runpy, sys
from pathlib import Path
original = fcntl.flock
def flock(stream, operation):
    Path(os.environ['LOCK_ATTEMPT']).touch()
    return original(stream, operation)
fcntl.flock = flock
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    processes, attempts = [], []
    try:
        # Hold the real ledger lock until both launches reach flock. The old
        # create_delay merely hoped the second process started within 0.2 s.
        with cloud.ledger.with_suffix(".json.lock").open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for label in ("first", "second"):
                attempt = cloud.ledger.with_name(label + ".lock-attempt")
                attempts.append(attempt)
                base[base.index("first")] = label
                processes.append(
                    subprocess.Popen(
                        [sys.executable, "-c", announce_lock, *base[1:]],
                        env=dict(cloud.env, LOCK_ATTEMPT=str(attempt)),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                )
            deadline = time.monotonic() + 60
            while not all(p.exists() for p in attempts) and time.monotonic() < deadline:
                assert all(p.poll() is None for p in processes), "launch exited before locking"
                time.sleep(0.01)
            assert all(p.exists() for p in attempts), "both launches must contend for the lock"
        for process in processes:
            process.communicate(timeout=60)
        assert sorted(p.returncode for p in processes) == [0, 2]
        assert len(cloud.read()["sessions"]) == 1
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=60)


def test_close_already_deleted_keeps_reservation_until_evidence(cloud):
    cloud.setup()
    cloud.launch()
    cloud.change(instances=[], disks=[])
    result = cloud.run("close", "run-a", ok=False)
    assert result.returncode != 0
    session = cloud.read()["sessions"][0]
    assert session["settled_cost_usd"] is None
    assert session["status"] == "needs_settlement"
    assert cloud.run("close", "run-a", "--settled-usd", "1", ok=False).returncode != 0
    cloud.run("close", "run-a", "--settled-usd", "1", "--evidence", "billing export fixture row 12")
    assert Decimal(cloud.read()["sessions"][0]["settled_cost_usd"]) == 1


def test_close_uses_real_timestamps_and_checks_disk_removal(cloud, monkeypatch):
    cloud.setup()
    cloud.launch()
    module = runpy.run_path(str(GUARD))
    frozen = datetime(2026, 11, 13, 1, tzinfo=timezone.utc)
    monkeypatch.setitem(module["Guard"].close.__globals__, "now", lambda: frozen)
    for key, value in cloud.env.items():
        monkeypatch.setenv(key, value)
    ledger = cloud.read()
    session = ledger["sessions"][0]
    session.update(last_start="2026-11-13T00:30:00Z", instance_created_at="2026-11-13T00:00:00Z")
    cloud.ledger.write_text(json.dumps(ledger))
    state = json.loads(cloud.state.read_text())
    state["instances"][0].update(
        status="TERMINATED",
        lastStopTimestamp=frozen.isoformat(),
        lastStartTimestamp=session["last_start"],
        creationTimestamp=session["instance_created_at"],
    )
    cloud.change(**state, orphan_disk=True)
    args = module["parser"]().parse_args(["--ledger", str(cloud.ledger), "close", "run-a"])
    with pytest.raises(module["GuardError"], match="disk remains"):
        module["Guard"](cloud.ledger).close(args)
    assert cloud.read()["sessions"][0]["settled_cost_usd"] is None
    cloud.change(disks=[])
    module["Guard"](cloud.ledger).close(args)
    assert cloud.read()["sessions"][0]["settled_cost_usd"] == "2.013699"


def test_shell_metacharacters_never_execute(cloud, tmp_path):
    cloud.setup()
    path = tmp_path / "pwned"
    for label in ("run;touch", "$(id)", "a,b"):
        assert cloud.launch(label, ok=False).returncode != 0
    assert (
        cloud.launch("safe", "--job-uri", f"gs://test-bucket/$(touch {path})", ok=False).returncode
        != 0
    )
    assert not path.exists()
    assert not cloud.read()["sessions"]


def test_sweep_unknown_overdue_missing_and_delete(cloud):
    cloud.setup()
    cloud.launch()
    token = cloud.read()["config"]["token"]
    stray = dict(
        name="forgotten", zone="europe-west4-a", status="RUNNING", labels={"guard-ledger": token}
    )
    cloud.change(instances=[stray], disks=[])
    result = cloud.run("sweep", ok=False)
    assert result.returncode == 1
    assert "forgotten" in result.stdout and "missing_instance" in result.stdout
    result = cloud.run("sweep", "--delete", ok=False)
    assert result.returncode == 1
    assert json.loads(cloud.state.read_text())["instances"] == []


def test_done_or_active_state_refuses_resume_and_status_has_markers(cloud):
    cloud.setup()
    cloud.launch()
    assert cloud.run("resume", "run-a", "run-b", ok=False).returncode != 0
    cloud.change(instances=[], disks=[], markers=["STARTED", "DONE", "HEARTBEAT"])
    assert cloud.run("resume", "run-a", "run-b", ok=False).returncode != 0
    status = json.loads(cloud.run("status", "run-a").stdout)
    assert status[0]["markers"]["DONE"]["name"] == "DONE"
    assert status[0]["markers"]["STARTED"]["name"] == "STARTED"
    assert status[0]["markers"]["HEARTBEAT"]["update_time"]


def test_different_job_launches_on_done_prefix_but_same_job_and_resume_are_refused(cloud):
    cloud.setup()
    cloud.launch()
    prefix = cloud.read()["sessions"][0]["state_uri"]
    old_job = "gs://test-bucket/jobs/job.sh"
    cloud.change(markers=["DONE"], marker_bodies={"DONE": f"job_uri={old_job}\n"})
    # The prefix's existing instance still blocks even a different job.
    active = cloud.launch(
        "run-b", "--state-uri", prefix, "--job-uri", "gs://test-bucket/jobs/leg-2.sh", ok=False
    )
    assert active.returncode == 2 and "already has an instance" in active.stderr
    cloud.change(instances=[], disks=[])
    before = cloud.ledger.read_bytes()
    same = cloud.launch("same", "--state-uri", prefix, ok=False)
    resumed = cloud.run("resume", "run-a", "resumed", ok=False)
    assert same.returncode == resumed.returncode == 2
    assert "DONE" in same.stderr and "DONE" in resumed.stderr
    assert cloud.ledger.read_bytes() == before
    cloud.launch("run-b", "--state-uri", prefix, "--job-uri", "gs://test-bucket/jobs/leg-2.sh")
    second = cloud.read()["sessions"][1]
    assert second["state_uri"] == prefix
    assert second["launch"]["job_uri"] == "gs://test-bucket/jobs/leg-2.sh"
    creates = [c for c in cloud.calls() if c[:3] == ["compute", "instances", "create"]]
    assert len(creates) == 2


@pytest.mark.parametrize("body", ["", "label=old\ntime=2026-11-13T00:00:00Z\n", "job_uri=\n"])
def test_legacy_done_refuses_launch_and_resume_with_recovery_message(cloud, body):
    cloud.setup()
    cloud.launch()
    prefix = cloud.read()["sessions"][0]["state_uri"]
    cloud.change(instances=[], disks=[], markers=["DONE"], marker_bodies={"DONE": body})
    before = cloud.ledger.read_bytes()
    for result in (
        cloud.launch(
            "run-b", "--state-uri", prefix, "--job-uri", "gs://test-bucket/other.sh", ok=False
        ),
        cloud.run("resume", "run-a", "run-c", ok=False),
    ):
        assert result.returncode == 2
        assert "no job URI" in result.stderr
        assert "archive" in result.stderr and "new state prefix" in result.stderr
    assert cloud.ledger.read_bytes() == before


def test_preempted_same_job_can_resume_with_original_job_uri(cloud):
    cloud.setup()
    cloud.launch()
    original = cloud.read()["sessions"][0]
    cloud.change(
        instances=[],
        disks=[],
        markers=["PREEMPTED"],
        marker_bodies={"PREEMPTED": "job_uri=gs://test-bucket/jobs/job.sh\n"},
    )
    cloud.run("resume", "run-a", "run-b")
    resumed = cloud.read()["sessions"][1]
    assert resumed["launch"]["job_uri"] == original["launch"]["job_uri"]
    assert resumed["state_uri"] == original["state_uri"]
    create = next(c for c in reversed(cloud.calls()) if c[:3] == ["compute", "instances", "create"])
    assert any("guard-job-uri=gs://test-bucket/jobs/job.sh" in a for a in create)


def test_status_shows_each_markers_job_uri(cloud):
    cloud.setup()
    cloud.launch()
    keys = ["STARTED", "HEARTBEAT", "DONE", "FAILED", "TIMEOUT", "PREEMPTED"]
    cloud.change(
        markers=keys,
        marker_bodies={key: f"job_uri=gs://test-bucket/{key}.sh\n" for key in keys},
    )
    markers = json.loads(cloud.run("status", "run-a").stdout)[0]["markers"]
    for key in keys:
        assert markers[key]["job_uri"] == f"gs://test-bucket/{key}.sh"
        assert markers[key]["name"] == key


def test_unreadable_done_blocks_launch_without_reservation(cloud):
    cloud.setup()
    cloud.change(markers=["DONE"], marker_read_error=True)
    assert cloud.launch(ok=False).returncode == 2
    assert not cloud.read()["sessions"]


@pytest.mark.parametrize("operation", ["launch", "resume", "close", "sweep"])
def test_dry_run_prints_exact_commands_without_mutations(cloud, operation):
    cloud.setup()
    cloud.launch()
    cloud.change(instances=[], disks=[])
    before = cloud.ledger.read_bytes()
    start = len(cloud.calls())
    if operation == "launch":
        result = cloud.launch("dry", "--dry-run")
    elif operation == "resume":
        result = cloud.run("resume", "run-a", "dry", "--dry-run")
    elif operation == "close":
        result = cloud.run("close", "run-a", "--dry-run")
    else:
        result = cloud.run("sweep", "--delete", "--dry-run", ok=False)
    assert cloud.ledger.read_bytes() == before
    for line in result.stdout.splitlines():
        data = json.loads(line)
        if isinstance(data, list):
            assert data[0] == "gcloud" and "--format=json" in data
    assert not any(
        c[:3]
        in (
            ["compute", "instances", "create"],
            ["compute", "instances", "delete"],
            ["compute", "disks", "delete"],
        )
        for c in cloud.calls()[start:]
    )


def test_missing_price_region_and_permission_fail_closed(cloud):
    cloud.setup()
    assert cloud.launch(model="STANDARD", ok=False).returncode != 0
    cloud.change(region="US-CENTRAL1")
    assert cloud.launch(ok=False).returncode != 0
    cloud.change(region="EUROPE-WEST4", location_type="multi-region")
    assert cloud.launch(ok=False).returncode != 0
    assert not cloud.read()["sessions"]


def test_cap_is_not_rounded_up(cloud):
    result = cloud.run(
        "init",
        "--project",
        "test-project",
        "--bucket",
        "test-bucket",
        "--cap",
        "1.0000001",
        "--enforce",
        "1.0000001",
        "--boot-disk-usd-per-gb-month",
        "0.1",
        ok=False,
    )
    assert result.returncode == 0, result.stderr
    assert Decimal(cloud.read()["config"]["enforce_usd"]) == Decimal("1.0000001")


def test_sweep_names_overdue_vm_and_unlabelled_boot_disk(cloud):
    cloud.setup()
    cloud.launch()
    ledger = cloud.read()
    ledger["sessions"][0]["reserved_end"] = "2020-01-01T00:00:00Z"
    cloud.ledger.write_text(json.dumps(ledger))
    result = cloud.run("sweep", ok=False)
    assert result.returncode == 1 and "past_reserved_end" in result.stdout
    state = json.loads(cloud.state.read_text())
    state["disks"][0]["labels"] = {}
    cloud.change(instances=[], disks=state["disks"])
    result = cloud.run("sweep", ok=False)
    assert result.returncode == 1
    assert ledger["sessions"][0]["boot_disk"] in result.stdout
    assert "orphan_disk" in result.stdout


def test_sweep_names_stopped_vm_that_still_has_billable_disk(cloud):
    cloud.setup()
    cloud.launch()
    state = json.loads(cloud.state.read_text())
    state["instances"][0]["status"] = "TERMINATED"
    cloud.change(**state)
    result = cloud.run("sweep", ok=False)
    assert result.returncode == 1
    assert "stopped_instance" in result.stdout


@pytest.mark.parametrize(
    "error",
    [
        "pd-balanced disk type cannot be used by g4-standard-48 machine type",
        "Connection reset before response",
        "Polling operation failed: quota for read requests exceeded",
    ],
)
def test_returned_create_error_cancels_after_two_absence_reads(cloud, error, monkeypatch):
    cloud.setup()
    cloud.change(create_error=error)
    from datetime import timedelta
    from types import SimpleNamespace

    module = runpy.run_path(str(GUARD))
    clock = [datetime(2026, 11, 13, 1, tzinfo=timezone.utc)]

    def advance(seconds):
        clock[0] += timedelta(seconds=seconds)

    namespace = module["Guard"].launch.__globals__
    monkeypatch.setitem(namespace, "now", lambda: clock[0])
    monkeypatch.setitem(namespace, "time", SimpleNamespace(sleep=advance))
    for key, value in cloud.env.items():
        monkeypatch.setenv(key, value)
    args = module["parser"]().parse_args(
        [
            "--ledger",
            str(cloud.ledger),
            "launch",
            "run-a",
            "--machine-type",
            "g2-standard-8",
            "--zone",
            "europe-west4-a",
            "--provisioning",
            "SPOT",
            "--hours",
            "1",
            "--boot-disk-size",
            "100",
            "--image-family",
            "fixture",
            "--image-project",
            "fixture",
            "--job-uri",
            "gs://test-bucket/jobs/job.sh",
        ]
    )
    with pytest.raises(module["GuardError"], match="create failed"):
        module["Guard"](cloud.ledger).launch(args)
    session = cloud.read()["sessions"][0]
    assert session["status"] == "cancelled"
    assert Decimal(session["settled_cost_usd"]) == 0
    assert error in session["error"]
    reads = session["create_failure_inventory"]
    assert len(reads) == 2
    assert all(read["instances"] == read["disks"] == [] for read in reads)
    times = [datetime.fromisoformat(read["observed_at"].replace("Z", "+00:00")) for read in reads]
    assert (times[1] - times[0]).total_seconds() == 3
    assert Decimal(json.loads(cloud.run("report").stdout)["committed_usd"]) == 0
    assert cloud.launch(ok=False).returncode == 2


@pytest.mark.parametrize("kind,resource", [("disks", "disk"), ("instances", "instance")])
@pytest.mark.parametrize("found_read", [0, 1])
def test_create_error_resource_in_either_read_retains_reservation(
    cloud, kind, resource, found_read
):
    cloud.setup()
    reads = [{}, {}]
    # No labels: an exact instance/disk name alone prevents cancellation.
    reads[found_read] = {kind: [{"name": resource, "zone": "europe-west4-a"}]}
    cloud.change(
        create_error="INVALID_ARGUMENT: incompatible configuration", failure_inventory=reads
    )
    assert cloud.launch(ok=False).returncode == 2
    session = cloud.read()["sessions"][0]
    assert session["status"] == "reserved" and session["settled_cost_usd"] is None
    evidence = session["create_failure_inventory"]
    assert len(evidence) == 2
    assert evidence[found_read][kind][0]["name"] == (
        session["boot_disk"] if kind == "disks" else "run-a"
    )
    assert Decimal(json.loads(cloud.run("report").stdout)["committed_usd"]) == Decimal(
        session["reservation_usd"]
    )


@pytest.mark.parametrize("fault", [{"disks_error": "403 Permission denied"}, {"disks": None}])
def test_create_error_inventory_failure_retains_reservation(cloud, fault):
    cloud.setup()
    cloud.change(create_error="rejected configuration", failure_inventory=[{}, fault])
    assert cloud.launch(ok=False).returncode == 2
    session = cloud.read()["sessions"][0]
    assert session["status"] == "reserved" and session["settled_cost_usd"] is None
    assert "rejected configuration" in session["error"]
    assert len(session["create_failure_inventory"]) == 2
    assert session["create_failure_inventory"][1]["error"]


@pytest.mark.parametrize("kind", ["instances", "disks"])
@pytest.mark.parametrize("warned_read", [0, 1])
def test_create_error_partial_inventory_warning_retains_reservation(cloud, kind, warned_read):
    cloud.setup()
    reads = [{}, {}]
    warning = "WARNING: Some requests did not succeed: zone europe-west4-a is unreachable"
    reads[warned_read][kind + "_warning"] = warning
    cloud.change(create_error="rejected configuration", failure_inventory=reads)
    assert cloud.launch(ok=False).returncode == 2
    session = cloud.read()["sessions"][0]
    assert session["status"] == "reserved" and session["settled_cost_usd"] is None
    assert len(session["create_failure_inventory"]) == 2
    assert warning in session["create_failure_inventory"][warned_read]["error"]


def test_create_failure_inventory_omits_account_and_metadata_fields(cloud):
    cloud.setup()
    resource = {
        "name": "instance",
        "zone": "europe-west4-a",
        "id": "123",
        "status": "PROVISIONING",
        "serviceAccounts": [{"email": "fixture@example.invalid"}],
        "metadata": {"items": [{"key": "ssh-keys", "value": "private fixture metadata"}]},
    }
    cloud.change(
        create_error="rejected configuration", failure_inventory=[{}, {"instances": [resource]}]
    )
    assert cloud.launch(ok=False).returncode == 2
    session = cloud.read()["sessions"][0]
    assert session["status"] == "reserved" and session["settled_cost_usd"] is None
    inventory = session["create_failure_inventory"][1]["instances"][0]
    assert inventory["name"] == "run-a" and inventory["id"] == "123"
    assert inventory["zone"] == "europe-west4-a" and inventory["status"] == "PROVISIONING"
    assert "serviceAccounts" not in inventory and "metadata" not in inventory


@pytest.mark.parametrize("response", [None, {}, [], [{}, {}]])
def test_unexpected_create_response_retains_reservation(cloud, response):
    cloud.setup()
    cloud.change(create_response=response)
    assert cloud.launch(ok=False).returncode == 2
    session = cloud.read()["sessions"][0]
    assert session["status"] == "reserved" and session["settled_cost_usd"] is None
    assert "create_failure_inventory" not in session


def test_permission_error_is_not_done_absence(cloud):
    cloud.setup()
    cloud.change(marker_error=True)
    assert cloud.launch(ok=False).returncode != 0
    assert not cloud.read()["sessions"]


def test_post_create_disk_label_error_preserves_reservation_and_sweep_finds_disk(cloud):
    cloud.setup()
    cloud.change(disk_label_error=True)
    assert cloud.launch(ok=False).returncode != 0
    session = cloud.read()["sessions"][0]
    assert session["status"] == "open" and session["settled_cost_usd"] is None
    cloud.change(instances=[])
    result = cloud.run("sweep", ok=False)
    assert result.returncode == 1 and session["boot_disk"] in result.stdout


def test_close_running_vm_stops_reads_timestamps_and_deletes(cloud):
    cloud.setup()
    cloud.launch()
    cloud.run("close", "run-a")
    session = cloud.read()["sessions"][0]
    assert session["status"] == "closed"
    assert Decimal(session["settled_cost_usd"]) >= Decimal("0.066666")
    calls = [c[:3] for c in cloud.calls()]
    assert calls.index(["compute", "instances", "stop"]) < calls.index(
        ["compute", "instances", "delete"]
    )


def test_dry_close_and_sweep_never_delete_existing_vm(cloud):
    cloud.setup()
    cloud.launch()
    before = cloud.ledger.read_bytes()
    count = len(cloud.calls())
    close = cloud.run("close", "run-a", "--dry-run")
    assert '"stop"' in close.stdout and '"delete"' in close.stdout
    ledger = cloud.read()
    ledger["sessions"][0]["reserved_end"] = "2020-01-01T00:00:00Z"
    cloud.ledger.write_text(json.dumps(ledger))
    cloud.run("sweep", "--delete", "--dry-run", ok=False)
    assert len(json.loads(cloud.state.read_text())["instances"]) == 1
    assert not any(c[2] in {"stop", "delete"} for c in cloud.calls()[count:] if c[0] == "compute")
    assert json.loads(before)["sessions"][0]["status"] == "open"


def test_missing_first_start_requires_manual_settlement(cloud):
    cloud.setup()
    cloud.launch()
    ledger = cloud.read()
    ledger["sessions"][0].pop("last_start")  # Crash before saving the create response.
    cloud.ledger.write_text(json.dumps(ledger))
    result = cloud.run("close", "run-a", ok=False)
    assert result.returncode != 0
    assert cloud.read()["sessions"][0]["settled_cost_usd"] is None
    assert cloud.read()["sessions"][0]["status"] == "needs_settlement"


def test_sweep_copied_labels_do_not_hide_unknown_instance(cloud):
    cloud.setup()
    cloud.launch()
    state = json.loads(cloud.state.read_text())
    clone = dict(state["instances"][0], name="forgotten-clone", id="999")
    cloud.change(instances=[*state["instances"], clone])
    result = cloud.run("sweep", ok=False)
    assert result.returncode == 1
    assert "forgotten-clone" in result.stdout
    assert "unknown_resource" in result.stdout


@pytest.mark.parametrize(
    "machine,scratch", [("g4-standard-48", True), ("g2-standard-8", True), ("a3-highgpu-1g", False)]
)
def test_close_discards_ssd_only_when_describe_reports_scratch(cloud, machine, scratch):
    cloud.setup()
    cloud.run(
        "price",
        "set",
        f"{machine}|SPOT|europe-west4",
        "--usd",
        "4",
        "--source",
        "fixture",
        "--fetched-on",
        "2026-11-13",
    )
    cloud.change(scratch_disk=scratch)
    cloud.launch("run-a", "--machine-type", machine)
    cloud.run("close", "run-a")
    stop = next(c for c in cloud.calls() if c[:3] == ["compute", "instances", "stop"])
    flags = [a for a in stop if a.startswith("--discard-local-ssd")]
    assert flags == (["--discard-local-ssd=true"] if scratch else [])
    assert cloud.read()["sessions"][0]["status"] == "closed"


def deleted_operations(cloud, terminal="compute.instances.preempted", recorded=True):
    """A one-hour run with an 80-minute boot-disk lifetime, independent of laptop time."""
    cloud.setup()
    cloud.launch()
    ledger = cloud.read()
    session = ledger["sessions"][0]
    session["instance_created_at"] = "2026-11-12T00:00:00Z"
    session["last_start"] = "2026-11-12T00:10:00Z" if recorded else None
    cloud.ledger.write_text(json.dumps(ledger))
    base = dict(
        kind="compute#operation",
        targetId=session["instance_id"],
        targetLink="https://www.googleapis.com/compute/v1/projects/test-project/zones/europe-west4-a/instances/run-a",
        zone="https://www.googleapis.com/compute/v1/projects/test-project/zones/europe-west4-a",
        status="DONE",
    )
    operations = [
        dict(
            base,
            name="insert-op",
            operationType="insert",
            insertTime="2026-11-12T00:00:00Z",
            startTime="2026-11-12T00:00:01Z",
            endTime="2026-11-12T00:10:00Z",
        ),
        dict(
            base,
            name="delete-op",
            operationType="delete",
            startTime="2026-11-12T01:10:00Z" if terminal == "delete" else "2026-11-12T01:19:00Z",
            endTime="2026-11-12T01:20:00Z",
        ),
    ]
    if terminal != "delete":
        operations.append(
            dict(
                base,
                name="stop-op",
                operationType=terminal,
                startTime="2026-11-12T01:10:00Z",
                endTime="2026-11-12T01:10:30Z",
            )
        )
    cloud.change(
        instances=[],
        disks=[],
        operations=list(reversed(operations)),
        markers=["STARTED", "DONE", "PREEMPTED"],
    )
    return operations


@pytest.mark.parametrize("terminal", ["compute.instances.preempted", "stop", "delete"])
@pytest.mark.parametrize("recorded", [True, False])
def test_deleted_instance_settles_from_zone_operations(cloud, terminal, recorded):
    operations = deleted_operations(cloud, terminal, recorded)
    cloud.run("close", "run-a")
    session = cloud.read()["sessions"][0]
    assert session["status"] == "closed"
    assert Decimal(session["settled_cost_usd"]) == Decimal("4.018265")
    assert {o["name"] for o in session["runtime_evidence"]["operations"]} == {
        o["name"] for o in operations
    }
    assert session["runtime_evidence"]["deleted"] == "2026-11-12T01:20:00Z"
    assert session["settlement_evidence"] == (
        "compute.operations.list targetId + zone; first boot/termination and completed delete; "
        "boot disk through verified deletion (conservative)"
    )


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "restart",
        "duplicate-insert",
        "no-delete",
        "failed-delete",
        "pending-delete",
        "missing-stop-time",
        "wrong-id",
        "wrong-zone",
        "wrong-resource",
        "reversed-times",
    ],
)
def test_deleted_ambiguous_operations_keep_full_reservation(cloud, fault):
    operations = deleted_operations(cloud)
    if fault == "missing":
        operations = []
    elif fault == "restart":
        operations.append(dict(operations[0], name="restart-op", operationType="start"))
    elif fault == "duplicate-insert":
        operations.append(dict(operations[0], name="insert-again"))
    elif fault == "no-delete":
        operations = [o for o in operations if o["operationType"] != "delete"]
        operations[-1]["operationType"] = "stop"
    elif fault == "failed-delete":
        operations[1]["error"] = {"errors": [{"code": "RESOURCE_IN_USE"}]}
    elif fault == "pending-delete":
        operations[1]["status"] = "RUNNING"
    elif fault == "missing-stop-time":
        operations[-1].pop("startTime")
    elif fault == "wrong-id":
        for o in operations:
            o["targetId"] = "999"
    elif fault == "wrong-zone":
        for o in operations:
            o["zone"] = "zones/us-central1-a"
    elif fault == "wrong-resource":
        for o in operations:
            o["targetLink"] = o["targetLink"].replace("instances/run-a", "disks/run-a")
    elif fault == "reversed-times":
        operations[-1]["startTime"] = "2026-11-11T00:00:00Z"
    cloud.change(operations=operations)
    result = cloud.run("close", "run-a", ok=False)
    assert result.returncode == 2, result.stdout + result.stderr
    session = cloud.read()["sessions"][0]
    assert session["status"] == "needs_settlement"
    assert session["settled_cost_usd"] is None
    assert Decimal(json.loads(cloud.run("report").stdout)["committed_usd"]) == Decimal(
        session["reservation_usd"]
    )


@pytest.mark.parametrize(
    "duration,seconds", [("0s", 0), ("90s", 90), ("1h2m3s", 3723), ("2h", 7200)]
)
def test_flex_wait_submits_async_with_bounded_budget(cloud, monkeypatch, duration, seconds):
    cloud.setup()
    cloud.price("FLEX_START")
    real_run = subprocess.run
    budgets = []

    def run(command, **kwargs):
        if command[:4] == ["gcloud", "compute", "instances", "create"]:
            budgets.append(kwargs["timeout"])
        return real_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    for key, value in cloud.env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        sys,
        "argv",
        cloud.command(
            "launch",
            "run-a",
            "--machine-type",
            "g2-standard-8",
            "--zone",
            "europe-west4-a",
            "--provisioning",
            "FLEX_START",
            "--hours",
            "1",
            "--boot-disk-size",
            "100",
            "--image-family",
            "fixture",
            "--image-project",
            "fixture",
            "--job-uri",
            "gs://test-bucket/jobs/job.sh",
            "--request-valid-for",
            duration,
        )[1:],
    )
    assert runpy.run_path(str(GUARD))["main"]() == 0
    assert budgets == [300]
    create = next(c for c in cloud.calls() if c[:3] == ["compute", "instances", "create"])
    assert f"--request-valid-for-duration={duration}" in create
    assert "--async" in create
    cloud.change(instances=[], disks=[])
    cloud.run("resume", "run-a", "run-b")
    creates = [c for c in cloud.calls() if c[:3] == ["compute", "instances", "create"]]
    assert f"--request-valid-for-duration={duration}" in creates[-1]


@pytest.mark.parametrize(
    "model,duration",
    [
        ("SPOT", "90s"),
        ("STANDARD", "0s"),
        ("FLEX_START", "89s"),
        ("FLEX_START", "2h1s"),
        ("FLEX_START", "-1s"),
        ("FLEX_START", "tomorrow"),
    ],
)
def test_invalid_flex_wait_is_rejected_before_reserving(cloud, model, duration):
    cloud.setup()
    cloud.price(model)
    result = cloud.launch("run-a", "--request-valid-for", duration, model=model, ok=False)
    assert result.returncode == 2
    assert not cloud.read()["sessions"]
    assert not any(c[:3] == ["compute", "instances", "create"] for c in cloud.calls())


def test_create_timeout_keeps_reservation(cloud, monkeypatch):
    cloud.setup()
    cloud.price("FLEX_START")
    real_run = subprocess.run
    budgets = []

    def run(command, **kwargs):
        if command[:4] == ["gcloud", "compute", "instances", "create"]:
            budgets.append(kwargs["timeout"])
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return real_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    for key, value in cloud.env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        sys,
        "argv",
        cloud.command(
            "launch",
            "run-a",
            "--machine-type",
            "g2-standard-8",
            "--zone",
            "europe-west4-a",
            "--provisioning",
            "FLEX_START",
            "--hours",
            "1",
            "--boot-disk-size",
            "100",
            "--image-family",
            "fixture",
            "--image-project",
            "fixture",
            "--job-uri",
            "gs://test-bucket/jobs/job.sh",
            "--request-valid-for",
            "90s",
        )[1:],
    )
    assert runpy.run_path(str(GUARD))["main"]() == 2
    assert budgets == [300]
    session = cloud.read()["sessions"][0]
    assert session["status"] == "reserved"
    assert session["settled_cost_usd"] is None


def test_missing_create_start_is_recovered_by_one_bounded_describe(cloud, monkeypatch):
    cloud.setup()
    cloud.change(omit_create_start=True)
    real_run = subprocess.run
    budgets = []

    def run(command, **kwargs):
        if command[:4] == ["gcloud", "compute", "instances", "describe"]:
            budgets.append(kwargs["timeout"])
        return real_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    for key, value in cloud.env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        sys,
        "argv",
        cloud.command(
            "launch",
            "run-a",
            "--machine-type",
            "g2-standard-8",
            "--zone",
            "europe-west4-a",
            "--provisioning",
            "SPOT",
            "--hours",
            "1",
            "--boot-disk-size",
            "100",
            "--image-family",
            "fixture",
            "--image-project",
            "fixture",
            "--job-uri",
            "gs://test-bucket/jobs/job.sh",
        )[1:],
    )
    assert runpy.run_path(str(GUARD))["main"]() == 0
    assert budgets == [60]
    assert cloud.read()["sessions"][0]["last_start"]
    cloud.run("close", "run-a")
    assert cloud.read()["sessions"][0]["status"] == "closed"


def test_missing_start_after_followup_still_requires_evidence(cloud):
    cloud.setup()
    cloud.change(omit_all_start=True)
    cloud.launch()
    assert len([c for c in cloud.calls() if c[:3] == ["compute", "instances", "describe"]]) == 1
    assert cloud.read()["sessions"][0]["last_start"] is None
    cloud.change(instances=[], disks=[])
    assert cloud.run("close", "run-a", ok=False).returncode == 2
    assert cloud.read()["sessions"][0]["status"] == "needs_settlement"


def insert_operation(status="PENDING", **fields):
    return dict(
        name="insert-run-a",
        zone="https://www.googleapis.com/compute/v1/projects/test-project/zones/europe-west4-a",
        targetLink="https://www.googleapis.com/compute/v1/projects/test-project/zones/europe-west4-a/instances/run-a",
        operationType="insert",
        status=status,
        **fields,
    )


@pytest.mark.parametrize("status", ["PENDING", "RUNNING", "DONE", "UNKNOWN"])
def test_polling_error_with_insert_and_clean_inventory_retains_reservation(cloud, status):
    cloud.setup()
    cloud.price("FLEX_START")
    cloud.change(
        create_error="Polling operation failed: HTTP 502", operations=[insert_operation(status)]
    )
    assert cloud.launch(model="FLEX_START", ok=False).returncode == 2
    session = cloud.read()["sessions"][0]
    assert session["settled_cost_usd"] is None
    assert len(session["create_failure_inventory"]) == 2
    assert session["create_failure_operations"]["operations"][0]["status"] == status
    assert Decimal(json.loads(cloud.run("report").stdout)["committed_usd"]) == Decimal(
        session["reservation_usd"]
    )


def test_done_with_error_insert_and_clean_inventory_cancels(cloud):
    cloud.setup()
    cloud.change(
        create_error="HTTP 502 while polling",
        operations=[
            insert_operation(
                "DONE",
                error={"errors": [{"code": "ZONE_RESOURCE_POOL_EXHAUSTED"}]},
                user="not-retained@example.invalid",
            )
        ],
    )
    assert cloud.launch(ok=False).returncode == 2
    session = cloud.read()["sessions"][0]
    assert session["status"] == "cancelled" and Decimal(session["settled_cost_usd"]) == 0
    assert "user" not in session["create_failure_operations"]["operations"][0]


@pytest.mark.parametrize(
    "fault",
    [
        {"operations_error": True},
        {"operations_warning": "partial results"},
        {"operations_response": None},
        {"operations_response": [{}]},
    ],
)
def test_create_error_unreadable_or_ambiguous_operations_keep_reservation(cloud, fault):
    cloud.setup()
    cloud.change(create_error="INVALID_ARGUMENT", **fault)
    assert cloud.launch(ok=False).returncode == 2
    session = cloud.read()["sessions"][0]
    assert session["settled_cost_usd"] is None
    assert session["create_failure_operations"]["error"]


@pytest.mark.parametrize("kind", ["instances", "disks"])
def test_cancelled_resource_reopens_on_close_even_over_cap(cloud, kind):
    cloud.setup(cap="6")
    cloud.launch()
    ledger = cloud.read()
    session = ledger["sessions"][0]
    session.update(status="cancelled", settled_cost_usd="0.000000")
    # A late resource has no saved first boot; reconciliation must retain money.
    session.pop("last_start")
    cloud.ledger.write_text(json.dumps(ledger))
    cloud.run("hold", "set", "storage", "5")
    if kind == "disks":
        cloud.change(instances=[])
    result = cloud.run("sweep", ok=False)
    assert result.returncode == 1 and "cancelled_session_resource" in result.stdout
    before = cloud.ledger.read_bytes()
    cloud.run("close", "run-a", "--dry-run")
    assert cloud.ledger.read_bytes() == before
    result = cloud.run("close", "run-a", ok=False)
    assert result.returncode == 2 and "overrun" in result.stderr
    reopened = cloud.read()["sessions"][0]
    assert reopened["status"] == "needs_settlement" and reopened["settled_cost_usd"] is None
    assert reopened["reopened"]["committed_usd"] == "10.017124"
    assert Decimal(json.loads(cloud.run("report").stdout)["committed_usd"]) > 6
    assert cloud.launch("run-b", ok=False).returncode == 2
    cloud.change(disks=[])
    cloud.run("close", "run-a", "--settled-usd", "1", "--evidence", "billing fixture")
    assert cloud.read()["sessions"][0]["status"] == "closed"


def test_sweep_delete_already_autodeleted_disk_keeps_findings_exit_code(cloud):
    cloud.setup()
    cloud.launch()
    ledger = cloud.read()
    ledger["sessions"][0].update(status="closed", settled_cost_usd="1")
    cloud.ledger.write_text(json.dumps(ledger))
    result = cloud.run("sweep", "--delete", ok=False)
    assert result.returncode == 1, result.stderr
    assert cloud.run("sweep").returncode == 0


def test_flex_async_records_operation_without_polling_or_disk_label(cloud):
    cloud.setup()
    cloud.price("FLEX_START")
    cloud.change(async_pending=True)
    result = cloud.launch("run-a", "--request-valid-for", "2h", model="FLEX_START")
    session = cloud.read()["sessions"][0]
    assert session["status"] == "open" and session["operation_name"] == "insert-run-a"
    assert session["settled_cost_usd"] is None
    assert "insert-run-a" in result.stdout
    create = next(c for c in cloud.calls() if c[:3] == ["compute", "instances", "create"])
    assert "--async" in create and any(
        "guard-reserved-end=" + session["reserved_end"] in a for a in create
    )
    assert not any(
        c[:3] in (["compute", "instances", "describe"], ["compute", "disks", "add-labels"])
        for c in cloud.calls()
    )


@pytest.mark.parametrize("outcome", ["pending", "running", "failed", "expired"])
def test_flex_status_and_wait_report_queue_state(cloud, outcome):
    cloud.setup()
    cloud.price("FLEX_START")
    cloud.change(async_pending=outcome != "running")
    cloud.launch("run-a", "--request-valid-for", "90s", model="FLEX_START")
    if outcome in {"failed", "expired"}:
        code = "FLEX_START_REQUEST_EXPIRED" if outcome == "expired" else "QUOTA_EXCEEDED"
        cloud.change(
            operations=[
                insert_operation(
                    "DONE",
                    error={
                        "errors": [
                            {
                                "code": code,
                                "message": "request expired" if outcome == "expired" else "quota",
                            }
                        ]
                    },
                )
            ]
        )
    status = json.loads(cloud.run("status", "run-a").stdout)[0]
    assert status["queue"]["state"] == outcome
    result = cloud.run("wait", "run-a", "--timeout", "0", ok=False)
    assert json.loads(result.stdout)["queue"]["state"] == outcome
    assert result.returncode == (0 if outcome == "running" else 1)
    assert cloud.read()["sessions"][0]["settled_cost_usd"] is None
    if outcome == "running":
        assert cloud.read()["sessions"][0]["last_start"]
        cloud.run("close", "run-a")
        assert Decimal(cloud.read()["sessions"][0]["settled_cost_usd"]) > 0


@pytest.mark.parametrize("code", ["FLEX_START_REQUEST_EXPIRED", "QUOTA_EXCEEDED"])
def test_close_failed_flex_request_settles_zero_with_operation_and_absence(cloud, code):
    cloud.setup()
    cloud.price("FLEX_START")
    cloud.change(async_pending=True)
    cloud.launch(model="FLEX_START")
    cloud.change(operations=[insert_operation("DONE", error={"errors": [{"code": code}]})])
    cloud.run("close", "run-a")
    session = cloud.read()["sessions"][0]
    assert session["status"] == "closed" and Decimal(session["settled_cost_usd"]) == 0
    assert len(session["create_failure_inventory"]) == 2
    assert session["create_failure_operations"]["operations"][0]["error"]


@pytest.mark.parametrize(
    "fault", ["pending", "missing", "unreadable", "success", "inventory-warning"]
)
def test_close_ambiguous_flex_request_retains_full_reservation(cloud, fault):
    cloud.setup()
    cloud.price("FLEX_START")
    cloud.change(async_pending=True)
    cloud.launch(model="FLEX_START")
    if fault == "missing":
        cloud.change(operations=[])
    elif fault == "unreadable":
        cloud.change(operations_error=True)
    elif fault == "success":
        cloud.change(operations=[insert_operation("DONE")])
    elif fault == "inventory-warning":
        cloud.change(
            operations=[insert_operation("DONE", error={"errors": [{"code": "QUOTA_EXCEEDED"}]})],
            create_failed={"instance": "run-a", "disk": cloud.read()["sessions"][0]["boot_disk"]},
            failure_inventory=[{}, {"disks_warning": "partial inventory"}],
        )
    assert cloud.run("close", "run-a", ok=False).returncode == 2
    session = cloud.read()["sessions"][0]
    assert session["status"] == "needs_settlement" and session["settled_cost_usd"] is None


def test_pending_insert_with_alternate_api_hostname_retains_reservation(cloud):
    cloud.setup()
    operation = insert_operation()
    operation["targetLink"] = operation["targetLink"].replace(
        "www.googleapis.com", "compute.googleapis.com"
    )
    cloud.change(create_error="HTTP 502", operations=[operation])
    assert cloud.launch(ok=False).returncode == 2
    assert cloud.read()["sessions"][0]["settled_cost_usd"] is None


def test_pending_flex_blocks_resume_and_shared_prefix_launch(cloud):
    cloud.setup()
    cloud.price("FLEX_START")
    cloud.change(async_pending=True)
    cloud.launch(model="FLEX_START")
    prefix = cloud.read()["sessions"][0]["state_uri"]
    assert cloud.run("resume", "run-a", "run-b", ok=False).returncode == 2
    assert cloud.launch("run-c", "--state-uri", prefix, ok=False).returncode == 2
    assert len(cloud.read()["sessions"]) == 1


def test_flex_dry_run_has_only_async_submission_and_changes_no_bytes(cloud):
    cloud.setup()
    cloud.price("FLEX_START")
    before = cloud.ledger.read_bytes()
    result = cloud.launch("dry", "--dry-run", model="FLEX_START")
    assert "--async" in result.stdout and "add-labels" not in result.stdout
    assert before == cloud.ledger.read_bytes()


@pytest.mark.parametrize(
    "field,value", [("zone", None), ("targetLink", None), ("status", []), ("error", {"errors": []})]
)
def test_malformed_insert_evidence_cannot_cancel(cloud, field, value):
    cloud.setup()
    operation = insert_operation("DONE", error={"errors": [{"code": "QUOTA_EXCEEDED"}]})
    operation[field] = value
    # Force the returned malformed object past the fake's server-side filtering.
    cloud.change(create_error="HTTP 502", operations_response=[operation])
    result = cloud.launch(ok=False)
    assert result.returncode == 2 and "Traceback" not in result.stderr
    assert cloud.read()["sessions"][0]["settled_cost_usd"] is None


def test_wait_releases_lock_and_observes_pending_to_running_without_extending_deadline(
    cloud, monkeypatch, capsys
):
    cloud.setup()
    cloud.price("FLEX_START")
    cloud.change(async_pending=True)
    cloud.launch(model="FLEX_START")
    original_end = cloud.read()["sessions"][0]["reserved_end"]

    from types import SimpleNamespace

    clock = [0.0]

    def advance(seconds):
        clock[0] += seconds
        with cloud.ledger.with_suffix(".json.lock").open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cloud.change(
            operations=[insert_operation("DONE", targetId="1")],
            instances=[
                dict(
                    name="run-a",
                    id="1",
                    zone="europe-west4-a",
                    status="RUNNING",
                    creationTimestamp="2026-11-13T00:00:00Z",
                    lastStartTimestamp="2026-11-13T00:01:00Z",
                )
            ],
        )

    for key, value in cloud.env.items():
        monkeypatch.setenv(key, value)
    module = runpy.run_path(str(GUARD))
    monkeypatch.setitem(
        module["wait_for_session"].__globals__,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], sleep=advance),
    )
    args = module["parser"]().parse_args(
        [
            "--ledger",
            str(cloud.ledger),
            "wait",
            "run-a",
            "--timeout",
            "20",
            "--poll-interval",
            "0.01",
        ]
    )
    assert module["wait_for_session"](cloud.ledger, args) == 0
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item["queue"]["state"] for item in output] == ["pending", "running"]
    assert cloud.read()["sessions"][0]["reserved_end"] == original_end


@pytest.mark.parametrize("seconds", [60, 600, 3600])
def test_copy_interval_stored_in_metadata_and_preserved_by_resume(cloud, seconds):
    cloud.setup()
    cloud.launch("run-a", "--copy-interval", str(seconds))
    assert cloud.read()["sessions"][0]["launch"]["copy_interval"] == seconds
    create = next(c for c in cloud.calls() if c[:3] == ["compute", "instances", "create"])
    assert f"guard-copy-interval={seconds}" in next(
        a for a in create if a.startswith("--metadata=")
    )
    cloud.change(instances=[], disks=[])
    cloud.run("resume", "run-a", "run-b")
    assert cloud.read()["sessions"][1]["launch"]["copy_interval"] == seconds


@pytest.mark.parametrize("seconds", ["59", "3601", "1.5", "oops"])
def test_invalid_copy_interval_never_creates_resources(cloud, seconds):
    cloud.setup()
    assert cloud.launch("run-a", "--copy-interval", seconds, ok=False).returncode == 2
    assert cloud.read()["sessions"] == []


LIVE_PREEMPTIONS = json.loads((ROOT / "tests/test_gcp/fixtures/spot_preemptions.json").read_text())


@pytest.mark.parametrize("fixture", LIVE_PREEMPTIONS, ids=lambda f: f["label"])
def test_live_spot_preemption_history_settles_through_operation_end(cloud, fixture):
    operations = deleted_operations(cloud)
    ledger = cloud.read()
    session = ledger["sessions"][0]
    session.update(
        last_start=fixture["last_start"], instance_created_at=fixture["instance_created_at"]
    )
    cloud.ledger.write_text(json.dumps(ledger))
    base = operations[0]
    ops = [
        dict(o, **{k: base[k] for k in ("targetId", "targetLink", "zone")})
        for o in fixture["operations"]
    ]
    cloud.change(operations=ops)
    cloud.run("close", "run-a")
    saved = cloud.read()["sessions"][0]
    end = next(o["endTime"] for o in ops if o["operationType"] == "compute.instances.preempted")
    assert saved["runtime_evidence"]["stop"] == saved["runtime_evidence"]["deleted"] == end
    assert saved["runtime_evidence"]["start"] == fixture["last_start"]
    assert saved["settlement_evidence"] == (
        "compute.operations.list targetId + zone; Spot DELETE: "
        "machine from recorded start and disk from creation through preemption endTime; "
        "two clean instance/disk absence reads"
    )
    assert len(saved["settlement_inventory"]) == 2

    def instant(s):
        return datetime.fromisoformat(s)

    machine = (
        Decimal(str((instant(end) - instant(fixture["last_start"])).total_seconds()))
        / 3600
        * Decimal("4")
    )
    disk = (
        Decimal(str((instant(end) - instant(fixture["instance_created_at"])).total_seconds()))
        / 3600
        * Decimal(saved["disk_hourly_usd"])
    )
    assert Decimal(saved["settled_cost_usd"]) >= machine + disk
    assert Decimal(saved["settled_cost_usd"]) < machine + disk + Decimal("0.000001")


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "duplicate",
        "failed",
        "pending",
        "stop",
        "malformed",
        "disk",
        "second-instance",
        "second-disk",
    ],
)
def test_ambiguous_spot_preemption_keeps_reservation(cloud, fault):
    operations = deleted_operations(cloud)
    operations = [o for o in operations if o["operationType"] != "delete"]
    if fault == "unknown":
        operations.append(dict(operations[0], operationType="setMetadata"))
    if fault == "duplicate":
        operations.append(dict(operations[0]))
    if fault == "failed":
        operations[-1]["error"] = {"errors": [{"code": "FAILED"}]}
    if fault == "pending":
        operations[-1]["status"] = "RUNNING"
    if fault == "stop":
        operations.append(dict(operations[-1], operationType="stop"))
    if fault == "malformed":
        operations.append(None)
    cloud.change(operations_response=operations)
    if fault == "disk":
        cloud.change(
            disks=[{"name": cloud.read()["sessions"][0]["boot_disk"], "zone": "europe-west4-a"}]
        )
    if fault.startswith("second-"):
        kind = "instances" if fault.endswith("instance") else "disks"
        cloud.change(
            create_failed={"instance": "run-a", "disk": cloud.read()["sessions"][0]["boot_disk"]},
            failure_inventory=[
                {},
                {
                    kind: [
                        {
                            "name": "instance" if kind == "instances" else "disk",
                            "zone": "europe-west4-a",
                        }
                    ]
                },
            ],
        )
    result = cloud.run("close", "run-a", ok=False)
    assert result.returncode == 2, result.stdout + result.stderr
    session = cloud.read()["sessions"][0]
    assert session["settled_cost_usd"] is None
    assert session["status"] == "needs_settlement"


def test_status_reports_preemption_source(cloud):
    cloud.setup()
    cloud.launch()
    cloud.change(
        markers=["PREEMPTED"],
        marker_bodies={"PREEMPTED": "job_uri=gs://test-bucket/jobs/job.sh\nsource=watcher\n"},
    )
    assert (
        json.loads(cloud.run("status", "run-a").stdout)[0]["markers"]["PREEMPTED"]["source"]
        == "watcher"
    )


@pytest.mark.parametrize("present", ["instance", "unknown-delete-op"])
def test_contradictory_preemption_or_unknown_delete_history_keeps_money(cloud, present):
    operations = deleted_operations(cloud)
    if present == "instance":
        operations = [o for o in operations if o["operationType"] != "delete"]
        # Same ID as the supposedly preempted VM; normal close must not hide the contradiction.
        cloud.change(
            instances=[
                dict(
                    name="run-a",
                    id="1",
                    zone="europe-west4-a",
                    status="RUNNING",
                    lastStartTimestamp="2026-11-12T00:10:00Z",
                    creationTimestamp="2026-11-12T00:00:00Z",
                    disks=[],
                )
            ]
        )
    else:
        operations.append(dict(operations[0], name="unknown-op", operationType="setMetadata"))
    cloud.change(operations=operations)
    assert cloud.run("close", "run-a", ok=False).returncode == 2
    assert cloud.read()["sessions"][0]["settled_cost_usd"] is None


@pytest.mark.parametrize(
    "fault",
    ["partial-operations", "missing-recorded-start", "second-preemption", "start-before-insert"],
)
def test_preemption_incomplete_evidence_keeps_full_reservation(cloud, fault):
    operations = deleted_operations(cloud)
    operations = [o for o in operations if o["operationType"] != "delete"]
    if fault == "partial-operations":
        cloud.change(operations_warning="WARNING: partial operation history")
    if fault == "missing-recorded-start":
        ledger = cloud.read()
        ledger["sessions"][0]["last_start"] = None
        cloud.ledger.write_text(json.dumps(ledger))
    if fault == "second-preemption":
        operations.append(dict(operations[-1], name="second-preemption"))
    if fault == "start-before-insert":
        ledger = cloud.read()
        ledger["sessions"][0]["last_start"] = "2026-11-12T00:00:00Z"
        cloud.ledger.write_text(json.dumps(ledger))
    cloud.change(operations=operations)
    assert cloud.run("close", "run-a", ok=False).returncode == 2
    assert cloud.read()["sessions"][0]["settled_cost_usd"] is None


def test_preemption_set_labels_is_the_only_additional_harmless_type(cloud):
    operations = deleted_operations(cloud)
    operations = [o for o in operations if o["operationType"] != "delete"]
    operations.append(dict(operations[0], name="labels", operationType="setLabels"))
    cloud.change(operations=operations)
    cloud.run("close", "run-a")
    assert cloud.read()["sessions"][0]["status"] == "closed"


def test_cancelled_session_reopens_money_before_preemption_history_errors(cloud):
    deleted_operations(cloud)
    ledger = cloud.read()
    session = ledger["sessions"][0]
    session.update(status="cancelled", settled_cost_usd="0.000000")
    cloud.ledger.write_text(json.dumps(ledger))
    cloud.change(
        instances=[dict(name="run-a", id="1", zone="europe-west4-a", status="RUNNING")],
        operations_warning="partial operation history",
    )
    assert cloud.run("close", "run-a", ok=False).returncode == 2
    saved = cloud.read()["sessions"][0]
    assert saved["status"] == "needs_settlement" and saved["settled_cost_usd"] is None


@pytest.mark.parametrize("command", ["status", "wait", "close"])
@pytest.mark.parametrize("vm_status", ["STAGING", "RUNNING"])
def test_reserved_session_adopts_owned_live_instance(cloud, command, vm_status):
    cloud.setup()
    cloud.launch()
    ledger = cloud.read()
    session = ledger["sessions"][0]
    session.update(status="reserved")
    for key in ("instance_id", "last_start", "instance_created_at"):
        session.pop(key, None)
    original_end = session["reserved_end"]
    cloud.ledger.write_text(json.dumps(ledger))
    state = json.loads(cloud.state.read_text())
    vm = state["instances"][0]
    vm["status"] = vm_status
    operation = insert_operation(
        "RUNNING" if vm_status == "STAGING" else "DONE",
        targetId=vm["id"],
        insertTime=vm["creationTimestamp"],
        startTime=vm["creationTimestamp"],
        endTime=vm["lastStartTimestamp"],
    )
    cloud.change(instances=[vm], operations=[operation])
    args = (command, "run-a", "--timeout", "0") if command == "wait" else (command, "run-a")
    result = cloud.run(*args, ok=False)
    saved = cloud.read()["sessions"][0]
    assert saved["instance_id"] == vm["id"], result.stderr
    assert saved["instance_created_at"] == vm["creationTimestamp"]
    assert saved["last_start"] == vm["lastStartTimestamp"]
    assert saved["reserved_end"] == original_end
    assert saved["adoption"]["source"] == "owned instance + insert operation"
    expected = (
        2
        if command == "close" and vm_status == "STAGING"
        else 1
        if command == "wait" and vm_status != "RUNNING"
        else 0
    )
    assert result.returncode == expected


@pytest.mark.parametrize("command", ["status", "wait", "close"])
@pytest.mark.parametrize("fault", ["token", "label", "disk-token", "operation-id", "failed-insert"])
def test_reserved_adoption_refuses_unproven_ownership(cloud, command, fault):
    cloud.setup()
    cloud.launch()
    ledger = cloud.read()
    session = ledger["sessions"][0]
    session["status"] = "reserved"
    for key in ("instance_id", "last_start", "instance_created_at"):
        session.pop(key, None)
    state = json.loads(cloud.state.read_text())
    vm = state["instances"][0]
    op = insert_operation(
        "DONE",
        targetId=vm["id"],
        insertTime=vm["creationTimestamp"],
        startTime=vm["creationTimestamp"],
        endTime=vm["lastStartTimestamp"],
    )
    if fault == "token":
        vm["labels"]["guard-ledger"] = "foreign"
    if fault == "label":
        vm["labels"]["guard-session"] = "foreign"
    if fault == "disk-token":
        session["boot_disk"] = "gcp-foreign-run-a"
    if fault == "operation-id":
        op["targetId"] = "foreign"
    if fault == "failed-insert":
        op["error"] = {"errors": [{"code": "FAILED"}]}
    cloud.ledger.write_text(json.dumps(ledger))
    cloud.change(instances=[vm], operations=[op])
    args = (command, "run-a", "--timeout", "0") if command == "wait" else (command, "run-a")
    cloud.run(*args, ok=False)
    saved = cloud.read()["sessions"][0]
    assert not saved.get("instance_id")
    assert saved["settled_cost_usd"] is None
    assert not any(
        c[:3] in (["compute", "instances", "stop"], ["compute", "instances", "delete"])
        for c in cloud.calls()
    )


@pytest.mark.parametrize("observe_first", [None, "status", "wait"])
def test_canary_w2_reserved_adoption_and_no_delete_settlement(cloud, observe_first):
    cloud.setup()
    fixture = json.loads((ROOT / "tests/test_gcp/fixtures/canary_w2.json").read_text())
    ledger = cloud.read()
    ledger["config"]["project"] = fixture["project"]
    session = fixture["session"]
    session["boot_disk"] = session["boot_disk"].replace("<ledger-token>", ledger["config"]["token"])
    ledger["sessions"] = [session]
    cloud.ledger.write_text(json.dumps(ledger))
    cloud.change(instances=[], disks=[], operations=fixture["operations"])
    if observe_first:
        args = (
            (observe_first, "canary-w2", "--timeout", "0")
            if observe_first == "wait"
            else (observe_first, "canary-w2")
        )
        cloud.run(*args, ok=observe_first != "wait")
        assert cloud.read()["sessions"][0]["status"] == "open"
    cloud.run("close", "canary-w2")
    saved = cloud.read()["sessions"][0]
    inserted = fixture["operations"][0]
    assert saved["instance_id"] == inserted["targetId"]
    assert saved["runtime_evidence"]["start"] == inserted["insertTime"]
    assert saved["runtime_evidence"]["creation"] == inserted["insertTime"]
    assert saved["runtime_evidence"]["stop"] == "2026-11-13T11:53:13.001Z"
    assert saved["settlement_evidence"] == (
        "compute.operations.list targetId + zone; Spot DELETE: "
        "machine and disk charged from insert request (no instance start recorded) "
        "through preemption endTime; two clean instance/disk absence reads"
    )
    assert saved["settled_cost_usd"] == "0.038568"
    assert len(saved["settlement_inventory"]) == 2


@pytest.mark.parametrize(
    "fault", ["no-retained-insert", "token", "old-insert", "duplicate", "mismatched-retained"]
)
def test_gone_reserved_session_without_ownership_proof_stays_held(cloud, fault):
    cloud.setup()
    fixture = json.loads((ROOT / "tests/test_gcp/fixtures/canary_w2.json").read_text())
    ledger = cloud.read()
    ledger["config"]["project"] = fixture["project"]
    session = fixture["session"]
    session["boot_disk"] = session["boot_disk"].replace("<ledger-token>", ledger["config"]["token"])
    if fault == "token":
        session["boot_disk"] = "gcp-foreign-canary-w2"
    if fault == "no-retained-insert":
        session.pop("create_failure_operations")
    if fault == "old-insert":
        session["created_at"] = "2026-11-13T11:50:00Z"
    if fault == "duplicate":
        fixture["operations"].append(dict(fixture["operations"][0], name="another-insert"))
    if fault == "mismatched-retained":
        session["create_failure_operations"]["operations"][0]["targetId"] = "foreign"
    ledger["sessions"] = [session]
    cloud.ledger.write_text(json.dumps(ledger))
    cloud.change(instances=[], disks=[], operations=fixture["operations"])
    result = cloud.run("close", "canary-w2", ok=False)
    assert result.returncode == 2
    assert cloud.read()["sessions"][0]["settled_cost_usd"] is None


def test_reserved_staging_adoption_updates_start_from_later_instance_read(cloud):
    cloud.setup()
    cloud.launch()
    ledger = cloud.read()
    session = ledger["sessions"][0]
    session["status"] = "reserved"
    for key in ("instance_id", "last_start", "instance_created_at"):
        session.pop(key, None)
    cloud.ledger.write_text(json.dumps(ledger))
    state = json.loads(cloud.state.read_text())
    vm = state["instances"][0]
    actual_start = vm.pop("lastStartTimestamp")
    vm["status"] = "STAGING"
    op = insert_operation(
        "RUNNING",
        targetId=vm["id"],
        insertTime=vm["creationTimestamp"],
        startTime=vm["creationTimestamp"],
    )
    cloud.change(instances=[vm], operations=[op])
    cloud.run("status")
    saved = cloud.read()["sessions"][0]
    assert saved["adoption"]["charge_from_insert_request"]
    assert saved["last_start"] == op["insertTime"]
    vm.update(lastStartTimestamp=actual_start, status="RUNNING")
    op.update(status="DONE", endTime=actual_start)
    cloud.change(instances=[vm], operations=[op])
    cloud.run("status")
    saved = cloud.read()["sessions"][0]
    assert saved["last_start"] == actual_start
    assert not saved["adoption"]["charge_from_insert_request"]
    assert saved["reserved_end"] == session["reserved_end"]


def test_adopted_session_repeated_start_cannot_hide_earlier_runtime(cloud):
    cloud.setup()
    cloud.launch()
    ledger = cloud.read()
    session = ledger["sessions"][0]
    session.update(status="reserved", created_at="2026-11-12T00:00:00Z")
    for key in ("instance_id", "last_start", "instance_created_at"):
        session.pop(key, None)
    cloud.ledger.write_text(json.dumps(ledger))
    vm = json.loads(cloud.state.read_text())["instances"][0]
    vm.update(creationTimestamp="2026-11-12T00:00:00Z", lastStartTimestamp="2026-11-12T00:10:00Z")
    op = insert_operation(
        "DONE",
        targetId=vm["id"],
        insertTime=vm["creationTimestamp"],
        startTime=vm["creationTimestamp"],
        endTime=vm["lastStartTimestamp"],
    )
    cloud.change(instances=[vm], operations=[op])
    cloud.run("status", "run-a")
    vm["lastStartTimestamp"] = "2026-11-12T00:20:00Z"
    cloud.change(instances=[vm])
    cloud.run("status", "run-a")
    saved = cloud.read()["sessions"][0]
    assert saved["last_start"] == "2026-11-12T00:10:00Z"
    assert saved["runtime_uncertain"]
    assert cloud.run("close", "run-a", ok=False).returncode == 2
    assert cloud.read()["sessions"][0]["settled_cost_usd"] is None


@pytest.mark.parametrize("orphan_disk", [False, True])
def test_close_inventory_evidence_excludes_account_and_guest_metadata(cloud, orphan_disk):
    cloud.setup()
    cloud.launch()
    state = json.loads(cloud.state.read_text())
    vm, disk = state["instances"][0], state["disks"][0]
    private = {
        "serviceAccounts": [{"email": "private-operator@example.test"}],
        "metadata": {"items": [{"key": "startup-script", "value": "private-guest-script"}]},
        "networkInterfaces": [{"networkIP": "private-network-address"}],
        "user": "private-account-id",
    }
    vm.update(private)
    disk.update(private, id="disk-id", status="READY")
    cloud.change(instances=[vm], disks=[disk], orphan_disk=orphan_disk)
    result = cloud.run("close", "run-a", ok=not orphan_disk)
    assert result.returncode == (2 if orphan_disk else 0)
    saved = cloud.read()["sessions"][0]
    reads = saved["settlement_inventory"]
    assert len(reads) == 2
    assert reads[0]["instances"] == [
        {
            "name": "run-a",
            "id": "1",
            "zone": "europe-west4-a",
            "status": "RUNNING",
            "labels": vm["labels"],
        }
    ]
    expected_disk = {
        "name": disk["name"],
        "id": "disk-id",
        "zone": "europe-west4-a",
        "status": "READY",
        "labels": disk["labels"],
    }
    assert reads[0]["disks"] == [expected_disk]
    assert reads[1]["instances"] == []
    assert reads[1]["disks"] == ([expected_disk] if orphan_disk else [])
    output = cloud.ledger.read_text() + result.stdout + cloud.run("report").stdout
    assert "private-" not in output
    assert saved["settled_cost_usd"] is None if orphan_disk else saved["status"] == "closed"


def reserved_live_session(cloud, provisional=False):
    cloud.setup()
    cloud.launch()
    ledger = cloud.read()
    session = ledger["sessions"][0]
    session["status"] = "reserved"
    for key in ("instance_id", "last_start", "instance_created_at"):
        session.pop(key, None)
    cloud.ledger.write_text(json.dumps(ledger))
    vm = json.loads(cloud.state.read_text())["instances"][0]
    operation = insert_operation(
        "DONE",
        targetId=vm["id"],
        insertTime=vm["creationTimestamp"],
        startTime=vm["creationTimestamp"],
        endTime=vm["lastStartTimestamp"],
    )
    if provisional:
        vm.pop("lastStartTimestamp")
        vm["status"] = "STAGING"
        operation.pop("endTime")
        operation["status"] = "RUNNING"
    cloud.change(instances=[vm], operations=[operation])
    return vm, operation


@pytest.mark.parametrize("session_state", ["reserved", "adopted", "open"])
@pytest.mark.parametrize("foreign", ["ledger", "session", "zone", "unlabeled"])
def test_status_ignores_foreign_same_name_instances(cloud, session_state, foreign):
    vm, _ = reserved_live_session(cloud)
    if session_state != "reserved":
        cloud.run("status", "run-a")
    if session_state == "open":
        ledger = cloud.read()
        ledger["sessions"][0].pop("adoption")
        cloud.ledger.write_text(json.dumps(ledger))
    original = cloud.read()["sessions"][0]
    if foreign == "ledger":
        vm["labels"]["guard-ledger"] = "another-ledger"
    elif foreign == "session":
        vm["labels"]["guard-session"] = "another-session"
    elif foreign == "zone":
        vm["zone"] = "europe-west4-b"
    else:
        vm.pop("labels")
    cloud.change(instances=[vm], operations=[])
    result = cloud.run("status")
    assert json.loads(result.stdout)[0]["instances"] == []
    saved = cloud.read()["sessions"][0]
    assert saved["settled_cost_usd"] is None
    assert saved["reservation_usd"] == original["reservation_usd"]
    assert saved["reserved_end"] == original["reserved_end"]
    assert saved.get("instance_id") == original.get("instance_id")
    assert not any(c[2] in {"stop", "delete"} for c in cloud.calls() if c[0] == "compute")


@pytest.mark.parametrize("command", ["status", "close"])
def test_adopted_session_ignores_same_name_instance_in_other_zone(cloud, command):
    vm, _ = reserved_live_session(cloud)
    cloud.run("status", "run-a")
    foreign = dict(vm, id="foreign-vm", zone="europe-west4-b")
    cloud.change(instances=[foreign, vm])
    result = cloud.run(command, "run-a")
    if command == "status":
        assert json.loads(result.stdout)[0]["instances"] == [vm]
    else:
        assert cloud.read()["sessions"][0]["status"] == "closed"
        assert json.loads(cloud.state.read_text())["instances"] == [foreign]


@pytest.mark.parametrize("command", ["status", "wait", "close"])
@pytest.mark.parametrize("history", ["expired", "partial", "unreadable"])
def test_adopted_session_survives_unavailable_insert_history(cloud, command, history):
    vm, _ = reserved_live_session(cloud)
    cloud.run("status", "run-a")
    original = cloud.read()["sessions"][0]
    cloud.change(
        operations=[],
        operations_warning="partial list" if history == "partial" else "",
        operations_error=history == "unreadable",
    )
    args = (
        ("wait", "run-a", "--timeout", "0")
        if command == "wait"
        else ("close", "run-a", "--settled-usd", "0.25", "--evidence", "saved billing export")
        if command == "close"
        else ("status", "run-a")
    )
    cloud.run(*args)
    saved = cloud.read()["sessions"][0]
    assert saved["instance_id"] == vm["id"]
    assert saved["last_start"] == original["last_start"]
    assert saved["reserved_end"] == original["reserved_end"]
    if command == "close":
        assert saved["status"] == "closed" and saved["settled_cost_usd"] == "0.250000"
        assert not json.loads(cloud.state.read_text())["instances"]
    else:
        assert saved["settled_cost_usd"] is None


@pytest.mark.parametrize("command", ["status", "close"])
def test_provisional_adoption_can_refresh_or_settle_after_insert_expires(cloud, command):
    vm, _ = reserved_live_session(cloud, provisional=True)
    cloud.run("status", "run-a")
    original = cloud.read()["sessions"][0]
    assert original["adoption"]["charge_from_insert_request"]
    if command == "status":
        vm.update(status="RUNNING", lastStartTimestamp=vm["creationTimestamp"])
    cloud.change(instances=[vm], operations=[])
    args = (
        ("status", "run-a")
        if command == "status"
        else ("close", "run-a", "--settled-usd", "0.25", "--evidence", "saved billing export")
    )
    cloud.run(*args)
    saved = cloud.read()["sessions"][0]
    assert saved["reserved_end"] == original["reserved_end"]
    if command == "status":
        assert saved["last_start"] == vm["lastStartTimestamp"]
        assert not saved["adoption"]["charge_from_insert_request"]
    else:
        assert saved["status"] == "closed" and saved["settled_cost_usd"] == "0.250000"


@pytest.mark.parametrize(
    "fault,manual",
    [
        ("ledger", False),
        ("session", False),
        ("zone", False),
        ("id", False),
        ("restart", False),
        ("gone", False),
        ("ledger", True),
        ("session", True),
        ("id", True),
    ],
)
def test_adopted_session_with_expired_history_retains_safety_checks(cloud, fault, manual):
    vm, _ = reserved_live_session(cloud)
    cloud.run("status", "run-a")
    original = cloud.read()["sessions"][0]
    if fault in {"ledger", "session"}:
        vm["labels"]["guard-" + fault] = "foreign"
    elif fault == "zone":
        vm["zone"] = "europe-west4-b"
    elif fault == "id":
        vm["id"] = "foreign"
    elif fault == "restart":
        vm["lastStartTimestamp"] = (
            datetime.fromisoformat(vm["lastStartTimestamp"]) + timedelta(microseconds=1)
        ).isoformat()
    cloud.change(instances=[] if fault == "gone" else [vm], operations=[])
    args = ("--settled-usd", "0.25", "--evidence", "saved billing export") if manual else ()
    result = cloud.run("close", "run-a", *args, ok=False)
    assert result.returncode == 2
    saved = cloud.read()["sessions"][0]
    assert saved["settled_cost_usd"] is None
    assert saved["reservation_usd"] == original["reservation_usd"]
    assert saved["last_start"] == original["last_start"]
    if fault == "restart":
        assert saved["runtime_uncertain"]


LIVE_GUEST_SHUTDOWNS = json.loads(
    (ROOT / "tests/test_gcp/fixtures/guest_shutdowns.json").read_text()
)


def guest_shutdown_history(cloud, fixture=LIVE_GUEST_SHUTDOWNS[1], *, model="SPOT"):
    operations = deleted_operations(cloud)
    ledger = cloud.read()
    session = ledger["sessions"][0]
    session["launch"]["provisioning"] = model
    for key in ("last_start", "instance_created_at", "hourly_usd", "disk_hourly_usd"):
        session[key] = fixture[key]
    cloud.ledger.write_text(json.dumps(ledger))
    base = operations[0]
    operations = [
        dict(op, **{key: base[key] for key in ("targetId", "targetLink", "zone")})
        for op in fixture["operations"]
    ]
    cloud.change(operations=operations)
    return operations


def close_in_process(cloud, monkeypatch, *, elapsed=None, frozen=None, extra=(), module=None):
    from types import SimpleNamespace

    if module is None:
        module = runpy.run_path(str(GUARD))
    for key, value in cloud.env.items():
        monkeypatch.setenv(key, value)
    globals_ = module["Guard"].close.__globals__
    if elapsed is not None:

        def sleep(seconds):
            elapsed[0] += seconds

        monkeypatch.setitem(
            globals_, "time", SimpleNamespace(monotonic=lambda: elapsed[0], sleep=sleep)
        )
    if frozen is not None:
        monkeypatch.setitem(globals_, "now", lambda: frozen)
    args = module["parser"]().parse_args(["--ledger", str(cloud.ledger), "close", "run-a", *extra])
    module["Guard"](cloud.ledger).close(args)


@pytest.mark.parametrize(
    "with_delete,expected,expected_evidence",
    [
        (
            True,
            "3.749472",
            "compute.operations.list targetId + zone; single guestTerminate: "
            "machine from recorded start and disk from creation through "
            "completed delete endTime (conservative); boot disk through verified deletion (conservative)",
        ),
        (
            False,
            "3.727433",
            "compute.operations.list targetId + zone; single guestTerminate: "
            "machine from recorded start and disk from creation through "
            "guestTerminate endTime; two clean instance/disk absence reads",
        ),
    ],
)
def test_real_guest_shutdown_history_settles_conservatively(
    cloud, with_delete, expected, expected_evidence
):
    operations = guest_shutdown_history(cloud)
    if not with_delete:
        operations = [op for op in operations if op["operationType"] != "delete"]
    # Failed harmless operations must also be checked; account data must not persist.
    operations.append(
        dict(operations[0], name="labels-op", operationType="setLabels", user="private-account")
    )
    cloud.change(operations=operations)
    cloud.run("close", "run-a")
    saved = cloud.read()["sessions"][0]
    assert saved["status"] == "closed"
    assert abs(Decimal(saved["settled_cost_usd"]) - Decimal(expected)) < Decimal("0.000002")
    end = "2026-11-13T10:38:35.546-08:00" if with_delete else "2026-11-13T10:37:59.733-08:00"
    assert saved["runtime_evidence"]["stop"] == saved["runtime_evidence"]["deleted"] == end
    assert saved["runtime_evidence"]["start"] == LIVE_GUEST_SHUTDOWNS[1]["last_start"]
    assert saved["settlement_evidence"] == expected_evidence
    assert len(saved["settlement_inventory"]) == 2
    assert all(
        not read["instances"] and not read["disks"] for read in saved["settlement_inventory"]
    )
    assert all("user" not in op for op in saved["settlement_operations"])


def test_ad2_completed_describe_path_unchanged(cloud, monkeypatch):
    fixture = LIVE_GUEST_SHUTDOWNS[0]
    guest_shutdown_history(cloud, fixture)
    cloud.change(
        instances=[
            dict(
                name="run-a",
                id="1",
                zone="europe-west4-a",
                status="TERMINATED",
                creationTimestamp=fixture["instance_created_at"],
                lastStartTimestamp=fixture["last_start"],
                lastStopTimestamp=fixture["runtime_evidence"]["stop"],
            )
        ]
    )
    close_in_process(cloud, monkeypatch, frozen=datetime.fromisoformat(fixture["closed_at"]))
    saved = cloud.read()["sessions"][0]
    assert saved["runtime_evidence"] == fixture["runtime_evidence"]
    assert saved["settlement_evidence"] == (
        "instances.describe lastStartTimestamp/lastStopTimestamp; "
        "boot disk through verified deletion (conservative)"
    )
    assert abs(Decimal(saved["settled_cost_usd"]) - Decimal("2.505759")) < Decimal("0.01")
    assert sum(call[:3] == ["compute", "instances", "describe"] for call in cloud.calls()) == 1


@pytest.mark.parametrize("initial_status", ["STOPPING", "TERMINATED"])
def test_close_waits_for_durable_stop_timestamps_before_delete(cloud, monkeypatch, initial_status):
    guest_shutdown_history(cloud)
    fixture = LIVE_GUEST_SHUTDOWNS[1]
    stop = "2026-11-13T18:38:55Z"
    cloud.change(
        instances=[
            dict(
                name="run-a",
                id="1",
                zone="europe-west4-a",
                status=initial_status,
                lastStartTimestamp=fixture["last_start"],
                creationTimestamp=fixture["instance_created_at"],
            )
        ],
        describe_updates=[{}, {}, dict(status="TERMINATED", lastStopTimestamp=stop)],
        require_runtime_evidence=stop,
    )
    elapsed = [0.0]
    close_in_process(cloud, monkeypatch, elapsed=elapsed)
    saved = cloud.read()["sessions"][0]
    assert saved["runtime_evidence"]["stop"] == stop
    assert saved["runtime_evidence"]["source"].startswith("instances.describe")
    assert saved["status"] == "closed"
    assert 0 < elapsed[0] <= 180
    calls = cloud.calls()
    assert not any(call[:3] == ["compute", "instances", "stop"] for call in calls)
    assert sum(call[:3] == ["compute", "instances", "describe"] for call in calls) == 3


@pytest.mark.parametrize("timeout", [0, 12, 180])
def test_close_runtime_wait_expires_then_deletes_and_recovers_operations(
    cloud, monkeypatch, timeout
):
    guest_shutdown_history(cloud)
    fixture = LIVE_GUEST_SHUTDOWNS[1]
    cloud.change(
        instances=[
            dict(
                name="run-a",
                id="1",
                zone="europe-west4-a",
                status="STOPPING",
                lastStartTimestamp=fixture["last_start"],
                creationTimestamp=fixture["instance_created_at"],
            )
        ]
    )
    elapsed = [0.0]
    extra = () if timeout == 180 else ("--runtime-timeout", str(timeout))
    close_in_process(cloud, monkeypatch, elapsed=elapsed, extra=extra)
    saved = cloud.read()["sessions"][0]
    assert elapsed[0] == timeout
    assert saved["settled_cost_usd"] == "3.749472"
    assert any(call[:3] == ["compute", "instances", "delete"] for call in cloud.calls())


@pytest.mark.parametrize("with_delete", [False, True])
@pytest.mark.parametrize(
    "fault",
    [
        "two-power-offs",
        "start",
        "resume",
        "reset",
        "automatic-restart",
        "failed-guest",
        "failed-insert",
        "failed-labels",
        "pending-guest",
        "duplicate-insert",
        "missing-insert",
        "missing-start",
        "missing-creation",
        "delete-issued-before-power-off",
        "wrong-target",
        "reversed-times",
        "unknown",
        "extra-stop",
        "disk-present",
        "instance-present",
    ],
)
def test_guest_shutdown_ambiguous_history_retains_reservation(cloud, with_delete, fault):
    operations = guest_shutdown_history(cloud)
    if not with_delete:
        operations = [op for op in operations if op["operationType"] != "delete"]
    guest = next(
        op for op in operations if op["operationType"] == "compute.instances.guestTerminate"
    )
    insert = next(op for op in operations if op["operationType"] == "insert")
    if fault == "two-power-offs":
        operations.append(dict(guest, name="second-power-off"))
    elif fault in {"start", "resume", "reset", "automatic-restart", "unknown", "extra-stop"}:
        kind = {
            "automatic-restart": "compute.instances.automaticRestart",
            "unknown": "setMetadata",
            "extra-stop": "stop",
        }.get(fault, fault)
        operations.append(dict(guest, name="extra-op", operationType=kind))
    elif fault.startswith("failed-"):
        op = guest if fault == "failed-guest" else insert
        if fault == "failed-labels":
            op = dict(insert, name="labels-op", operationType="setLabels")
            operations.append(op)
        op["error"] = {"errors": [{"code": "FAILED"}]}
    elif fault == "pending-guest":
        guest["status"] = "RUNNING"
    elif fault == "duplicate-insert":
        operations.append(dict(insert, name="second-insert"))
    elif fault == "missing-insert":
        operations.remove(insert)
    elif fault in {"missing-start", "missing-creation"}:
        ledger = cloud.read()
        ledger["sessions"][0].pop(
            "last_start" if fault == "missing-start" else "instance_created_at"
        )
        cloud.ledger.write_text(json.dumps(ledger))
    elif fault == "delete-issued-before-power-off":
        delete = next((op for op in operations if op["operationType"] == "delete"), None)
        if delete is None:
            operations.append(dict(guest, name="bad-delete", operationType="delete"))
            delete = operations[-1]
        delete["insertTime"] = "2026-11-13T18:37:58Z"
    elif fault == "wrong-target":
        guest["targetLink"] += "-foreign"
    elif fault == "reversed-times":
        guest["startTime"] = "2026-11-13T18:38:10Z"
    elif fault == "disk-present":
        cloud.change(
            disks=[dict(name=cloud.read()["sessions"][0]["boot_disk"], zone="europe-west4-a")]
        )
    elif fault == "instance-present":
        cloud.change(
            instances=[dict(name="run-a", id="1", zone="europe-west4-a", status="STAGING")],
            delete_noop=True,
        )
    cloud.change(operations=operations)
    result = cloud.run("close", "run-a", ok=False)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    saved = cloud.read()["sessions"][0]
    assert saved["status"] == "needs_settlement" and saved["settled_cost_usd"] is None
    assert Decimal(json.loads(cloud.run("report").stdout)["committed_usd"]) == Decimal(
        saved["reservation_usd"]
    )


@pytest.mark.parametrize(
    "option,value",
    [
        ("--runtime-timeout", "-1"),
        ("--runtime-timeout", "nan"),
        ("--runtime-timeout", "inf"),
        ("--poll-interval", "0"),
        ("--poll-interval", "61"),
        ("--poll-interval", "nan"),
    ],
)
def test_close_invalid_wait_budget_refuses_before_mutations(cloud, option, value):
    cloud.setup()
    cloud.launch()
    before = len(cloud.calls())
    assert cloud.run("close", "run-a", option, value, ok=False).returncode == 2
    assert not any(
        call[:3] in (["compute", "instances", "stop"], ["compute", "instances", "delete"])
        for call in cloud.calls()[before:]
    )


@pytest.mark.parametrize(
    "fault", ["changed-id", "changed-start", "delete-failed", "read-timeout", "disappeared"]
)
def test_close_runtime_poll_failures_preserve_money_and_evidence(cloud, monkeypatch, fault):
    module = runpy.run_path(str(GUARD))
    guest_shutdown_history(cloud)
    fixture = LIVE_GUEST_SHUTDOWNS[1]
    stop = "2026-11-13T18:38:55Z"
    cloud.change(
        instances=[
            dict(
                name="run-a",
                id="1",
                zone="europe-west4-a",
                status="STOPPING",
                lastStartTimestamp=fixture["last_start"],
                creationTimestamp=fixture["instance_created_at"],
            )
        ],
        describe_updates=[{}, dict(status="TERMINATED", lastStopTimestamp=stop)],
    )
    if fault == "changed-id":
        cloud.change(
            describe_updates=[{}, dict(id="foreign", status="TERMINATED", lastStopTimestamp=stop)]
        )
    elif fault == "changed-start":
        cloud.change(
            describe_updates=[
                {},
                dict(
                    lastStartTimestamp="2026-11-13T18:00:00Z",
                    status="TERMINATED",
                    lastStopTimestamp=stop,
                ),
            ]
        )
    elif fault == "delete-failed":
        cloud.change(delete_error=True, require_runtime_evidence=stop)
    elif fault in {"read-timeout", "disappeared"}:
        real_cloud = module["Guard"].cloud
        describe_reads = []
        elapsed = [0.0]

        def read(guard, *args, **kwargs):
            if args[:3] == ("compute", "instances", "describe"):
                describe_reads.append(kwargs.get("timeout", 300))
                if len(describe_reads) == 2:
                    if fault == "read-timeout":
                        elapsed[0] += kwargs["timeout"]
                        raise subprocess.TimeoutExpired("gcloud", kwargs["timeout"])
                    cloud.change(instances=[], disks=[])
                    return None
            return real_cloud(guard, *args, **kwargs)

        monkeypatch.setattr(module["Guard"], "cloud", read)
        close_in_process(
            cloud,
            monkeypatch,
            elapsed=elapsed,
            extra=("--runtime-timeout", "12", "--poll-interval", "5"),
            module=module,
        )
        assert len(describe_reads) == 2 and describe_reads[-1] <= 7
        assert cloud.read()["sessions"][0]["settled_cost_usd"] == "3.749472"
        return
    with pytest.raises((module["GuardError"], subprocess.CalledProcessError)):
        close_in_process(cloud, monkeypatch, elapsed=[0.0], module=module)
    saved = cloud.read()["sessions"][0]
    assert saved["settled_cost_usd"] is None and saved["status"] == "needs_settlement"
    if fault == "changed-id":
        assert not any(call[:3] == ["compute", "instances", "delete"] for call in cloud.calls())
    elif fault == "changed-start":
        assert saved["runtime_uncertain"]
    elif fault == "delete-failed":
        assert saved["runtime_evidence"]["stop"] == stop


def test_close_dry_run_does_not_wait_for_guest_stop(cloud):
    guest_shutdown_history(cloud)
    cloud.change(instances=[dict(name="run-a", id="1", zone="europe-west4-a", status="STOPPING")])
    before = cloud.ledger.read_bytes()
    # The CLI dry-run flag is normally handled by main; use its real subprocess path.
    result = cloud.run("close", "run-a", "--dry-run")
    assert "delete" in result.stdout
    assert cloud.ledger.read_bytes() == before


def test_close_retries_timed_out_runtime_read_until_overall_deadline(cloud, monkeypatch):
    guest_shutdown_history(cloud)
    fixture = LIVE_GUEST_SHUTDOWNS[1]
    stop = "2026-11-13T18:38:55Z"
    cloud.change(
        instances=[
            dict(
                name="run-a",
                id="1",
                zone="europe-west4-a",
                status="STOPPING",
                lastStartTimestamp=fixture["last_start"],
                creationTimestamp=fixture["instance_created_at"],
            )
        ],
        describe_updates=[{}, dict(status="TERMINATED", lastStopTimestamp=stop)],
        require_runtime_evidence=stop,
    )
    module = runpy.run_path(str(GUARD))
    real_cloud = module["Guard"].cloud
    elapsed = [0.0]
    reads = []

    def read(guard, *args, **kwargs):
        if args[:3] == ("compute", "instances", "describe"):
            reads.append(kwargs.get("timeout", 300))
            if len(reads) == 2:
                elapsed[0] += kwargs["timeout"]
                raise subprocess.TimeoutExpired("gcloud", kwargs["timeout"])
        return real_cloud(guard, *args, **kwargs)

    monkeypatch.setattr(module["Guard"], "cloud", read)
    close_in_process(cloud, monkeypatch, elapsed=elapsed, module=module)
    saved = cloud.read()["sessions"][0]
    assert len(reads) == 3 and reads[1] == 60
    assert elapsed[0] == 70
    assert saved["runtime_evidence"]["source"].startswith("instances.describe")
    assert saved["status"] == "closed"


def test_close_retains_reservation_after_contradictory_intermediate_start(cloud, monkeypatch):
    guest_shutdown_history(cloud)
    fixture = LIVE_GUEST_SHUTDOWNS[1]
    cloud.change(
        instances=[
            dict(
                name="run-a",
                id="1",
                zone="europe-west4-a",
                status="STOPPING",
                lastStartTimestamp=fixture["last_start"],
                creationTimestamp=fixture["instance_created_at"],
            )
        ],
        describe_updates=[
            dict(lastStartTimestamp="2026-11-13T18:00:00Z"),
            dict(
                status="TERMINATED",
                lastStartTimestamp=fixture["last_start"],
                lastStopTimestamp="2026-11-13T18:38:55Z",
            ),
        ],
    )
    module = runpy.run_path(str(GUARD))
    with pytest.raises(module["GuardError"], match="reservation retained"):
        close_in_process(cloud, monkeypatch, elapsed=[0.0], module=module)
    saved = cloud.read()["sessions"][0]
    assert saved["runtime_uncertain"]
    assert saved["settled_cost_usd"] is None and saved["status"] == "needs_settlement"
    assert any(call[:3] == ["compute", "instances", "delete"] for call in cloud.calls())


@pytest.mark.parametrize("model", ["SPOT", "STANDARD", "FLEX_START"])
def test_close_retains_reservation_when_terminal_stop_predates_guest_event(
    cloud, monkeypatch, model
):
    operations = guest_shutdown_history(cloud, model=model)
    fixture = LIVE_GUEST_SHUTDOWNS[1]
    cloud.change(
        operations=[op for op in operations if op["operationType"] != "delete"],
        instances=[
            dict(
                name="run-a",
                id="1",
                zone="europe-west4-a",
                status="STOPPING",
                lastStartTimestamp=fixture["last_start"],
                creationTimestamp=fixture["instance_created_at"],
                lastStopTimestamp="2026-11-13T18:00:00Z",
            )
        ],
        describe_updates=[{}, dict(status="TERMINATED")],
    )
    before = len(cloud.calls())
    module = runpy.run_path(str(GUARD))
    with pytest.raises(module["GuardError"], match="reservation retained"):
        close_in_process(cloud, monkeypatch, elapsed=[0.0], module=module)
    saved = cloud.read()["sessions"][0]
    assert saved["runtime_uncertain"]
    assert saved["settled_cost_usd"] is None and saved["status"] == "needs_settlement"
    assert Decimal(json.loads(cloud.run("report").stdout)["committed_usd"]) == Decimal(
        saved["reservation_usd"]
    )
    calls = [call[:3] for call in cloud.calls()[before:]]
    assert calls.count(["compute", "operations", "list"]) == 2
    assert calls.index(["compute", "operations", "list"]) < calls.index(
        ["compute", "instances", "describe"]
    )
    assert ["compute", "instances", "delete"] in calls


@pytest.mark.parametrize("model", ["SPOT", "STANDARD", "FLEX_START"])
def test_close_manual_settlement_does_not_fetch_guest_operations(cloud, model):
    guest_shutdown_history(cloud, model=model)
    fixture = LIVE_GUEST_SHUTDOWNS[1]
    cloud.change(
        instances=[
            dict(
                name="run-a",
                id="1",
                zone="europe-west4-a",
                status="TERMINATED",
                lastStartTimestamp=fixture["last_start"],
                creationTimestamp=fixture["instance_created_at"],
                lastStopTimestamp="2026-11-13T18:00:00Z",
            )
        ],
        operations_error=True,
    )
    before = len(cloud.calls())
    cloud.run("close", "run-a", "--settled-usd", "4", "--evidence", "saved billing export")
    saved = cloud.read()["sessions"][0]
    assert saved["status"] == "closed" and saved["settled_cost_usd"] == "4.000000"
    assert saved["settlement_evidence"] == "saved billing export"
    assert not any(call[:3] == ["compute", "operations", "list"] for call in cloud.calls()[before:])


def test_guest_shutdown_malformed_timestamp_is_a_clean_refusal(cloud):
    operations = guest_shutdown_history(cloud)
    next(op for op in operations if op["operationType"] == "compute.instances.guestTerminate")[
        "endTime"
    ] = 5
    cloud.change(operations=operations)
    result = cloud.run("close", "run-a", ok=False)
    assert result.returncode == 2 and "Traceback" not in result.stderr
    assert cloud.read()["sessions"][0]["settled_cost_usd"] is None
