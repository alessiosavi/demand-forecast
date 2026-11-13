"""Real operator CLI against a fake gcloud; never uses an account or cloud API."""

import json
import os
import runpy
import subprocess
import sys
from datetime import datetime, timezone
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
if args[:3] == ['compute', 'instances', 'create']:
    ledger = json.loads(Path(os.environ['LEDGER']).read_text())
    session = ledger['sessions'][-1]
    assert session['status'] == 'reserved' and session['settled_cost_usd'] is None
    assert Decimal(session['reservation_usd']) > 0
    time.sleep(state.get('create_delay', 0))
    if state.get('create_error'):
        state['create_failed'] = dict(instance=args[3], disk=session['boot_disk'])
        end(error=state['create_error'])
    if 'create_response' in state: end(state['create_response'])
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
    matches = [v for v in state.get('instances', []) if v['name'] == args[3]]
    if not matches: end(error='The resource was not found')
    end(matches[0])
if args[:3] == ['compute', 'instances', 'stop']:
    from datetime import datetime, timezone
    for v in state['instances']:
        if v['name'] == args[3]:
            if any(d['type'] == 'SCRATCH' for d in v['disks']) and opt('--discard-local-ssd') != 'true':
                end(error='Local SSD data must be explicitly discarded')
            v.update(status='TERMINATED', lastStopTimestamp=datetime.now(timezone.utc).isoformat())
    end({})
if args[:3] == ['compute', 'instances', 'delete']:
    state['instances'] = [v for v in state.get('instances', []) if v['name'] != args[3]]
    if not state.get('orphan_disk'): state['disks'] = []
    end({})
if args[:3] == ['compute', 'disks', 'add-labels']:
    if state.get('disk_label_error'): end(error='label update failed')
    for d in state.get('disks', []):
        if d['name'] == args[3]: d['labels'] = dict(x.split('=', 1) for x in opt('--labels').split(','))
    end({})
if args[:3] == ['compute', 'disks', 'delete']:
    state['disks'] = [d for d in state.get('disks', []) if d['name'] != args[3]]
    end({})
if args[:3] == ['compute', 'operations', 'list']:
    assert opt('--zones') == 'europe-west4-a', args
    target = opt('--filter')
    assert target and target.startswith('targetId='), args
    if state.get('operations_error'): end(error='403 Permission denied')
    end([o for o in state.get('operations', []) if str(o.get('targetId')) == target.split('=', 1)[1]
         and o.get('zone', '').rsplit('/', 1)[-1] == opt('--zones')])
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
    cloud.change(create_delay=0.2)
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
    a = subprocess.Popen(base, env=cloud.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    base[base.index("first")] = "second"
    b = subprocess.Popen(base, env=cloud.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    a.communicate(timeout=20)
    b.communicate(timeout=20)
    assert sorted([a.returncode, b.returncode]) == [0, 2]
    assert len(cloud.read()["sessions"]) == 1


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


def test_close_uses_real_timestamps_and_checks_disk_removal(cloud):
    cloud.setup()
    cloud.change(start_age=1800, creation_age=3600)
    cloud.launch()
    state = json.loads(cloud.state.read_text())
    now = datetime.now(timezone.utc)
    state["instances"][0].update(status="TERMINATED", lastStopTimestamp=now.isoformat())
    cloud.change(**state, orphan_disk=True)
    assert cloud.run("close", "run-a", ok=False).returncode != 0
    assert cloud.read()["sessions"][0]["settled_cost_usd"] is None
    cloud.change(disks=[])
    cloud.run("close", "run-a")
    cost = Decimal(cloud.read()["sessions"][0]["settled_cost_usd"])
    assert Decimal("2.013") < cost < Decimal("2.02")


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
def test_returned_create_error_cancels_after_two_absence_reads(cloud, error):
    cloud.setup()
    cloud.change(create_error=error)
    assert cloud.launch(ok=False).returncode == 2
    session = cloud.read()["sessions"][0]
    assert session["status"] == "cancelled"
    assert Decimal(session["settled_cost_usd"]) == 0
    assert error in session["error"]
    reads = session["create_failure_inventory"]
    assert len(reads) == 2
    assert all(read["instances"] == read["disks"] == [] for read in reads)
    times = [datetime.fromisoformat(read["observed_at"].replace("Z", "+00:00")) for read in reads]
    assert (times[1] - times[0]).total_seconds() >= 3
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
    assert "operations" in session["settlement_evidence"]


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
def test_flex_wait_reaches_create_and_sets_subprocess_budget(cloud, monkeypatch, duration, seconds):
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
    assert budgets == [seconds + 300]
    create = next(c for c in cloud.calls() if c[:3] == ["compute", "instances", "create"])
    assert f"--request-valid-for-duration={duration}" in create
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
    cloud.change(create_delay=2)
    real_run = subprocess.run
    budgets = []

    def run(command, **kwargs):
        if command[:4] == ["gcloud", "compute", "instances", "create"]:
            budgets.append(kwargs["timeout"])
            kwargs["timeout"] = (
                0.2  # Exercise a real subprocess timeout without waiting 390 seconds.
            )
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
    assert budgets == [390]
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
