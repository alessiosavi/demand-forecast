"""Weighted cloud guest with real saved forecasts and fev, using offline storage."""

import hashlib
import shutil
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("fev")
np = pytest.importorskip("numpy")
yaml = pytest.importorskip("yaml")

from demand_forecast.bench.yardsticks import run_yardstick  # noqa: E402
from scripts.fev_ensemble import main as ensemble_main  # noqa: E402
from tests.test_bench.test_fev_ensemble import OffsetAdapter  # noqa: E402
from tests.test_bench.test_yardsticks import (  # noqa: E402
    offline_only as offline_only,
)
from tests.test_bench.test_yardsticks import (  # noqa: E402
    scenario as scenario,
)
from tests.test_gcp_jobs.test_ensemble_job import (  # noqa: E402
    ROOT,
    URIS,
    load,
    rehash,
    sha,
    synthetic_member,
    write,
)
from tests.test_gcp_jobs.test_ensemble_job import (  # noqa: E402
    jobs as jobs,
)
from tests.test_gcp_jobs.test_ensemble_job import (  # noqa: E402
    members as members,
)


@pytest.mark.parametrize("processing", ["raw", "published", "both"])
@pytest.mark.parametrize("reference_first", [False, True])
def test_weighted_mixed_guest_real_fev(
    members, scenario, tmp_path, monkeypatch, processing, reference_first
):
    guest, config, root, scratch, cloud = members
    tasks, options, _ = scenario
    uris = [*URIS, "gs://offline/score/seed33"]
    reference_index = 0 if reference_first else 2
    weight_text = [".5", ".25", ".25"] if reference_first else [".25", ".25", ".5"]
    modes = ["raw", "published"] if processing == "both" else [processing]
    config.update(member_uris=uris, member_weights=weight_text, modes=modes)
    config["name"] += "_" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]
    (cloud / "tasks.yaml").write_text(yaml.safe_dump({"tasks": [task.to_dict() for task in tasks]}))
    config["tasks_sha256"] = sha(cloud / "tasks.yaml")
    sources = []
    for index, uri in enumerate(uris):
        directory = cloud / uri.removeprefix("gs://offline/")
        if not directory.exists():
            synthetic_member(directory, index)
        reference = index == reference_index
        folders = ["raw"] if reference else modes
        weights = tmp_path / f"mixed-weights-{index}"
        weights.mkdir()
        (weights / "config.json").write_text("{}")
        (weights / "model.safetensors").write_bytes(f"seed {index}".encode())
        for mode in ["raw", "published"]:
            shutil.rmtree(directory / mode)
        for folder in folders:
            run_yardstick(
                tasks,
                replace(
                    options,
                    model="timestral" if reference else "simons",
                    processing=None if reference else folder,
                    weights=str(weights),
                    model_name=f"seed{index}-{folder}",
                ),
                directory / folder,
                adapter_factory=lambda _, i=index: OffsetAdapter(i * 2),
            )
        inputs = load(directory / "inputs.json")
        inputs.update(tasks_sha256=config["tasks_sha256"], modes=folders)
        if reference:
            inputs.update(model="timestral", model_package="synthetic native reference")
        write(directory / "inputs.json", inputs)
        write(directory / "selected.json", [task.task_name for task in tasks])
        marker = load(directory / "completed.json")
        marker["inputs"] = inputs
        marker["modes"] = {
            folder: {
                **marker["modes"][folder],
                "requested": [task.task_name for task in tasks],
                "completed": [task.task_name for task in tasks],
            }
            for folder in folders
        }
        write(directory / "completed.json", marker)
        rehash(directory)
        sources.append(directory)
    original_run = subprocess.run
    calls = []

    def execute(args, **kwargs):
        if len(args) < 2 or not str(args[1]).endswith("fev_ensemble.py"):
            return original_run(args, **kwargs)
        calls.append(args)
        return SimpleNamespace(returncode=ensemble_main(args[2:]))

    monkeypatch.setattr(subprocess, "run", execute)
    assert guest["preflight"](config, root) == 10
    (root / "environment.txt").write_text("real fev, offline test storage\n")
    assert guest["average"](config, root, scratch, ROOT, sys.executable) == 0
    assert guest["preflight"](config, root) == 0
    marker = load(root / "completed.json")
    parity_count = 5 if processing == "both" else 3
    assert len(marker["parity"]) == parity_count
    assert ["--allow-identical-members" in args for args in calls] == [True] * parity_count + [
        False
    ] * len(modes)
    for args in calls[:parity_count]:
        assert "--member-weight" not in args
    for mode in modes:
        record = load(root / mode / tasks[0].task_name / "run.json")
        if mode == "published":
            assert record["configuration"]["options"]["processing"] == "published"
        for index in range(tasks[0].num_windows):
            name = tasks[0].task_name + f"/window_{index}.npz"
            contributions = []
            for member_index, (source, weight) in enumerate(zip(sources, weight_text, strict=True)):
                folder = "raw" if member_index == reference_index else mode
                with np.load(source / folder / name) as saved:
                    contributions.append(float(weight) * saved["arrays"].astype(np.float64))
            expected = contributions[0] + contributions[1] + contributions[2]
            with np.load(root / mode / name) as actual:
                np.testing.assert_array_equal(actual["arrays"], expected)
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert guest["preflight"](config, root) == 0
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
