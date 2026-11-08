"""Runner orchestration tests (fake CommandRunner; no network, no Kaggle CLI).

The fake runner records argv and returns scripted ``CompletedProcess``
results, so the whole stage sequence is exercised hermetically. The score-stage
and mlcspp tests below need a parquet engine and skip (never fail) without
pyarrow, keeping the rest of the module runnable in the dev-only CI lap.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from demand_forecast.config import DatasetConfig
from demand_forecast.kaggle import runner
from demand_forecast.kaggle.runner import (
    STAGES,
    CommandRunner,
    RunnerConfig,
    Stage,
    fetch_outputs,
    plan_stages,
    run_stages,
    wait_for_kernel,
)
from demand_forecast.kaggle.settings import build_settings, load_template, write_settings

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE = _REPO_ROOT / "configs/kaggle-v2-l3-finetune.yaml"
_SLUG = "store-sales-time-series-forecasting"
_DATASET_ID = "alessiosavi/kb-store-sales"
_KERNEL_ID = "alessiosavi/kb-store-sales-run"
_ASSETS_DATASET = "alessiosavi/kb-assets"
_REPO_SHA = "0123456789abcdef0123456789abcdef01234567"

Responder = Callable[[list[str]], subprocess.CompletedProcess[str] | None]


def _config(tmp_path: Path, *, slug: str = _SLUG) -> RunnerConfig:
    return RunnerConfig(
        slug=slug,
        evidence_root=tmp_path / "runs" / "store-sales",
        dataset_id=_DATASET_ID,
        kernel_id=_KERNEL_ID,
        assets_dataset=_ASSETS_DATASET,
        repo_sha=_REPO_SHA,
    )


def _recording_runner(calls: list[list[str]], responder: Responder | None = None) -> CommandRunner:
    """A CommandRunner that records argv and returns scripted results."""

    def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        if responder is not None:
            result = responder(argv)
            if result is not None:
                return result
        return subprocess.CompletedProcess(argv, 0, "", "")

    return run


def _status_words(*words: str) -> Responder:
    """Answer kernel-status calls with the next word (the last word repeats)."""
    queue = list(words)

    def responder(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
        if argv[:3] != ["kaggle", "kernels", "status"]:
            return None
        word = queue.pop(0) if len(queue) > 1 else queue[0]
        output = f'{_KERNEL_ID} has status "KernelWorkerStatus.{word}"'
        return subprocess.CompletedProcess(argv, 0, output, "")

    return responder


def _combine(*responders: Responder) -> Responder:
    def responder(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
        for candidate in responders:
            result = candidate(argv)
            if result is not None:
                return result
        return None

    return responder


def _emit_output_tree() -> Responder:
    """Mimic `kaggle kernels output` by writing a minimal fetched tree."""

    def responder(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
        if argv[:3] != ["kaggle", "kernels", "output"]:
            return None
        dest = Path(argv[argv.index("-p") + 1])
        root = dest / "kb-out"
        (root / "finetune").mkdir(parents=True, exist_ok=True)
        (root / "workload.json").write_text(
            json.dumps({"modes": ["zero_shot", "finetune"], "windows": ["mirror", "official"]}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    return responder


# --------------------------------------------------------------------------- #
# Stage planning and the all-stages fake run
# --------------------------------------------------------------------------- #


def test_plan_stages_lists_expected_commands(tmp_path):
    config = _config(tmp_path)
    plan = plan_stages(config)

    assert len(plan) == len(STAGES) == 7
    assert plan[0] == [
        "kaggle",
        "competitions",
        "download",
        "-c",
        _SLUG,
        "-p",
        str(config.raw_dir),
        "--unzip",
    ]
    assert plan[1] == []  # prepare calls store_sales.prepare directly
    assert plan[2][:3] == ["kaggle", "datasets", "version"]
    assert plan[2][3:5] == ["-p", str(config.upload_dir)]
    assert plan[2][plan[2].index("-m") + 1]  # version always carries a message
    assert plan[3] == [
        "kaggle",
        "kernels",
        "push",
        "-p",
        str(config.kernel_dir),
        "--accelerator",
        "NvidiaTeslaT4",
        "-t",
        "21600",
    ]
    assert plan[4] == [
        "kaggle",
        "kernels",
        "output",
        "-p",
        str(config.fetched_dir),
        _KERNEL_ID,
    ]
    assert plan[5] == []  # score runs the scorer and assembly locally
    assert plan[6] == [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        _SLUG,
        "-f",
        str(config.submission_path),
        "-m",
        "demand-forecast L3",
    ]
    assert not config.evidence_root.exists()  # planning has no side effects


def test_run_stages_records_commands_with_fake_runner(tmp_path):
    config = _config(tmp_path)
    config.prepared_dir.mkdir(parents=True)  # prepare ran before upload
    calls: list[list[str]] = []
    responder = _combine(_emit_output_tree(), _status_words("COMPLETE"))

    run_stages(
        config,
        _recording_runner(calls, responder),
        stages=["download", "upload", "kernel", "fetch"],
    )

    prefixes = [argv[:3] for argv in calls]
    assert prefixes[0] == ["kaggle", "competitions", "download"]
    assert ["kaggle", "datasets", "status"] in prefixes
    version = next(argv for argv in calls if argv[:3] == ["kaggle", "datasets", "version"])
    assert version[3:5] == ["-p", str(config.upload_dir)]
    push = next(argv for argv in calls if argv[:3] == ["kaggle", "kernels", "push"])
    assert push[3:5] == ["-p", str(config.kernel_dir)]
    assert prefixes[-1] == ["kaggle", "kernels", "output"]
    for earlier, later in (
        (["kaggle", "competitions", "download"], ["kaggle", "datasets", "version"]),
        (["kaggle", "datasets", "version"], ["kaggle", "kernels", "push"]),
        (["kaggle", "kernels", "push"], ["kaggle", "kernels", "output"]),
    ):
        assert prefixes.index(earlier) < prefixes.index(later)
    # The packaging stages wrote their artifacts, with the pinned mounts and SHA.
    assert json.loads(
        (config.upload_dir / "dataset-metadata.json").read_text(encoding="utf-8")
    ) == {"title": f"kb {_SLUG} prepared", "id": _DATASET_ID, "licenses": [{"name": "other"}]}
    metadata = json.loads((config.kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == _KERNEL_ID
    assert metadata["dataset_sources"] == [_DATASET_ID, _ASSETS_DATASET]
    driver = (config.kernel_dir / "main.py").read_text(encoding="utf-8")
    assert _REPO_SHA in driver
    assert "/kaggle/input/datasets/alessiosavi/kb-store-sales" in driver
    assert "/kaggle/input/datasets/alessiosavi/kb-assets/bundle-L3-first-try" in driver
    marker = config.fetched_dir / "kb-out" / "workload.json"
    assert marker.is_file()


def test_run_stages_rejects_unknown_stage(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="unknown stage"):
        run_stages(config, _recording_runner([]), stages=[cast(Stage, "bogus")])


# --------------------------------------------------------------------------- #
# Direct-call stages and dataset upload
# --------------------------------------------------------------------------- #


def test_prepare_stage_calls_store_sales_prepare(tmp_path, monkeypatch):
    config = _config(tmp_path)
    recorded: list[tuple[Path, Path]] = []

    def fake_prepare(raw_dir: Path, out_dir: Path) -> None:
        recorded.append((Path(raw_dir), Path(out_dir)))

    monkeypatch.setattr(runner, "prepare", fake_prepare)

    run_stages(config, _recording_runner([]), stages=["prepare"])

    assert recorded == [(config.raw_dir, config.prepared_dir)]


def test_upload_versions_existing_dataset(tmp_path):
    config = _config(tmp_path)
    config.prepared_dir.mkdir(parents=True)
    calls: list[list[str]] = []

    run_stages(config, _recording_runner(calls), stages=["upload"])

    assert calls[0] == ["kaggle", "datasets", "status", _DATASET_ID]
    assert calls[1][:3] == ["kaggle", "datasets", "version"]
    assert calls[1][3:5] == ["-p", str(config.upload_dir)]
    assert calls[1][calls[1].index("-m") + 1] == f"demand-forecast L3 {_SLUG} prepared"


def test_upload_creates_missing_dataset(tmp_path):
    config = _config(tmp_path)
    config.prepared_dir.mkdir(parents=True)
    calls: list[list[str]] = []

    def status_responder(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
        if argv[:3] == ["kaggle", "datasets", "status"]:
            return subprocess.CompletedProcess(argv, 1, "", "403 Client Error: Forbidden")
        return None

    run_stages(config, _recording_runner(calls, status_responder), stages=["upload"])

    assert calls[-1] == [
        "kaggle",
        "datasets",
        "create",
        "-p",
        str(config.upload_dir),
        "--dir-mode",
        "zip",
    ]
    assert "-m" not in calls[-1]  # create takes no message


_UPLOAD_FIXTURE: dict[str, bytes] = {
    "official/long.parquet": b"official long bytes",
    "official/sample_submission.csv": b"id,sales\n1,0.0\n",
    "mirror/prepare.json": b'{"mirror_origin": "2017-08-03"}',
    "mirror/static.parquet": b"mirror static bytes",
    "mirror/truth.parquet": b"mirror truth bytes",
}


def _write_minimal_prepared_tree(config: RunnerConfig) -> None:
    """A prepared-tree fixture of plain bytes: no parquet engine required."""
    for relative, content in _UPLOAD_FIXTURE.items():
        path = config.prepared_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def test_upload_stages_prepared_tree_as_single_archive(tmp_path):
    config = _config(tmp_path)
    _write_minimal_prepared_tree(config)
    calls: list[list[str]] = []

    run_stages(config, _recording_runner(calls), stages=["upload"])

    assert sorted(path.name for path in config.upload_dir.iterdir()) == [
        "dataset-metadata.json",
        f"{_SLUG}-prepared.zip",
    ]
    with zipfile.ZipFile(config.upload_dir / f"{_SLUG}-prepared.zip") as bundle:
        # The entries are the prepared tree's relative paths: the dataset layout.
        assert set(bundle.namelist()) == set(_UPLOAD_FIXTURE)
        assert bundle.read("mirror/truth.parquet") == b"mirror truth bytes"


def test_upload_rerun_rebuilds_staging_without_stale_entries(tmp_path):
    config = _config(tmp_path)
    _write_minimal_prepared_tree(config)
    calls: list[list[str]] = []

    run_stages(config, _recording_runner(calls), stages=["upload"])
    stale_file = config.upload_dir / "stale.txt"
    stale_file.write_text("stale", encoding="utf-8")
    stale_dir = config.upload_dir / "stale-dir"
    stale_dir.mkdir()
    (stale_dir / "junk.parquet").write_bytes(b"junk")

    run_stages(config, _recording_runner(calls), stages=["upload"])

    assert not stale_file.exists()
    assert not stale_dir.exists()
    assert sorted(path.name for path in config.upload_dir.iterdir()) == [
        "dataset-metadata.json",
        f"{_SLUG}-prepared.zip",
    ]
    with zipfile.ZipFile(config.upload_dir / f"{_SLUG}-prepared.zip") as bundle:
        assert set(bundle.namelist()) == set(_UPLOAD_FIXTURE)


def test_upload_without_prepared_tree_raises(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(RuntimeError, match="run 'prepare' first"):
        run_stages(config, _recording_runner([]), stages=["upload"])


# --------------------------------------------------------------------------- #
# Kernel polling and failure surfacing
# --------------------------------------------------------------------------- #


def test_wait_for_kernel_polls_and_completes(tmp_path):
    config = _config(tmp_path)
    calls: list[list[str]] = []
    sleeps: list[float] = []
    responder = _status_words("NEW_SCRIPT", "RUNNING", "COMPLETE")

    wait_for_kernel(
        config, _recording_runner(calls, responder), sleep=sleeps.append, interval_s=17.0
    )

    assert [argv[:3] for argv in calls] == [["kaggle", "kernels", "status"]] * 3
    assert [argv[3] for argv in calls] == [_KERNEL_ID] * 3
    assert sleeps == [17.0, 17.0]


def test_wait_for_kernel_graces_first_not_found_polls(tmp_path):
    config = _config(tmp_path)
    calls: list[list[str]] = []
    sleeps: list[float] = []
    misses = {"remaining": 2}

    def responder(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
        if argv[:3] != ["kaggle", "kernels", "status"]:
            return None
        if misses["remaining"] > 0:
            misses["remaining"] -= 1
            return subprocess.CompletedProcess(argv, 1, "", "404 Not Found: GetKernelSessionStatus")
        return subprocess.CompletedProcess(
            argv, 0, f'{_KERNEL_ID} has status "KernelWorkerStatus.COMPLETE"', ""
        )

    wait_for_kernel(
        config,
        _recording_runner(calls, responder),
        sleep=sleeps.append,
        interval_s=17.0,
        not_found_grace=5,
    )

    assert len(calls) == 3  # two not-found polls inside the grace, then COMPLETE
    assert sleeps == [17.0, 17.0]


def test_wait_for_kernel_raises_when_not_found_grace_is_exceeded(tmp_path):
    config = _config(tmp_path)
    calls: list[list[str]] = []

    def responder(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
        if argv[:3] == ["kaggle", "kernels", "status"]:
            return subprocess.CompletedProcess(
                argv, 1, "", "404 Client Error: Not Found ... GetKernelSessionStatus"
            )
        return None

    with pytest.raises(RuntimeError, match="404 Client Error"):
        wait_for_kernel(
            config,
            _recording_runner(calls, responder),
            sleep=lambda _seconds: None,
            not_found_grace=2,
        )

    assert len(calls) == 3  # two polls inside the grace, the third exceeds it


def test_wait_for_kernel_raises_immediately_on_other_status_cli_failures(tmp_path):
    config = _config(tmp_path)
    calls: list[list[str]] = []
    sleeps: list[float] = []

    def responder(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
        if argv[:3] == ["kaggle", "kernels", "status"]:
            return subprocess.CompletedProcess(argv, 1, "", "403 - Permission 'kernels.get' denied")
        return None

    with pytest.raises(RuntimeError, match="kernels.get"):
        wait_for_kernel(config, _recording_runner(calls, responder), sleep=sleeps.append)

    assert len(calls) == 1  # a non-not-found failure stays fatal on the first poll
    assert sleeps == []


@pytest.mark.parametrize("word", ["ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED"])
def test_kernel_stage_raises_on_failed_status(tmp_path, word):
    config = _config(tmp_path)
    calls: list[list[str]] = []

    with pytest.raises(RuntimeError, match=word):
        run_stages(config, _recording_runner(calls, _status_words(word)), stages=["kernel"])

    # The failure ends the poll loop immediately; nothing is fetched.
    assert [argv[:3] for argv in calls] == [
        ["kaggle", "kernels", "push"],
        ["kaggle", "kernels", "status"],
    ]


def test_kernel_stage_raises_on_unparseable_status(tmp_path):
    config = _config(tmp_path)

    def responder(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
        if argv[:3] == ["kaggle", "kernels", "status"]:
            return subprocess.CompletedProcess(argv, 0, "mystery output", "")
        return None

    with pytest.raises(ValueError, match="no kernel status"):
        run_stages(config, _recording_runner([], responder), stages=["kernel"])


def test_kernel_stage_raises_on_status_cli_failure(tmp_path):
    config = _config(tmp_path)

    def responder(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
        if argv[:3] == ["kaggle", "kernels", "status"]:
            return subprocess.CompletedProcess(argv, 1, "", "403 - Permission 'kernels.get' denied")
        return None

    with pytest.raises(RuntimeError, match="kernels.get"):
        run_stages(config, _recording_runner([], responder), stages=["kernel"])


def test_kernel_stage_raises_on_rejected_push_without_polling(tmp_path):
    config = _config(tmp_path)
    calls: list[list[str]] = []
    rejection = "Kernel push error: Maximum batch GPU session count of 2 reached."

    def responder(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
        if argv[:3] == ["kaggle", "kernels", "push"]:
            # Live CLI behavior: the rejection goes to stdout with exit code 0.
            return subprocess.CompletedProcess(argv, 0, rejection, "")
        return None

    with pytest.raises(RuntimeError, match="Maximum batch GPU session count of 2 reached"):
        run_stages(config, _recording_runner(calls, responder), stages=["kernel"])

    assert [argv[:3] for argv in calls] == [["kaggle", "kernels", "push"]]


# --------------------------------------------------------------------------- #
# Fetch: marker discipline
# --------------------------------------------------------------------------- #


def test_fetch_without_output_raises(tmp_path):
    config = _config(tmp_path)

    def log_only_responder(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
        if argv[:3] != ["kaggle", "kernels", "output"]:
            return None
        dest = Path(argv[argv.index("-p") + 1])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{_KERNEL_ID.split('/')[1]}.log").write_text("crashed", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "", "")

    with pytest.raises(RuntimeError, match="no kernel output"):
        fetch_outputs(config, _recording_runner([], log_only_responder))


def test_fetch_returns_workload_marker(tmp_path):
    config = _config(tmp_path)
    marker = fetch_outputs(config, _recording_runner([], _emit_output_tree()))

    assert marker == config.fetched_dir / "kb-out" / "workload.json"
    assert marker.is_file()


def test_fetch_raises_on_output_cli_failure(tmp_path):
    config = _config(tmp_path)

    def responder(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
        if argv[:3] == ["kaggle", "kernels", "output"]:
            return subprocess.CompletedProcess(argv, 1, "", "429 Too Many Requests")
        return None

    with pytest.raises(RuntimeError, match="429"):
        fetch_outputs(config, _recording_runner([], responder))


def test_fetch_replaces_stale_tree(tmp_path):
    config = _config(tmp_path)
    stale = config.fetched_dir / "kb-out" / "workload.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="no kernel output"):
        fetch_outputs(config, _recording_runner([]))  # fetches nothing

    assert not stale.exists()  # a stale marker cannot mask a missing output


# --------------------------------------------------------------------------- #
# Submit: validate, submit once, record
# --------------------------------------------------------------------------- #


def _write_submit_fixture(
    config: RunnerConfig, *, submission_rows: list[tuple[int, float]] | None = None
) -> None:
    official = config.prepared_dir / "official"
    official.mkdir(parents=True)
    ids = [3, 1, 2]
    pd.DataFrame({"id": ids, "sales": [0.0, 0.0, 0.0]}).to_csv(
        official / "sample_submission.csv", index=False
    )
    rows = submission_rows if submission_rows is not None else [(3, 1.0), (1, 2.0), (2, 3.0)]
    pd.DataFrame(rows, columns=["id", "sales"]).to_csv(config.submission_path, index=False)
    config.scores_path.write_text(json.dumps({"baseline": 0.5, "finetune": 0.3}), encoding="utf-8")


def test_submit_stage_submits_once_and_records_ref(tmp_path):
    config = _config(tmp_path)
    _write_submit_fixture(config)
    calls: list[list[str]] = []
    submit_stdout = "Submission ref: 4242\n2 submissions remaining today.\n"
    listing_stdout = "ref   fileName        date\n4242  submission.csv  2017-08-16\n"

    def responder(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
        if argv[:3] == ["kaggle", "competitions", "submit"]:
            return subprocess.CompletedProcess(argv, 0, submit_stdout, "")
        if argv[:3] == ["kaggle", "competitions", "submissions"]:
            return subprocess.CompletedProcess(argv, 0, listing_stdout, "")
        return None

    run_stages(config, _recording_runner(calls, responder), stages=["submit"])

    assert calls == [
        [
            "kaggle",
            "competitions",
            "submit",
            "-c",
            _SLUG,
            "-f",
            str(config.submission_path),
            "-m",
            "demand-forecast L3",
        ],
        ["kaggle", "competitions", "submissions", "-c", _SLUG],
    ]
    assert config.submissions_path.read_text(encoding="utf-8") == listing_stdout
    report = config.report_path.read_text(encoding="utf-8")
    assert "submission_ref: 4242" in report
    assert "mirror (non-official)" in report


def test_submit_stage_rejects_invalid_submission_before_cli(tmp_path):
    config = _config(tmp_path)
    _write_submit_fixture(config, submission_rows=[(1, 1.0), (3, 2.0), (2, 3.0)])  # swapped ids
    calls: list[list[str]] = []

    with pytest.raises(ValueError, match="order"):
        run_stages(config, _recording_runner(calls), stages=["submit"])

    assert calls == []  # a mismatched candidate never reaches the CLI


def test_submit_stage_requires_scores(tmp_path):
    config = _config(tmp_path)
    _write_submit_fixture(config)
    config.scores_path.unlink()
    calls: list[list[str]] = []

    with pytest.raises(RuntimeError, match="score"):
        run_stages(config, _recording_runner(calls), stages=["submit"])

    assert calls == []


def test_submit_stage_requires_submission(tmp_path):
    config = _config(tmp_path)
    _write_submit_fixture(config)
    config.submission_path.unlink()
    calls: list[list[str]] = []

    with pytest.raises(RuntimeError, match="score"):
        run_stages(config, _recording_runner(calls), stages=["submit"])

    assert calls == []


def test_submit_stage_raises_with_cli_message_and_never_retries(tmp_path):
    config = _config(tmp_path)
    _write_submit_fixture(config)
    calls: list[list[str]] = []

    def responder(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
        if argv[:3] == ["kaggle", "competitions", "submit"]:
            return subprocess.CompletedProcess(argv, 1, "", "used all available submissions")
        return None

    with pytest.raises(RuntimeError, match="all available submissions"):
        run_stages(config, _recording_runner(calls, responder), stages=["submit"])

    assert [argv[:3] for argv in calls] == [["kaggle", "competitions", "submit"]]


# --------------------------------------------------------------------------- #
# Score stage (needs a parquet engine: skipped, never failed, without pyarrow)
# --------------------------------------------------------------------------- #

_DAYS = pd.date_range("2017-08-01", periods=4, freq="D")
_MIRROR_ORIGIN = "2017-08-03"
_HISTORY = {(1, "AUTOMOTIVE"): [8.0, 10.0], (2, "BABY CARE"): [25.0, 30.0]}
_MIRROR_TRUTH = {(1, "AUTOMOTIVE"): [12.0, 13.0], (2, "BABY CARE"): [20.0, 21.0]}
_OFFICIAL_TEST_IDS = [
    {"id": 3, "date": "2017-08-16", "store_nbr": 1, "family": "AUTOMOTIVE"},
    {"id": 1, "date": "2017-08-16", "store_nbr": 2, "family": "AUTOMOTIVE"},
    {"id": 2, "date": "2017-08-17", "store_nbr": 1, "family": "BABY CARE"},
]
_OFFICIAL_POINTS = [
    {"series_id": "1|AUTOMOTIVE", "timestamp": "2017-08-16", "point": 5.0, "q0.5": 5.0},
    {"series_id": "2|AUTOMOTIVE", "timestamp": "2017-08-16", "point": 7.0, "q0.5": 7.0},
    {"series_id": "1|BABY CARE", "timestamp": "2017-08-17", "point": 9.0, "q0.5": 9.0},
]


def _dataset_config(directory: Path) -> DatasetConfig:
    """The long/static roles the store-sales templates consume."""
    return DatasetConfig(
        source="long",
        path=directory / "long.parquet",
        id_columns=["store_nbr", "family"],
        timestamp_column="date",
        target="sales",
        target_support="nonnegative",
        known_dynamic=["onpromotion", "is_national_holiday"],
        past_dynamic=["oil_price", "transactions"],
        static_path=directory / "static.parquet",
        static_categorical=["store_nbr", "family", "city", "state", "type", "cluster"],
        freq="D",
        missing_target="zero",
        duplicates="error",
    )


def _write_prepared_tree(config: RunnerConfig) -> None:
    mirror = config.prepared_dir / "mirror"
    official = config.prepared_dir / "official"
    mirror.mkdir(parents=True)
    official.mkdir(parents=True)
    long_rows = []
    for index, day in enumerate(_DAYS):
        for (store_nbr, family), values in _HISTORY.items():
            observed = index < 2
            long_rows.append(
                {
                    "store_nbr": store_nbr,
                    "family": family,
                    "date": day,
                    "sales": values[index] if observed else float("nan"),
                    "onpromotion": int(day.day % 2 == 0),
                    "is_national_holiday": 0,
                    "oil_price": 40.0 + index if observed else float("nan"),
                    "transactions": 100.0 * store_nbr if observed else float("nan"),
                }
            )
    pd.DataFrame(long_rows).to_parquet(mirror / "long.parquet", index=False)
    pd.DataFrame(
        {
            "store_nbr": [1, 2],
            "family": ["AUTOMOTIVE", "BABY CARE"],
            "city": ["Quito", "Guayaquil"],
            "state": ["Pichincha", "Guayas"],
            "type": ["A", "B"],
            "cluster": [3, 7],
        }
    ).to_parquet(mirror / "static.parquet", index=False)
    truth_rows = [
        {"store_nbr": store_nbr, "family": family, "date": _DAYS[2 + step], "sales": value}
        for (store_nbr, family), values in _MIRROR_TRUTH.items()
        for step, value in enumerate(values)
    ]
    pd.DataFrame(truth_rows).to_parquet(mirror / "truth.parquet", index=False)
    settings = build_settings(load_template(_TEMPLATE), data=_dataset_config(mirror), horizon=2)
    write_settings(settings, mirror / "settings-finetune.yaml")
    (mirror / "prepare.json").write_text(
        json.dumps({"mirror_origin": _MIRROR_ORIGIN}), encoding="utf-8"
    )
    test_ids = pd.DataFrame(_OFFICIAL_TEST_IDS)
    test_ids["date"] = pd.to_datetime(test_ids["date"])
    test_ids.to_parquet(official / "test_ids.parquet", index=False)
    pd.DataFrame({"id": test_ids["id"], "sales": 0.0}).to_csv(
        official / "sample_submission.csv", index=False
    )


def _mirror_points(values: dict[tuple[int, str], list[float]]) -> pd.DataFrame:
    rows = [
        {
            "series_id": f"{store_nbr}|{family}",
            "timestamp": _DAYS[2 + step],
            "point": value,
            "q0.5": value,
        }
        for (store_nbr, family), series_values in values.items()
        for step, value in enumerate(series_values)
    ]
    return pd.DataFrame(rows).iloc[[3, 0, 2, 1]].reset_index(drop=True)  # shuffled


def _write_fetched_tree(config: RunnerConfig) -> None:
    root = config.fetched_dir / "kb-out"
    (root / "zero_shot" / "mirror").mkdir(parents=True)
    (root / "finetune" / "mirror").mkdir(parents=True)
    (root / "finetune" / "official").mkdir(parents=True)
    (root / "workload.json").write_text(
        json.dumps({"modes": ["zero_shot", "finetune"], "windows": ["mirror", "official"]}),
        encoding="utf-8",
    )
    _mirror_points(_MIRROR_TRUTH).to_csv(
        root / "zero_shot" / "mirror" / "predictions.csv", index=False
    )
    offset = {key: [value + 1.0 for value in values] for key, values in _MIRROR_TRUTH.items()}
    _mirror_points(offset).to_csv(root / "finetune" / "mirror" / "predictions.csv", index=False)
    pd.DataFrame(_OFFICIAL_POINTS).to_csv(
        root / "finetune" / "official" / "predictions.csv", index=False
    )


def test_score_stage_scores_assembles_submission_and_writes_report(tmp_path):
    pytest.importorskip("pyarrow")
    config = _config(tmp_path)
    _write_prepared_tree(config)
    _write_fetched_tree(config)
    calls: list[list[str]] = []

    run_stages(config, _recording_runner(calls), stages=["score"])

    assert calls == []  # scoring runs locally: no CLI commands
    scores = json.loads(config.scores_path.read_text(encoding="utf-8"))
    assert list(scores) == ["baseline", "zero_shot", "finetune"]
    assert scores["zero_shot"] == 0.0  # identical points prove the row alignment
    assert all(math.isfinite(value) for value in scores.values())
    submission = pd.read_csv(config.submission_path)
    assert submission["id"].tolist() == [3, 1, 2]
    assert submission["sales"].tolist() == [5.0, 7.0, 9.0]
    report = config.report_path.read_text(encoding="utf-8")
    assert "mirror (non-official)" in report
    assert "corpus_overlap: pending" in report
    assert "| zero_shot |" in report
    assert "| finetune |" in report


def test_score_stage_requires_finetune_mode(tmp_path):
    config = _config(tmp_path)
    root = config.fetched_dir / "kb-out"
    root.mkdir(parents=True)
    (root / "workload.json").write_text('{"modes": ["zero_shot"]}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="finetune"):
        run_stages(config, _recording_runner([]), stages=["score"])


def test_score_stage_rejects_non_mapping_workload(tmp_path):
    config = _config(tmp_path)
    root = config.fetched_dir / "kb-out"
    root.mkdir(parents=True)
    (root / "workload.json").write_text("[1, 2]", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not a mapping"):
        run_stages(config, _recording_runner([]), stages=["score"])


# --------------------------------------------------------------------------- #
# mlcspp dispatch (the fixture satisfies the pinned horizon=157 default)
# --------------------------------------------------------------------------- #

_MLCSPP_SLUG = "mlcspp"
_MLCSPP_DATASET_ID = "alessiosavi/kb-mlcspp"
_MLCSPP_KERNEL_ID = "alessiosavi/kb-mlcspp-run"
_MLCSPP_STOCKS = ("1101", "2327", "2352", "2385", "2449", "2915", "3005", "3532", "6176", "9945")
_MLCSPP_FIELDS = (
    "Turnover",
    "Volume",
    "close",
    "eps",
    "high",
    "low",
    "open",
    "ups_and_downs",
    "ups_percent",
)
_MLCSPP_TOTAL_DAYS = 317
_MLCSPP_HORIZON = 157


def _mlcspp_config(tmp_path: Path) -> RunnerConfig:
    return RunnerConfig(
        slug=_MLCSPP_SLUG,
        evidence_root=tmp_path / "runs" / _MLCSPP_SLUG,
        dataset_id=_MLCSPP_DATASET_ID,
        kernel_id=_MLCSPP_KERNEL_ID,
        assets_dataset=_ASSETS_DATASET,
        repo_sha=_REPO_SHA,
    )


def _write_mlcspp_raw(raw: Path) -> None:
    """A raw mlcspp fixture the pinned horizon fits: 160 observed + 157 missing days."""
    raw.mkdir(parents=True, exist_ok=True)
    dates = list(range(1000, 1000 + _MLCSPP_TOTAL_DAYS))
    observed = _MLCSPP_TOTAL_DAYS - _MLCSPP_HORIZON
    rank_of = {date: rank for rank, date in enumerate(dates)}
    order = dates[::2] + dates[1::2]  # target file: evens then odds, never sorted
    pd.DataFrame(
        {
            "date": order,
            "0056_close_y": [
                10.0 + rank_of[date] if rank_of[date] < observed else float("nan") for date in order
            ],
        }
    ).to_csv(raw / "0056.csv", index=False)
    for index, stock in enumerate(_MLCSPP_STOCKS):
        rotated = order[index:] + order[:index]
        data = {"date": rotated}
        for field in _MLCSPP_FIELDS:
            data[f"{stock}_{field}"] = [float(date) for date in rotated]
        pd.DataFrame(data).to_csv(raw / f"{stock}.csv", index=False)
    pd.DataFrame(
        {
            "date": list(reversed(dates[-_MLCSPP_HORIZON:])),
            "0056_close_y": [0.0] * _MLCSPP_HORIZON,
        }
    ).to_csv(raw / "sample_submission.csv", index=False)


def test_prepare_stage_routes_mlcspp_to_mlcspp_prepare(tmp_path):
    pytest.importorskip("pyarrow")
    config = _mlcspp_config(tmp_path)
    _write_mlcspp_raw(config.raw_dir)

    run_stages(config, _recording_runner([]), stages=["prepare"])

    report = json.loads(
        (config.prepared_dir / "official" / "prepare.json").read_text(encoding="utf-8")
    )
    assert report["slug"] == "mlcspp"
    assert report["train_days"] == _MLCSPP_TOTAL_DAYS - _MLCSPP_HORIZON
    assert report["test_days"] == _MLCSPP_HORIZON
    assert (config.prepared_dir / "official" / "date_map.parquet").is_file()
    assert (config.prepared_dir / "mirror" / "truth.parquet").is_file()


def test_prepare_stage_rejects_unknown_competition_slug(tmp_path):
    config = _config(tmp_path, slug="mystery-competition")

    with pytest.raises(RuntimeError, match="mystery-competition"):
        run_stages(config, _recording_runner([]), stages=["prepare"])


def test_score_stage_rejects_unknown_competition_slug(tmp_path):
    config = _config(tmp_path, slug="mystery-competition")
    root = config.fetched_dir / "kb-out"
    (root / "finetune" / "mirror").mkdir(parents=True)
    (root / "workload.json").write_text(json.dumps({"modes": ["finetune"]}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="mystery-competition"):
        run_stages(config, _recording_runner([]), stages=["score"])

    assert not config.scores_path.exists()  # an unknown slug writes nothing


def _write_mlcspp_fetched_tree(config: RunnerConfig) -> dict[int, float]:
    """Fake kernel output for the mlcspp tree; return the expected official points."""
    root = config.fetched_dir / "kb-out"
    (root / "finetune" / "mirror").mkdir(parents=True)
    (root / "finetune" / "official").mkdir(parents=True)
    (root / "workload.json").write_text(
        json.dumps({"modes": ["finetune"], "windows": ["mirror", "official"]}),
        encoding="utf-8",
    )

    truth = pd.read_parquet(config.prepared_dir / "mirror" / "truth.parquet")
    mirror_points = truth["close"] + 1.0
    pd.DataFrame(
        {
            "series_id": truth["series_id"],
            "timestamp": truth["date"],
            "point": mirror_points,
            "q0.5": mirror_points,
        }
    ).to_csv(root / "finetune" / "mirror" / "predictions.csv", index=False)

    sample = pd.read_csv(config.prepared_dir / "official" / "sample_submission.csv")
    date_map = pd.read_parquet(config.prepared_dir / "official" / "date_map.parquet")
    window = date_map[date_map["date_int"].isin(sample["date"])]
    raw_dates = [int(value) for value in window["date_int"]]
    points = [float(index + 1) for index in range(len(window))]
    stamps = window["date"].reset_index(drop=True)
    pd.DataFrame(
        {"series_id": "0056", "timestamp": stamps, "point": points, "q0.5": points}
    ).to_csv(root / "finetune" / "official" / "predictions.csv", index=False)
    return dict(zip(raw_dates, points))


def test_score_stage_assembles_mlcspp_submission(tmp_path):
    pytest.importorskip("pyarrow")
    config = _mlcspp_config(tmp_path)
    _write_mlcspp_raw(config.raw_dir)
    run_stages(config, _recording_runner([]), stages=["prepare"])
    expected_points = _write_mlcspp_fetched_tree(config)
    calls: list[list[str]] = []

    run_stages(config, _recording_runner(calls), stages=["score"])

    assert calls == []  # scoring runs locally: no CLI commands
    scores = json.loads(config.scores_path.read_text(encoding="utf-8"))
    assert list(scores) == ["baseline", "finetune"]
    assert scores["finetune"] == pytest.approx(1.0)  # mirror points are truth + 1
    assert all(math.isfinite(value) for value in scores.values())
    sample = pd.read_csv(config.prepared_dir / "official" / "sample_submission.csv")
    submission = pd.read_csv(config.submission_path)
    assert submission["date"].tolist() == sample["date"].tolist()
    assert submission["0056_close_y"].tolist() == [
        expected_points[date] for date in submission["date"]
    ]
    report = config.report_path.read_text(encoding="utf-8")
    assert "mirror (non-official)" in report
    assert "| finetune |" in report


# --------------------------------------------------------------------------- #
# Operator script dry run
# --------------------------------------------------------------------------- #


def _load_run_script():
    path = _REPO_ROOT / "scripts" / "kaggle" / "run.py"
    spec = importlib.util.spec_from_file_location("kaggle_run_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_py_dry_run_lists_stages(capsys):
    module = _load_run_script()

    assert module.main(["--dry-run", "--stage", "all"]) == 0

    out = capsys.readouterr().out
    for stage in STAGES:
        assert f"{stage}:" in out
    assert f"evidence root: {module._EVIDENCE_ROOT / _SLUG}" in out
    assert "kaggle competitions download" in out
    assert "kaggle datasets version" in out
    assert "kaggle kernels push" in out
    assert "kaggle kernels output" in out
    assert "kaggle competitions submit" in out
    assert "(direct call)" in out  # prepare and score run in-process


def test_run_py_dry_run_all_until_submit_omits_submit(capsys):
    module = _load_run_script()

    assert module.main(["--dry-run", "--stage", "all-until-submit"]) == 0

    out = capsys.readouterr().out
    assert "fetch:" in out
    assert "\nsubmit:" not in out
