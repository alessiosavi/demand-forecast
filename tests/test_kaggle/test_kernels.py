"""Kernel packaging and CLI command builder tests (pure: files only, no CLI)."""

import ast
import json
from pathlib import Path

import pytest

from demand_forecast.kaggle.kernels import (
    check_push,
    check_status,
    output_argv,
    parse_status,
    push_argv,
    status_argv,
    write_kernel_dir,
)

_KERNEL_ID = "alessiosavi/kb-store-sales-run"
_TITLE = "kb store sales L3 run"
_PREPARED = "alessiosavi/kb-store-sales"
_ASSETS = "alessiosavi/kb-assets"
_REPO_SHA = "0123456789abcdef0123456789abcdef01234567"
_BUNDLE_IDENTITY = "83d6acb1f7f2c1b57ba33c0ca0a870775369d52bf8fb739a9e3f7e28858826c2"
_WORKLOAD_KWARGS = {
    "prepared_root": "/kaggle/input/kb-store-sales",
    "out_root": "/kaggle/working/kb-out",
    "init_from": "/kaggle/input/kb-assets/bundle-L3-first-try",
    "bundle_identity": _BUNDLE_IDENTITY,
}
# The verbatim live rejection: printed on stdout with exit code 0.
_PUSH_REJECTION = "Kernel push error: Maximum batch GPU session count of 2 reached."
_PUSH_SUCCESS = (
    "Kernel version 1 successfully pushed.  "
    "Please check progress at https://www.kaggle.com/code/alessiosavi/kb-store-sales-run"
)


def _write_kernel(directory: Path, *, kernel_id: str = _KERNEL_ID) -> Path:
    return write_kernel_dir(
        directory,
        kernel_id=kernel_id,
        title=_TITLE,
        prepared_dataset=_PREPARED,
        assets_dataset=_ASSETS,
        repo_sha=_REPO_SHA,
        workload_kwargs=dict(_WORKLOAD_KWARGS),
    )


def test_kernel_metadata_fields(tmp_path: Path):
    directory = _write_kernel(tmp_path)
    assert directory == tmp_path
    assert (tmp_path / "main.py").is_file()
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata == {
        "id": _KERNEL_ID,
        "title": _TITLE,
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "",
        "dataset_sources": [_PREPARED, _ASSETS],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def test_driver_pins_repo_sha_and_calls_workload(tmp_path: Path):
    _write_kernel(tmp_path)
    driver = (tmp_path / "main.py").read_text(encoding="utf-8")
    assert (
        'subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", '
        f'"git+https://github.com/alessiosavi/demand-forecast@{_REPO_SHA}"])' in driver
    )
    assert "/kaggle/input/kb-store-sales" in driver
    assert "/kaggle/input/kb-assets/bundle-L3-first-try" in driver
    assert "/kaggle/working/kb-out" in driver
    assert "from demand_forecast.kaggle.kernel_workload import run_workload" in driver
    assert "run_workload(**WORKLOAD_KWARGS)" in driver
    assert driver.rstrip().endswith('print("WORKLOAD_OK", flush=True)')

    module = ast.parse(driver)
    assignments = [
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == "WORKLOAD_KWARGS" for target in node.targets)
    ]
    assert len(assignments) == 1
    assert ast.literal_eval(assignments[0].value) == _WORKLOAD_KWARGS


def test_push_status_output_argv(tmp_path: Path):
    assert push_argv(tmp_path) == [
        "kaggle",
        "kernels",
        "push",
        "-p",
        str(tmp_path),
        "--accelerator",
        "NvidiaTeslaT4",
        "-t",
        "21600",
    ]
    assert push_argv(tmp_path, accelerator="NvidiaTeslaA100", timeout_s=3600) == [
        "kaggle",
        "kernels",
        "push",
        "-p",
        str(tmp_path),
        "--accelerator",
        "NvidiaTeslaA100",
        "-t",
        "3600",
    ]
    assert status_argv(_KERNEL_ID) == ["kaggle", "kernels", "status", _KERNEL_ID]
    assert output_argv(_KERNEL_ID, tmp_path / "fetched") == [
        "kaggle",
        "kernels",
        "output",
        "-p",
        str(tmp_path / "fetched"),
        _KERNEL_ID,
    ]


def test_status_failure_raises():
    with pytest.raises(RuntimeError, match="ERROR"):
        check_status(0, f'{_KERNEL_ID} has status "KernelWorkerStatus.ERROR"')
    with pytest.raises(RuntimeError, match="CANCEL_REQUESTED"):
        check_status(0, f'{_KERNEL_ID} has status "KernelWorkerStatus.CANCEL_REQUESTED"')
    with pytest.raises(RuntimeError, match="CANCEL_ACKNOWLEDGED"):
        check_status(0, f'{_KERNEL_ID} has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"')
    with pytest.raises(RuntimeError, match="out of memory"):
        check_status(
            0,
            f'{_KERNEL_ID} has status "KernelWorkerStatus.ERROR"\nFailure message: "out of memory"',
        )


@pytest.mark.parametrize(
    ("output", "word"),
    [
        (f'{_KERNEL_ID} has status "KernelWorkerStatus.RUNNING"', "RUNNING"),
        (f'{_KERNEL_ID} has status "KernelWorkerStatus.COMPLETE"', "COMPLETE"),
        (
            f'{_KERNEL_ID} has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"',
            "CANCEL_ACKNOWLEDGED",
        ),
        ("KernelWorkerStatus.ERROR", "ERROR"),
        ("QUEUED", "QUEUED"),
    ],
)
def test_parse_status_reads_the_status_word(output: str, word: str):
    assert parse_status(output) == word


def test_parse_status_rejects_unparseable_output():
    with pytest.raises(ValueError, match="no kernel status"):
        parse_status("some unexpected text")


def test_check_status_allows_queued_running_and_complete():
    check_status(0, f'{_KERNEL_ID} has status "KernelWorkerStatus.QUEUED"')
    check_status(0, f'{_KERNEL_ID} has status "KernelWorkerStatus.RUNNING"')
    check_status(0, f'{_KERNEL_ID} has status "KernelWorkerStatus.COMPLETE"')


def test_check_status_raises_on_cli_failure():
    with pytest.raises(RuntimeError, match="exit code 1"):
        check_status(1, "403 - Permission 'kernels.get' was denied")


def test_check_push_raises_on_rejected_push():
    with pytest.raises(RuntimeError, match="Maximum batch GPU session count of 2 reached"):
        check_push(_PUSH_REJECTION)


def test_check_push_raises_when_rejection_is_mid_output():
    output = f"Pushing kernel {_KERNEL_ID}...\n{_PUSH_REJECTION}\n"
    with pytest.raises(RuntimeError, match="Maximum batch GPU session count of 2 reached"):
        check_push(output)


def test_check_push_passes_on_success_and_empty_output():
    check_push(_PUSH_SUCCESS)
    check_push("")


def test_rerun_with_same_kernel_id_overwrites_cleanly(tmp_path: Path):
    directory = tmp_path / "kernel"
    assert _write_kernel(directory) == directory
    first = (directory / "main.py").read_text(encoding="utf-8")
    assert _write_kernel(directory) == directory
    assert (directory / "main.py").read_text(encoding="utf-8") == first


def test_existing_kernel_id_mismatch_raises(tmp_path: Path):
    _write_kernel(tmp_path)
    with pytest.raises(ValueError, match="kb-other-run"):
        _write_kernel(tmp_path, kernel_id="alessiosavi/kb-other-run")
