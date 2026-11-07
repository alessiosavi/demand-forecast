"""Stage orchestration for the Kaggle bench.

One shared runner drives a competition through the pinned stage sequence —
``download``, ``prepare``, ``upload``, ``kernel``, ``fetch``, ``score``,
``submit``. Subprocess stages receive every argv through an injectable
``CommandRunner``, so the whole sequence runs against fakes in tests without
touching the network or the Kaggle CLI; the direct-call stages (``prepare``,
``score``) call the bench modules in-process.

Evidence lives under ``RunnerConfig.evidence_root`` (``runs/<slug>/`` per the
plan):

- ``raw/`` — downloaded competition files (never published);
- ``prepared/`` — the competition module's converter output (official and
  mirror variants, settings, ``prepare.json``);
- ``upload/`` — the dataset upload staging area: ``dataset-metadata.json``
  plus one zip of the prepared tree whose entries are the tree's relative
  paths, i.e. the dataset's final layout. Uploading ``prepared/`` directly
  cannot work: ``--dir-mode zip`` archives each top-level directory flat (the
  variants collide on filenames) and ``datasets version`` skips directories by
  default, while the server extracts uploaded archives preserving their
  internal paths, so one pre-built archive uploads correctly through both;
- ``kernel/`` — the pushable kernel directory (metadata and generated driver);
- ``fetched/`` — the kernel output tree; a ``workload.json`` marker (written at
  the workload's out root) is the completeness check;
- ``scores.json``, ``submission.csv``, ``submissions.txt`` and ``REPORT.md``.

Failure surfacing is strict: a non-zero CLI exit, an ``ERROR`` or ``CANCEL*``
kernel status, unparseable status output, a fetched tree without
``workload.json`` and an invalid submission all raise instead of writing
success artifacts; the submit stage never retries — submission slots are
limited.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd  # type: ignore[import-untyped]  # repo venv carries no pandas stubs

from demand_forecast.data.loader_long import read_table
from demand_forecast.kaggle.dataset import dataset_argv, write_dataset_metadata
from demand_forecast.kaggle.kernels import (
    check_status,
    output_argv,
    parse_status,
    push_argv,
    status_argv,
    write_kernel_dir,
)
from demand_forecast.kaggle.report import write_report
from demand_forecast.kaggle.scoring import score_mirror
from demand_forecast.kaggle.store_sales import prepare
from demand_forecast.kaggle.submission import build_store_sales_submission, validate_submission

__all__ = [
    "CommandRunner",
    "RunnerConfig",
    "STAGES",
    "Stage",
    "fetch_outputs",
    "plan_stages",
    "run_stages",
    "wait_for_kernel",
]

CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]
Stage = Literal["download", "prepare", "upload", "kernel", "fetch", "score", "submit"]
STAGES: tuple[Stage, ...] = ("download", "prepare", "upload", "kernel", "fetch", "score", "submit")

_WORKLOAD_MARKER = "workload.json"
_KERNEL_COMPLETE = "COMPLETE"
_KERNEL_POLL_INTERVAL_S = 60.0
_KERNEL_OUT_ROOT = "/kaggle/working/kb-out"
_BUNDLE_DIRNAME = "bundle-L3-first-try"
# The pinned first-try L3 pretrain bundle identity (plan global constraint).
_BUNDLE_IDENTITY = "83d6acb1f7f2c1b57ba33c0ca0a870775369d52bf8fb739a9e3f7e28858826c2"
_SUBMISSION_MESSAGE = "demand-forecast L3"
_SUBMISSION_REF = re.compile(r"Submission ref:\s*(\d+)")


@dataclass(frozen=True)
class RunnerConfig:
    """Everything the stages need to locate their inputs, outputs and refs.

    ``evidence_root`` is the run directory (``runs/<slug>/``); every other path
    is derived from it so the layout has exactly one definition.
    """

    slug: str
    evidence_root: Path
    dataset_id: str
    kernel_id: str
    assets_dataset: str
    repo_sha: str

    @property
    def raw_dir(self) -> Path:
        return self.evidence_root / "raw"

    @property
    def prepared_dir(self) -> Path:
        return self.evidence_root / "prepared"

    @property
    def upload_dir(self) -> Path:
        return self.evidence_root / "upload"

    @property
    def kernel_dir(self) -> Path:
        return self.evidence_root / "kernel"

    @property
    def fetched_dir(self) -> Path:
        return self.evidence_root / "fetched"

    @property
    def scores_path(self) -> Path:
        return self.evidence_root / "scores.json"

    @property
    def submission_path(self) -> Path:
        return self.evidence_root / "submission.csv"

    @property
    def submissions_path(self) -> Path:
        return self.evidence_root / "submissions.txt"

    @property
    def report_path(self) -> Path:
        return self.evidence_root / "REPORT.md"


def _dataset_mount(dataset_id: str) -> str:
    """The ``/kaggle/input`` mount point of an ``owner/slug`` dataset."""
    return f"/kaggle/input/{dataset_id.split('/', 1)[1]}"


def _download_argv(slug: str, raw_dir: Path) -> list[str]:
    return ["kaggle", "competitions", "download", "-c", slug, "-p", str(raw_dir), "--unzip"]


def _submit_argv(slug: str, submission_path: Path) -> list[str]:
    return [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        slug,
        "-f",
        str(submission_path),
        "-m",
        _SUBMISSION_MESSAGE,
    ]


def _submissions_argv(slug: str) -> list[str]:
    return ["kaggle", "competitions", "submissions", "-c", slug]


def _upload_message(slug: str) -> str:
    return f"demand-forecast L3 {slug} prepared"


def _workload_kwargs(config: RunnerConfig) -> dict[str, str]:
    """The kernel workload kwargs: mount paths plus the pinned bundle identity."""
    return {
        "prepared_root": _dataset_mount(config.dataset_id),
        "out_root": _KERNEL_OUT_ROOT,
        "init_from": f"{_dataset_mount(config.assets_dataset)}/{_BUNDLE_DIRNAME}",
        "bundle_identity": _BUNDLE_IDENTITY,
    }


def _cli_output(result: subprocess.CompletedProcess[str]) -> str:
    """stdout plus stderr when present, so CLI failures carry their message."""
    stderr = result.stderr.strip()
    return result.stdout if not stderr else f"{result.stdout}\n{result.stderr}"


def _checked(run: CommandRunner, argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one CLI command; raise ``RuntimeError`` with its message on failure."""
    result = run(argv)
    if result.returncode != 0:
        detail = _cli_output(result).strip() or "(no CLI output)"
        raise RuntimeError(f"{' '.join(argv)} failed with exit code {result.returncode}: {detail}")
    return result


def plan_stages(config: RunnerConfig) -> list[list[str]]:
    """The command each stage would run, in canonical stage order.

    One argv vector per stage; the direct-call stages (``prepare`` and
    ``score``) contribute an empty vector because they run Python functions,
    not processes. Two entries are steady-state representatives: ``upload``
    lists the version command — at runtime the stage probes the dataset with
    ``kaggle datasets status`` and creates it instead when it does not exist
    yet — and ``submit`` lists the submit command, with the submissions
    listing recorded afterwards. The kernel program is written, not executed
    by argv, before ``kaggle kernels push``.
    """
    return [
        _download_argv(config.slug, config.raw_dir),
        [],
        dataset_argv("version", config.upload_dir, message=_upload_message(config.slug)),
        push_argv(config.kernel_dir),
        output_argv(config.kernel_id, config.fetched_dir),
        [],
        _submit_argv(config.slug, config.submission_path),
    ]


def wait_for_kernel(
    config: RunnerConfig,
    run: CommandRunner,
    *,
    sleep: Callable[[float], None] = time.sleep,
    interval_s: float = _KERNEL_POLL_INTERVAL_S,
) -> None:
    """Poll ``kaggle kernels status`` until the run reaches ``COMPLETE``.

    ``check_status`` raises ``RuntimeError`` on a CLI failure and on ``ERROR``
    or every ``CANCEL*`` status; ``parse_status`` raises ``ValueError`` on
    unparseable output (a return code of zero with no status word). Both
    surface to the caller. ``QUEUED``, ``RUNNING`` and ``NEW_SCRIPT`` are not
    terminal, so the loop sleeps and keeps polling.
    """
    while True:
        result = run(status_argv(config.kernel_id))
        output = _cli_output(result)
        check_status(result.returncode, output)
        if parse_status(output) == _KERNEL_COMPLETE:
            return
        sleep(interval_s)


def _find_workload(fetched: Path) -> Path | None:
    """The first ``workload.json`` under the fetched tree, if any."""
    if not fetched.is_dir():
        return None
    return next(iter(sorted(fetched.rglob(_WORKLOAD_MARKER))), None)


def _require_workload(fetched: Path) -> Path:
    marker = _find_workload(fetched)
    if marker is None:
        raise RuntimeError("no kernel output")
    return marker


def fetch_outputs(config: RunnerConfig, run: CommandRunner) -> Path:
    """Download the kernel output tree; return its ``workload.json`` marker.

    The fetched tree is cleared first: it is re-downloadable evidence of a
    download, and clearing keeps a marker from a previous run from masking a
    missing output. Raises ``RuntimeError`` when the CLI fails or the fetched
    tree carries no ``workload.json`` — a per-run ``run.json`` exists even for
    failed or partial trees, so the marker is the only completeness check.
    """
    fetched = config.fetched_dir
    shutil.rmtree(fetched, ignore_errors=True)
    _checked(run, output_argv(config.kernel_id, fetched))
    return _require_workload(fetched)


def _download(config: RunnerConfig, run: CommandRunner) -> None:
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    _checked(run, _download_argv(config.slug, config.raw_dir))


def _prepare(config: RunnerConfig, run: CommandRunner) -> None:
    prepare(config.raw_dir, config.prepared_dir)


def _stage_upload(config: RunnerConfig) -> Path:
    """Rebuild the upload staging area from the prepared tree; return it.

    The staging area holds ``dataset-metadata.json`` plus a single zip whose
    entries are the prepared tree's relative paths, POSIX-style
    (``official/long.parquet``, ``mirror/truth.parquet``, …) — exactly the
    dataset's final layout. Uploading ``prepared_dir`` directly cannot work:
    ``--dir-mode zip`` archives each top-level directory flat (the official
    and mirror variants would collide on filenames), and ``datasets version``
    skips directories by default. The server extracts uploaded archives
    preserving their internal paths — observed live: a folder holding only
    such an archive uploads cleanly through both commands, its paths becoming
    the dataset layout. The area is cleared first so re-runs leave nothing
    stale.
    """
    upload = config.upload_dir
    shutil.rmtree(upload, ignore_errors=True)
    upload.mkdir(parents=True, exist_ok=True)
    write_dataset_metadata(upload, dataset_id=config.dataset_id, title=f"kb {config.slug} prepared")
    with zipfile.ZipFile(
        upload / f"{config.slug}-prepared.zip", "w", zipfile.ZIP_DEFLATED
    ) as bundle:
        for path in sorted(config.prepared_dir.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(config.prepared_dir).as_posix())
    return upload


def _upload(config: RunnerConfig, run: CommandRunner) -> None:
    """Create the private prepared dataset on first upload, version it later.

    The prepared tree is staged first (see ``_stage_upload``) and the staged
    directory — never ``prepared_dir`` — is what the CLI receives.
    ``kaggle datasets status`` probes membership: success means the dataset
    exists and gets a new version (always carrying a message — the CLI
    requires ``-m`` for ``version``), failure means it is created empty of
    changelog text (``create`` takes no message).
    """
    if not config.prepared_dir.is_dir():
        raise RuntimeError(f"prepared tree {config.prepared_dir} is missing; run 'prepare' first")
    staged = _stage_upload(config)
    status = run(["kaggle", "datasets", "status", config.dataset_id])
    if status.returncode == 0:
        argv = dataset_argv("version", staged, message=_upload_message(config.slug))
    else:
        argv = dataset_argv("create", staged)
    _checked(run, argv)


def _kernel(config: RunnerConfig, run: CommandRunner) -> None:
    """Package and push the kernel, then poll until the run reaches COMPLETE."""
    write_kernel_dir(
        config.kernel_dir,
        kernel_id=config.kernel_id,
        title=f"kb {config.slug} run",
        prepared_dataset=config.dataset_id,
        assets_dataset=config.assets_dataset,
        repo_sha=config.repo_sha,
        workload_kwargs=_workload_kwargs(config),
    )
    _checked(run, push_argv(config.kernel_dir))
    wait_for_kernel(config, run)


def _fetch(config: RunnerConfig, run: CommandRunner) -> None:
    fetch_outputs(config, run)


def _read_workload(marker: Path) -> dict[str, object]:
    workload = json.loads(marker.read_text(encoding="utf-8"))
    if not isinstance(workload, dict):
        raise RuntimeError(f"{marker}: workload.json is not a mapping")
    return workload


def _score(config: RunnerConfig, run: CommandRunner) -> None:
    """Score the mirror window, assemble the submission, and write the report.

    The fetched predictions are anchored at the marker's parent — the kernel
    workload's out root, which holds ``<mode>/<window>/predictions.csv``. The
    submission candidate is assembled from the fetched official fine-tune
    predictions and validated against the prepared ``sample_submission.csv``;
    a validation failure raises here, before the submit stage can see it.
    """
    marker = _require_workload(config.fetched_dir)
    workload = _read_workload(marker)
    modes = workload.get("modes")
    if not isinstance(modes, list) or "finetune" not in modes:
        raise RuntimeError(f"{marker}: workload.json does not list the finetune mode")
    out_root = marker.parent
    frames = {
        str(mode): pd.read_csv(out_root / str(mode) / "mirror" / "predictions.csv")
        for mode in modes
    }
    prepared = config.prepared_dir
    scores = score_mirror(prepared / "mirror", frames)
    config.scores_path.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    predictions = pd.read_csv(out_root / "finetune" / "official" / "predictions.csv")
    test_ids = read_table(prepared / "official" / "test_ids.parquet")
    candidate = build_store_sales_submission(predictions, test_ids, config.submission_path)
    sample = pd.read_csv(prepared / "official" / "sample_submission.csv")
    validate_submission(candidate, sample)
    write_report(config.evidence_root, scores)


def _read_scores(config: RunnerConfig) -> dict[str, float]:
    if not config.scores_path.is_file():
        raise RuntimeError(f"{config.scores_path} is missing; run the 'score' stage first")
    raw = json.loads(config.scores_path.read_text(encoding="utf-8"))
    return {str(name): float(value) for name, value in raw.items()}


def _submission_ref(output: str) -> str | None:
    """The backend submission ref the CLI prints after a successful submit."""
    match = _SUBMISSION_REF.search(output)
    return None if match is None else match.group(1)


def _record_submission(config: RunnerConfig, run: CommandRunner, submit_stdout: str) -> None:
    """Record the ``kaggle competitions submissions`` listing for the evidence.

    Best effort by design: the submission is already made, so a listing hiccup
    must not make the operator think the upload failed (and re-submit blindly).
    """
    listing = run(_submissions_argv(config.slug))
    output = listing.stdout if listing.returncode == 0 and listing.stdout.strip() else submit_stdout
    config.submissions_path.write_text(output, encoding="utf-8")


def _submit(config: RunnerConfig, run: CommandRunner) -> None:
    """Submit the validated candidate once, then record the listing and ref.

    The candidate is re-validated against the prepared sample immediately
    before the upload, so a mismatched file can never reach the server. On a
    non-zero CLI exit the stage raises with the CLI message and never retries:
    submission slots are limited and a blind retry risks a duplicate.
    """
    scores = _read_scores(config)
    sample = pd.read_csv(config.prepared_dir / "official" / "sample_submission.csv")
    if not config.submission_path.is_file():
        raise RuntimeError(f"{config.submission_path} is missing; run the 'score' stage first")
    candidate = pd.read_csv(config.submission_path)
    validate_submission(candidate, sample)
    result = _checked(run, _submit_argv(config.slug, config.submission_path))
    _record_submission(config, run, result.stdout)
    ref = _submission_ref(result.stdout)
    if ref is not None:
        write_report(config.evidence_root, scores, submission_ref=ref)


_StageHandler = Callable[[RunnerConfig, CommandRunner], None]

_STAGE_HANDLERS: dict[Stage, _StageHandler] = {
    "download": _download,
    "prepare": _prepare,
    "upload": _upload,
    "kernel": _kernel,
    "fetch": _fetch,
    "score": _score,
    "submit": _submit,
}


def run_stages(config: RunnerConfig, run: CommandRunner, *, stages: Sequence[Stage]) -> None:
    """Execute ``stages`` in the given order against ``config``.

    Every stage fails loudly: exceptions from a stage stop the sequence before
    any later stage can write a success artifact on top of a failed one.
    """
    config.evidence_root.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        handler = _STAGE_HANDLERS.get(stage)
        if handler is None:
            raise ValueError(f"unknown stage {stage!r}; expected one of {sorted(STAGES)}")
        handler(config, run)
