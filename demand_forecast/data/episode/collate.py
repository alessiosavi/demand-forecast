"""Pad numpy episodes and convert model arrays at the torch boundary."""

from dataclasses import dataclass, fields, replace

import numpy as np
import torch

from .builder import Episode
from .schema import NAIVE_OPTIONS, QUANTILES


@dataclass
class EpisodeBatch:
    """Padded episode fields and masks, with context padding on the left."""

    x: torch.Tensor
    patch_valid: torch.Tensor
    row_kind: torch.Tensor
    row_role: torch.Tensor
    row_names: list[list[str]]
    row_summary: torch.Tensor
    row_summary_flags: torch.Tensor
    row_score: torch.Tensor
    row_cand: torch.Tensor
    forecast_rows: torch.Tensor
    keep: torch.Tensor
    labels: torch.Tensor
    label_mask: torch.Tensor
    loc: torch.Tensor
    scale: torch.Tensor
    fallback: torch.Tensor
    known_rows: torch.Tensor
    xh: torch.Tensor
    cal_phases: torch.Tensor
    cal_ids: torch.Tensor
    calh_phases: torch.Tensor
    calh_ids: torch.Tensor
    cand_names: list[list[str]]
    cand_kind: torch.Tensor
    cand_summary: torch.Tensor
    cand_flags: torch.Tensor
    cand_score: torch.Tensor
    attr_stats: torch.Tensor
    attr_flags: torch.Tensor
    attr_cand: torch.Tensor
    attr_names: list[list[str]]
    static_num: torch.Tensor
    static_num_flags: torch.Tensor
    static_num_cand: torch.Tensor
    freq_class: torch.Tensor
    horizon: torch.Tensor
    origin_patches: torch.Tensor
    context_target: list[str | None]
    row_mask: torch.Tensor
    forecast_mask: torch.Tensor
    known_mask: torch.Tensor
    cand_mask: torch.Tensor
    attr_mask: torch.Tensor
    static_num_mask: torch.Tensor
    origin_mask: torch.Tensor
    horizon_mask: torch.Tensor

    def to(self, device: torch.device | str) -> "EpisodeBatch":
        """Return a batch whose tensors are on device, preserving reporting names."""
        return replace(
            self,
            **{
                field.name: getattr(self, field.name).to(device)
                for field in fields(self)
                if isinstance(getattr(self, field.name), torch.Tensor)
            },
        )


@dataclass
class EpisodeBatchWithRowKeys(EpisodeBatch):
    """An episode batch with candidate provenance for the unpadded rows."""

    row_keys: list[list[tuple[str, str] | None]]


@dataclass
class EpisodeBatchWithNaive(EpisodeBatch):
    """An episode batch with uncompressed naive paths in label units."""

    naive_paths: torch.Tensor
    naive_seen: torch.Tensor


@dataclass
class EpisodeBatchWithNaiveAndRowKeys(EpisodeBatchWithNaive, EpisodeBatchWithRowKeys):
    """Naive paths and row provenance for a forecasting batch hook."""


@dataclass
class EpisodeBatchWithNaiveEvidence(EpisodeBatchWithNaive):
    """Naive paths with per-patch backtest errors and signed residual quantiles."""

    naive_error: torch.Tensor
    naive_error_seen: torch.Tensor
    naive_resq: torch.Tensor


@dataclass
class EpisodeBatchWithNaiveEvidenceAndRowKeys(
    EpisodeBatchWithNaiveEvidence, EpisodeBatchWithNaiveAndRowKeys
):
    """Naive backtest evidence and row provenance for a forecasting batch hook."""


def collate(episodes: list[Episode], *, include_row_keys: bool = False) -> EpisodeBatch:
    """Pad each independent axis and shift origins by their context's left padding."""
    if not episodes:
        raise ValueError("cannot collate an empty list of episodes")
    result = {}
    lists = {"row_names", "cand_names", "attr_names", "context_target"}
    for field in fields(Episode):
        name = field.name
        if name in (
            "trace",
            "row_keys",
            "naive_paths",
            "naive_seen",
            "naive_error",
            "naive_error_seen",
            "naive_resq",
        ):
            continue
        values = [getattr(episode, name) for episode in episodes]
        if name in lists:
            result[name] = [value.copy() if isinstance(value, list) else value for value in values]
            continue
        if name in ("freq_class", "horizon"):
            result[name] = torch.tensor(values, dtype=torch.long)
            continue
        shape = tuple(max(value.shape[axis] for value in values) for axis in range(values[0].ndim))
        dtype = values[0].dtype
        if np.issubdtype(dtype, np.integer):
            dtype = np.int64
        padded = np.full(
            (len(episodes), *shape), -1 if name in ("cal_ids", "calh_ids") else 0, dtype=dtype
        )
        for index, value in enumerate(values):
            slices = [slice(0, length) for length in value.shape]
            context_axis = (
                1
                if name in ("x", "patch_valid")
                else 0
                if name in ("cal_phases", "cal_ids")
                else None
            )
            if context_axis is not None:
                slices[context_axis] = slice(
                    shape[context_axis] - value.shape[context_axis], shape[context_axis]
                )
            if name == "origin_patches":
                value = value + max(e.x.shape[1] for e in episodes) - episodes[index].x.shape[1]
            padded[(index, *slices)] = value
        result[name] = torch.from_numpy(padded)
    for name, lengths in (
        ("row_mask", [len(e.x) for e in episodes]),
        ("forecast_mask", [len(e.forecast_rows) for e in episodes]),
        ("known_mask", [len(e.known_rows) for e in episodes]),
        ("cand_mask", [len(e.cand_names) for e in episodes]),
        ("attr_mask", [len(e.attr_stats) for e in episodes]),
        ("static_num_mask", [len(e.static_num) for e in episodes]),
        ("origin_mask", [len(e.origin_patches) for e in episodes]),
        ("horizon_mask", [e.calh_phases.shape[1] for e in episodes]),
    ):
        result[name] = torch.arange(max(lengths))[None, :] < torch.tensor(lengths)[:, None]
    have_naive = any(episode.naive_paths is not None for episode in episodes)
    if have_naive:
        shape = (*result["labels"].shape, len(NAIVE_OPTIONS))
        paths, seen = np.zeros(shape, np.float32), np.zeros(shape, bool)
        for index, episode in enumerate(episodes):
            if episode.naive_paths is not None:
                slices = (index, *(slice(0, length) for length in episode.labels.shape))
                paths[slices] = episode.naive_paths
                seen[slices] = episode.naive_seen
        result.update(naive_paths=torch.from_numpy(paths), naive_seen=torch.from_numpy(seen))
    have_evidence = any(episode.naive_error is not None for episode in episodes)
    if have_evidence:
        shape = (*result["labels"].shape[:3], result["horizon_mask"].shape[1], len(NAIVE_OPTIONS))
        error = np.zeros(shape, np.float32)
        error_seen = np.zeros(shape, bool)
        resq = np.zeros((*shape, len(QUANTILES)), np.float32)
        for index, episode in enumerate(episodes):
            if episode.naive_error is not None:
                slices = (index, *(slice(0, length) for length in episode.naive_error.shape))
                error[slices] = episode.naive_error
                error_seen[slices] = episode.naive_error_seen
                resq[slices] = episode.naive_resq
        result.update(
            naive_error=torch.from_numpy(error),
            naive_error_seen=torch.from_numpy(error_seen),
            naive_resq=torch.from_numpy(resq),
        )
    if include_row_keys:
        result["row_keys"] = [list(episode.row_keys) for episode in episodes]
        if have_evidence:
            return EpisodeBatchWithNaiveEvidenceAndRowKeys(**result)
        return (
            EpisodeBatchWithNaiveAndRowKeys(**result)
            if have_naive
            else EpisodeBatchWithRowKeys(**result)
        )
    if have_evidence:
        return EpisodeBatchWithNaiveEvidence(**result)
    return EpisodeBatchWithNaive(**result) if have_naive else EpisodeBatch(**result)
