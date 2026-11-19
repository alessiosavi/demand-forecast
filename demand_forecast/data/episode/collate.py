"""Pad numpy episodes and convert model arrays at the torch boundary."""

from dataclasses import dataclass, fields, replace

import numpy as np
import torch

from .builder import Episode


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


def collate(episodes: list[Episode], *, include_row_keys: bool = False) -> EpisodeBatch:
    """Pad each independent axis and shift origins by their context's left padding."""
    if not episodes:
        raise ValueError("cannot collate an empty list of episodes")
    result = {}
    lists = {"row_names", "cand_names", "attr_names", "context_target"}
    for field in fields(Episode):
        name = field.name
        if name in ("trace", "row_keys"):
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
    if include_row_keys:
        return EpisodeBatchWithRowKeys(
            **result, row_keys=[list(episode.row_keys) for episode in episodes]
        )
    return EpisodeBatch(**result)
