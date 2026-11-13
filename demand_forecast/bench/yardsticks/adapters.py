# SPDX-License-Identifier: Apache-2.0
# Portions adapted from autogluon/fev (Copyright Amazon.com, Inc. or its affiliates).
# Simons conversion follows Abel-ai-lab/Simons (Copyright 2026 Abel AI Lab).
# This product includes software developed at Abel AI Lab (https://abel.ai).
# Modified for shared ablations and audit inputs; see LICENSE-APACHE-2.0.
"""Published fev adapters, with explicit shared-runner ablations.

Settings/preprocessing follow autogluon/fev@9cf943c models/{chronos-2,timestral}.
Timestral batching and categorical typing are adapted from its Apache-2.0
fev implementation. Model libraries are imported only when loading weights.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .runner import SIMONS_SOURCE, Options, WindowInput


def _numpy(value):
    return value.detach().double().cpu().numpy()


class Chronos2Adapter:
    """Joint targets/covariates; cross-learning matches the published entry."""

    def __init__(self, options: Options, task, *, pipeline=None):
        self.options = options
        if pipeline is None:
            import torch
            from chronos import BaseChronosPipeline

            pipeline = BaseChronosPipeline.from_pretrained(
                options.weights,
                revision=options.weights_revision,
                device_map=options.device,
                torch_dtype=torch.float32,
            )
        self.pipeline = pipeline

    def prepare(self, task):
        """No task-wide preprocessing is required for Chronos-2."""

    def predict(
        self, windows: list[WindowInput], horizon: int, levels: list[float]
    ) -> list[list[np.ndarray]]:
        """Match predict_fev's role conversion and minimum variate batch size."""
        import pyarrow as pa

        result = []
        for window in windows:
            # Upstream casts nonnumeric columns (including booleans) to Arrow
            # strings, preserving nulls, before the pipeline's own encoding.
            categorical = {
                c for c, a in window.items[0].past.items() if not np.issubdtype(a.dtype, np.number)
            }

            def cast(columns):
                return {
                    c: pa.array(a).cast(pa.string()).to_numpy(zero_copy_only=False)
                    if c in categorical
                    else a.astype(np.float64)
                    for c, a in columns.items()
                }

            inputs = []
            for item in window.items:
                entry = {"target": item.target}
                if item.past:
                    entry["past_covariates"] = cast(item.past)
                if item.future:
                    entry["future_covariates"] = cast(item.future)
                inputs.append(entry)
            variates = max(it.target.shape[0] + len(it.past) for it in window.items)
            kwargs = dict(
                batch_size=max(self.options.batch_size, variates),
                cross_learning=self.options.cross_series == "on",
                limit_prediction_length=False,
            )
            if self.options.max_context is not None:
                kwargs["context_length"] = self.options.max_context
            q, _ = self.pipeline.predict_quantiles(
                inputs=inputs, prediction_length=horizon, quantile_levels=levels, **kwargs
            )
            result.append([_numpy(a) for a in q])
        return result


class SimonsAdapter:
    """Local Simons exports with isolated item groups and explicit processing."""

    def __init__(self, options: Options, task):
        self.options = options.resolved()
        try:
            from simons import Simons, forecast_series
        except ImportError as exc:
            raise ImportError(
                "Simons inference is optional; install "
                f"'simons @ git+https://github.com/Abel-ai-lab/Simons.git@{SIMONS_SOURCE}'"
            ) from exc
        self.model = Simons.from_pretrained(options.weights, device=options.device)
        self.forecast_series = forecast_series
        if options.processing == "raw":
            # The public API accepts a views override only. Replace the frozen
            # config in one operation (xreg_guard requires xreg) for all six rules.
            # Learned covariate modules and intrinsic normalization are retained.
            self.model.config = replace(
                self.model.config,
                views=False,
                sign_clip=False,
                integer_snap=False,
                fit_row=False,
                xreg=False,
                xreg_guard=False,
            )

    def prepare(self, task):
        """No dataset-wide category inference; upstream encodes each input window."""

    def predict(
        self, windows: list[WindowInput], horizon: int, levels: list[float]
    ) -> list[list[np.ndarray]]:
        """Match pinned fev quantile sorting/interpolation and known-covariate roles."""
        result = []
        model_levels = np.arange(1, 10) / 10
        for window in windows:
            series, covariates, sizes = [], [], []
            for item in window.items:
                target = item.target
                if self.options.max_context is not None:
                    target = target[:, -self.options.max_context :]
                known = {}
                if self.options.covariates == "all":
                    for column, future in item.future.items():
                        if len(future) < horizon:
                            raise ValueError(
                                f"{column}: known covariate is shorter than the horizon"
                            )
                        known[column] = np.concatenate(
                            [
                                np.asarray(item.past[column][-target.shape[-1] :], dtype=object),
                                np.asarray(future[:horizon], dtype=object),
                            ]
                        )
                # Contiguous blocks match forecast_items' MAX_VARIATES=32 split.
                # Never concatenate targets from different fev items into a group.
                blocks = [target[i : i + 32].astype(np.float32) for i in range(0, len(target), 32)]
                series.extend(blocks)
                covariates.extend([known or None] * len(blocks))
                sizes.append(len(blocks))
            outputs = self.forecast_series(
                self.model,
                series,
                horizon,
                future_covariates=covariates,
                # Past-only covariates are unsupported. Counts live in run.json.
                past_covariates=None,
                shared_categories=False,
                batch_size=self.options.batch_size,
            )
            if len(outputs) != len(series):
                raise ValueError("Simons forecast group count mismatch")
            values = []
            for target, output in zip(series, outputs, strict=True):
                q = np.sort(np.asarray(output), axis=-1)
                if q.shape != (len(target), horizon, 9):
                    raise ValueError("Simons forecast shape mismatch")
                values.append(
                    np.stack(
                        [
                            np.apply_along_axis(lambda a: np.interp(level, model_levels, a), -1, q)
                            for level in levels
                        ],
                        axis=-1,
                    )
                )
            items, start = [], 0
            for size in sizes:
                items.append(np.concatenate(values[start : start + size], axis=0))
                start += size
            result.append(items)
        return result


@dataclass(frozen=True)
class _ColumnKind:
    kind: str
    categories: tuple = ()


def _infer_kind(values):
    series = pd.Series(values)
    non_null = series.dropna()
    if pd.api.types.is_bool_dtype(series) or (
        len(non_null) > 0 and non_null.map(lambda v: isinstance(v, (bool, np.bool_))).all()
    ):
        return _ColumnKind("bool", (False, True))
    if pd.to_numeric(series, errors="coerce").notna().equals(series.notna()):
        return _ColumnKind("numeric")
    return _ColumnKind("string", tuple(pd.Categorical(series).categories))


def _encode(values, kind):
    if kind.kind == "string":
        codes = pd.Categorical(values, categories=list(kind.categories)).codes.astype(np.float64)
        codes[codes < 0] = np.nan
        return codes
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)


@dataclass
class _Item:
    target: np.ndarray
    past: np.ndarray
    past_cat: np.ndarray
    known: np.ndarray
    known_cat: np.ndarray

    @property
    def length(self):
        return self.target.shape[-1]

    @property
    def rows(self):
        return sum(
            a.shape[0] for a in (self.target, self.past, self.past_cat, self.known, self.known_cat)
        )


def _stack(arrays, length):
    return np.stack(arrays) if arrays else np.zeros((0, length))


def _pad(arrays, length):
    if arrays[0].shape[0] == 0:
        return None
    out = np.full((len(arrays), arrays[0].shape[0], length), np.nan, dtype=np.float32)
    for i, a in enumerate(arrays):
        out[i, :, length - a.shape[-1] :] = a
    return out


class TimestralAdapter:
    """Joint targets, per-series categorical encoding, adaptive cross-window batches."""

    batch_budget = 2**23

    def __init__(self, options: Options, task, *, forecaster=None):
        self.options = options
        if forecaster is None:
            from timestral import TimestralForecaster

            forecaster = TimestralForecaster.from_pretrained(
                options.weights, device=options.device, revision=options.weights_revision
            )
        self.forecaster = forecaster
        if options.max_context is not None:
            self.forecaster.max_context = min(forecaster.max_context, options.max_context)
        self.prepare(task)

    def prepare(self, task):
        """Infer column types/categories over the full dataset, as published.

        Only types and category vocabulary use the full column. Target encoding
        happens inside the forecaster using the current past context only.
        """
        import pyarrow.compute as pc

        self.kinds = {}
        if self.options.covariates == "all":
            table = task.load_full_dataset().data.table
            self.kinds = {
                c: _infer_kind(pc.list_flatten(table.column(c)).to_numpy(zero_copy_only=False))
                for c in task.dynamic_columns
            }

    def _item(self, item, horizon):
        cap = self.forecaster.max_context
        target = item.target[:, -cap:]
        t = target.shape[-1]
        past, past_cat, known, known_cat = [], [], [], []
        for c, a in item.past.items():
            kind = self.kinds[c]
            values = _encode(a[-cap:], kind)
            if c in item.future:
                values = np.concatenate([values, _encode(item.future[c], kind)])
                (known if kind.kind == "numeric" else known_cat).append(values)
            else:
                (past if kind.kind == "numeric" else past_cat).append(values)
        return _Item(
            target,
            _stack(past, t),
            _stack(past_cat, t),
            _stack(known, t + horizon),
            _stack(known_cat, t + horizon),
        )

    def _batches(self, items, horizon):
        order = sorted(range(len(items)), key=lambda i: items[i].length)
        batches, current, maximum = [], [], 0
        for i in order:
            new_max = max(maximum, items[i].length + horizon)
            rows = sum(items[j].rows for j in current) + items[i].rows
            if current and (
                rows * new_max > self.batch_budget or len(current) >= self.options.batch_size
            ):
                batches.append(current)
                current, new_max = [], items[i].length + horizon
            current.append(i)
            maximum = new_max
        if current:
            batches.append(current)
        return batches

    def _forecast_batch(self, batch, horizon):
        length = max(it.length for it in batch)
        forecast = self.forecaster.forecast(
            _pad([it.target for it in batch], length),
            horizon,
            past_covariates=_pad([it.past for it in batch], length),
            known_covariates=_pad([it.known for it in batch], length + horizon),
            categorical_past_covariates=_pad([it.past_cat for it in batch], length),
            categorical_known_covariates=_pad([it.known_cat for it in batch], length + horizon),
        )
        return _numpy(forecast.quantiles)

    def predict(
        self, windows: list[WindowInput], horizon: int, levels: list[float]
    ) -> list[list[np.ndarray]]:
        """Batch every window together, then restore original item/window order."""
        import torch

        items = [self._item(it, horizon) for w in windows for it in w.items]
        outputs = [None] * len(items)
        pending = self._batches(items, horizon)
        model_levels = list(self.forecaster.quantile_levels)
        while pending:
            indices = pending.pop(0)
            try:
                q = self._forecast_batch([items[i] for i in indices], horizon)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(indices) == 1:
                    raise
                half = len(indices) // 2
                pending[:0] = [indices[:half], indices[half:]]
                continue
            # Same level selection/interpolation as the published adapter.
            cols = []
            for level in levels:
                match = [i for i, m in enumerate(model_levels) if abs(m - level) < 1e-9]
                cols.append(
                    q[..., match[0]]
                    if match
                    else np.apply_along_axis(lambda a: np.interp(level, model_levels, a), -1, q)
                )
            values = np.stack(cols, axis=-1)
            for j, i in enumerate(indices):
                outputs[i] = values[j]
        result, start = [], 0
        for window in windows:
            result.append(outputs[start : start + len(window.items)])
            start += len(window.items)
        return result
