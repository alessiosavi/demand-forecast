# Ported from fev v0.10.0, models/statsforecast/model.py:
# https://github.com/autogluon/fev/blob/v0.10.0/models/statsforecast/model.py
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified: classical arms and their median, local seasonality, guarded optional
# imports, configurable serial execution, and fractional quantile interval levels.
# See THIRD_PARTY_NOTICES.md for the upstream notice and license.
"""Classical StatsForecast baselines through fev's evaluation protocol."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from demand_forecast.data.frequency import seasonal_period

try:
    import fev
except ImportError as exc:
    raise ImportError("install demand-forecast[bench]") from exc


BASELINE_MODELS = ("seasonal_naive", "auto_ets", "auto_theta", "median3")


class StatsForecastBaseline(fev.ForecastingModel):
    """Vendored StatsForecast baselines with fev's positional output.

    As upstream, at most the latest 1,000 observations per series are used by
    default, and the combined fit/forecast call counts as inference time.
    ``n_jobs=1`` avoids process startup for small panels; callers can opt into
    parallel execution. The seasonal period is never reduced by a context cap.
    """

    def __init__(
        self,
        model: str = "seasonal_naive",
        max_context_length: int | None = 1_000,
        n_jobs: int = 1,
    ):
        if model not in BASELINE_MODELS:
            raise ValueError(f"Unknown baseline model {model!r}; choose from {BASELINE_MODELS}")
        super().__init__()
        self.model_name = model
        self.max_context_length = max_context_length
        self.n_jobs = n_jobs

    def _fit_predict(self, task: fev.Task) -> list:
        try:
            from statsforecast import StatsForecast
            from statsforecast.models import AutoETS, AutoTheta, SeasonalNaive
        except ImportError as exc:
            raise ImportError("install demand-forecast[bench]") from exc

        task.load_full_dataset()
        season_length = max(task.seasonality, seasonal_period(task.freq))
        model_classes = {
            "seasonal_naive": SeasonalNaive,
            "auto_ets": AutoETS,
            "auto_theta": AutoTheta,
        }
        members = BASELINE_MODELS[:3] if self.model_name == "median3" else (self.model_name,)
        predictor = StatsForecast(
            freq="D",
            models=[model_classes[member](season_length=season_length) for member in members],
            fallback_model=SeasonalNaive(season_length=season_length),
            n_jobs=self.n_jobs,
            verbose=False,
        )
        return [
            self._predict_window(window, predictor, quantile_levels=task.quantile_levels)
            for window in task.iter_windows()
        ]

    def _predict_window(
        self, window: fev.EvaluationWindow, predictor, quantile_levels: list[float]
    ):
        train_df, _, _ = fev.convert_input_data(window, adapter="nixtla", as_univariate=True)
        if train_df["y"].isna().any():
            train_df["y"] = train_df.groupby("unique_id", sort=False)["y"].ffill().fillna(0.0)

        # Dates only label rows for these models. Match upstream's synthetic
        # daily forecast grid, with integer steps for out-of-bounds dates.
        if (train_df["ds"] > pd.Timestamp.max).any():
            train_df["ds"] = train_df.groupby("unique_id", sort=False).cumcount()
            predictor.freq = 1

        if self.max_context_length is not None:
            train_df = (
                train_df.groupby("unique_id", as_index=False)
                .tail(self.max_context_length)
                .reset_index(drop=True)
            )

        levels = sorted({round(abs(q - 0.5) * 200, 10) for q in quantile_levels})
        with self._record_inference_time(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast_df = predictor.forecast(
                h=window.horizon, df=train_df[["unique_id", "ds", "y"]], level=levels
            )

        if self.model_name == "median3":
            model_names = [str(model) for model in predictor.models]
            forecast_df["predictions"] = np.median(
                forecast_df[model_names].fillna(0.0).to_numpy(), axis=1
            )
            if quantile_levels:
                # Each row is one item/horizon step; combine only across models.
                quantiles = np.stack(
                    [
                        np.column_stack(
                            [
                                forecast_df[
                                    f"{name}-{'lo' if q < 0.5 else 'hi'}-"
                                    f"{round(abs(q - 0.5) * 200, 10)}"
                                ]
                                .fillna(0.0)
                                .to_numpy()
                                for q in quantile_levels
                            ]
                        )
                        for name in model_names
                    ]
                )
                forecast_df[[str(q) for q in quantile_levels]] = np.sort(
                    np.median(quantiles, axis=0), axis=-1
                )
        else:
            model_name = str(predictor.models[0])
            forecast_df["predictions"] = forecast_df[model_name]
            for q in quantile_levels:
                side = "lo" if q < 0.5 else "hi"
                level = round(abs(q - 0.5) * 200, 10)
                forecast_df[str(q)] = forecast_df[f"{model_name}-{side}-{level}"]

        return fev.utils.convert_forecast_df_to_predictions(
            forecast_df.fillna(0.0),
            horizon=window.horizon,
            quantile_levels=quantile_levels,
            target_columns=window.target_columns,
        )


class SeasonalNaiveBaseline(StatsForecastBaseline):
    """Backward-compatible seasonal-naive baseline and constructor."""

    model_name = "seasonal_naive"

    def __init__(self, max_context_length: int | None = 1_000, n_jobs: int = 1):
        super().__init__("seasonal_naive", max_context_length, n_jobs)


def make_baseline(model: str) -> StatsForecastBaseline:
    """Build a baseline by its summary-row model name."""
    return SeasonalNaiveBaseline() if model == "seasonal_naive" else StatsForecastBaseline(model)


def baseline_summary(task: fev.Task, model: str = "seasonal_naive") -> dict:
    """Evaluate a baseline on every window, including fev's timing fields."""
    baseline = make_baseline(model)
    return task.evaluation_summary(
        baseline.fit_predict(task),
        model_name=baseline.model_name,
        training_time_s=baseline.training_time,
        inference_time_s=baseline.inference_time,
    )
