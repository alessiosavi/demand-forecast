# Ported from fev v0.10.0, models/statsforecast/model.py:
# https://github.com/autogluon/fev/blob/v0.10.0/models/statsforecast/model.py
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified: SeasonalNaive only, local seasonality, guarded optional imports,
# configurable serial execution, and fractional quantile interval levels.
# See THIRD_PARTY_NOTICES.md for the upstream notice and license.
"""Seasonal-naive reference forecasts through fev's evaluation protocol."""

from __future__ import annotations

import warnings

import pandas as pd

from demand_forecast.data.frequency import seasonal_period

try:
    import fev
except ImportError as exc:
    raise ImportError("install demand-forecast[bench]") from exc


class SeasonalNaiveBaseline(fev.ForecastingModel):
    """Vendored StatsForecast seasonal naive with fev's positional output.

    As upstream, at most the latest 1,000 observations per series are used by
    default, and the combined fit/forecast call counts as inference time.
    ``n_jobs=1`` avoids process startup for small panels; callers can opt into
    parallel execution. The seasonal period is never reduced by a context cap.
    """

    model_name = "seasonal_naive"

    def __init__(self, max_context_length: int | None = 1_000, n_jobs: int = 1):
        super().__init__()
        self.max_context_length = max_context_length
        self.n_jobs = n_jobs

    def _fit_predict(self, task: fev.Task) -> list:
        try:
            from statsforecast import StatsForecast
            from statsforecast.models import SeasonalNaive
        except ImportError as exc:
            raise ImportError("install demand-forecast[bench]") from exc

        task.load_full_dataset()
        season_length = max(task.seasonality, seasonal_period(task.freq))
        predictor = StatsForecast(
            freq="D",
            models=[SeasonalNaive(season_length=season_length)],
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

        # Dates only label rows for this model. Match upstream's synthetic
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


def baseline_summary(task: fev.Task) -> dict:
    """Evaluate seasonal naive on every window, including fev's timing fields."""
    model = SeasonalNaiveBaseline()
    return task.evaluation_summary(
        model.fit_predict(task),
        model_name=model.model_name,
        training_time_s=model.training_time,
        inference_time_s=model.inference_time,
    )
