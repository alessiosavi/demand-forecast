"""Dated forecasting with independently fitted selection and refit preprocessing."""

import inspect
import logging
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from demand_forecast.config.settings import Settings
from demand_forecast.core.trainer import EarlyStopConfig, Trainer, move_batch
from demand_forecast.data.dataloader import create_dataloader, create_dataloaders
from demand_forecast.data.dataset import create_time_series_data
from demand_forecast.data.feature_engineering import (
    CategoricalEncoder,
    extract_metafeatures,
    get_categorical_columns,
)
from demand_forecast.data.loader import load_sales_data
from demand_forecast.data.preprocessor import (
    ScalerManager,
    catalogue_weekly_grid,
    create_sku_index,
    filter_skus,
    resample_series,
)
from demand_forecast.models.losses import point_forecast
from demand_forecast.models.wrapper import MODEL_REGISTRY, ModelWrapper
from demand_forecast.utils.checkpoint import load_checkpoint
from demand_forecast.utils.time_features import calculate_time_features

logger = logging.getLogger(__name__)
TIME_COLUMNS = ["p_t_sin", "p_t_cos", "p_m_sin", "p_m_cos"]


class ForecastPipeline:
    """All fitted state comes from rows strictly before ``cutoff``."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = torch.device(
            settings.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._set_seeds(settings.seed)
        self.scaler_manager = ScalerManager()
        self.categorical_encoder = CategoricalEncoder()
        self.sku_to_index = {}
        self.model = None
        self._series = None
        self._weekly = None
        self._raw_datasets = None
        self.products = None
        self.cutoff = None
        self.cluster_map = {}
        self.clip_bounds = {}
        self.static = None
        self.preprocessing_seconds = []
        self._loaded = False

    def _set_seeds(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @property
    def store_column(self):
        return self.settings.data.store_column or "store_id"

    @property
    def continuous_features(self):
        return [f"continuous_{name}" for name in self.categorical_encoder.continuous_stats]

    def load_data(self, max_products=None):
        cfg = self.settings.data
        sales = load_sales_data(
            cfg.input_path,
            cfg.date_column,
            cfg.sku_column,
            cfg.quantity_column,
            cfg.store_column,
            cfg.product_id_column,
            cfg.sales_qty_column,
        )
        quantities = sales[cfg.quantity_column].to_numpy(dtype=float)
        if not np.isfinite(quantities).all() or (quantities < 0).any():
            raise ValueError("Sales quantities must be finite and non-negative")
        if cfg.products_path:
            columns = [cfg.product_key_column, *cfg.product_categorical, *cfg.product_continuous]
            products = pd.read_csv(
                cfg.products_path, usecols=columns, dtype={cfg.product_key_column: str}
            )
            if (
                products[cfg.product_key_column].isna().any()
                or products[cfg.product_key_column].duplicated().any()
            ):
                raise ValueError("Catalogue IDs must be non-null and unique")
            products = products.sort_values(cfg.product_key_column).reset_index(drop=True)
            if max_products is not None:
                products = products.head(max_products)
            if products.empty:
                raise ValueError("Empty catalogue")
            self.products = products
            self._weekly = catalogue_weekly_grid(sales, products, cfg)
        else:
            self._weekly = resample_series(
                sales, cfg.resample_period, cfg.sku_column, self.store_column, cfg.quantity_column
            )
        return self._weekly

    def cutoffs(self):
        if self._weekly is None:
            raise ValueError("Load history first")
        ts = self.settings.timeseries
        future = self._weekly.index.max() + pd.Timedelta(weeks=1)
        holdout = (
            pd.Timestamp(ts.holdout_from)
            if ts.holdout_from
            else future - pd.Timedelta(weeks=ts.n_out)
        )
        validation = holdout - pd.Timedelta(weeks=ts.validation_periods)
        if self.settings.data.products_path and (holdout.weekday() != 0 or future.weekday() != 0):
            raise ValueError("Forecast origins must be Mondays")
        if ts.validation_periods != ts.n_out:
            raise ValueError("validation_periods must equal n_out for one validation block")
        if holdout + pd.Timedelta(weeks=ts.n_out) != future:
            raise ValueError("Holdout must be the final n_out complete weeks")
        return validation, holdout, future

    def load_and_preprocess(self):
        started = time.monotonic()
        self.load_data()
        if self._loaded:
            self._series = self._transform(self._weekly)
        else:
            self.fit_preprocessing(self.cutoffs()[0])
        logger.info("Preprocessing seconds=%.3f", time.monotonic() - started)
        return self._series

    def fit_preprocessing(self, cutoff):
        """Discard all stage state; fit on < cutoff, then transform the whole grid."""
        started = time.monotonic()
        if self._weekly is None:
            self.load_data()
        cfg = self.settings.data
        self.cutoff = pd.Timestamp(cutoff)
        train = self._weekly.loc[self._weekly.index < self.cutoff].copy()
        if cfg.filter_sparse_skus:
            train = filter_skus(
                train,
                self.settings.timeseries.window,
                self.settings.timeseries.n_out,
                cfg.max_zeros_ratio,
                cfg.sku_column,
                self.store_column,
                cfg.quantity_column,
            )
        if train.empty:
            raise ValueError("No training rows before cutoff")
        self.sku_to_index = create_sku_index(train, cfg.sku_column)
        self.scaler_manager = ScalerManager()
        self.categorical_encoder = CategoricalEncoder()
        self.clip_bounds = {}
        self.cluster_map = {sku: 0 for sku in self.sku_to_index}
        train["sku_code"] = train[cfg.sku_column].map(self.sku_to_index)
        if cfg.k > 1:
            features = extract_metafeatures(
                train,
                store_column=self.store_column,
                quantity_column=cfg.quantity_column,
                date_column=cfg.date_column,
            )
            matrix = MinMaxScaler().fit_transform(features.iloc[:, 2:])
            labels = KMeans(
                n_clusters=cfg.k, random_state=self.settings.seed, n_init=10
            ).fit_predict(matrix)
            index_to_sku = {i: sku for sku, i in self.sku_to_index.items()}
            self.cluster_map = {
                index_to_sku[int(i)]: int(label) for i, label in zip(features.sku_code, labels)
            }
        train["bins"] = train[cfg.sku_column].map(self.cluster_map)
        for sku, group in train.groupby(cfg.sku_column):
            values = group[cfg.quantity_column].to_numpy(dtype=float)
            self.clip_bounds[sku] = (
                float(values.mean() - 5 * values.std()),
                float(values.mean() + 5 * values.std()),
            )
        train = self._clip(train)
        group_column = cfg.sku_column if cfg.scaling_group == "sku" else "bins"
        for key, group in train.groupby(group_column):
            self.scaler_manager.fit_transform(
                str(key), group[cfg.quantity_column].to_numpy(dtype=float)
            )
        cats = (
            cfg.product_categorical
            if cfg.products_path
            else (
                cfg.categorical_columns or get_categorical_columns(train, exclude_patterns=["sku"])
            )
        )
        cont = cfg.product_continuous if cfg.products_path else []
        # One catalogue row per training SKU; no weighting by sales or future presence.
        self.static = train.groupby(cfg.sku_column, sort=False)[cats + cont].first()
        self.categorical_encoder.fit(self.static, cats, cont)
        self._series = self._transform(self._weekly)
        self._raw_datasets = None
        self.model = None
        elapsed = time.monotonic() - started
        self.preprocessing_seconds.append(elapsed)
        logger.info("Preprocessing cutoff=%s seconds=%.3f", self.cutoff.date(), elapsed)
        return self._series

    def _clip(self, data):
        data = data.copy()
        if self.settings.data.clip_outliers:
            cfg = self.settings.data
            lower = data[cfg.sku_column].map({k: v[0] for k, v in self.clip_bounds.items()})
            upper = data[cfg.sku_column].map({k: v[1] for k, v in self.clip_bounds.items()})
            data[cfg.quantity_column] = np.clip(
                data[cfg.quantity_column].to_numpy(dtype=float), lower, upper
            )
        return data

    def _transform(self, weekly):
        cfg = self.settings.data
        series = self._clip(weekly[weekly[cfg.sku_column].isin(self.sku_to_index)])
        series["sku_code"] = series[cfg.sku_column].map(self.sku_to_index)
        series["bins"] = series[cfg.sku_column].map(self.cluster_map)
        for column in self.static.columns:
            series[column] = series[cfg.sku_column].map(self.static[column])
        series["qty_scaled"] = 0.0
        group_column = cfg.sku_column if cfg.scaling_group == "sku" else "bins"
        for key, positions in series.groupby(group_column).indices.items():
            values = series.iloc[positions][cfg.quantity_column].to_numpy(dtype=float)
            series.iloc[positions, series.columns.get_loc("qty_scaled")] = (
                self.scaler_manager.transform(str(key), values)
            )
        series = self.categorical_encoder.transform(series)
        calculate_time_features(series)
        return series

    def create_datasets(self, validation_origin=None, refit=False):
        if self._series is None:
            raise ValueError("Must call load_and_preprocess first")
        ts = self.settings.timeseries
        origin = pd.Timestamp(validation_origin) if validation_origin is not None else self.cutoff
        datasets = {}
        for label, group in self._series.groupby("bins"):
            windows = create_time_series_data(
                group,
                ["qty_scaled", "sku_code", *TIME_COLUMNS],
                self.categorical_encoder.get_encoded_columns(),
                window=ts.window,
                n_out=ts.n_out,
                store_column=self.store_column,
                target_end_max=self.cutoff - pd.Timedelta(weeks=1)
                if refit
                else origin + pd.Timedelta(weeks=ts.n_out - 1),
                continuous_features=self.continuous_features,
            )
            ends = windows.origins + np.timedelta64(7 * (ts.n_out - 1), "D")
            train = windows.select(ends < self.cutoff.to_datetime64())
            val = None if refit else windows.select(windows.origins == origin.to_datetime64())
            if not len(train.y) or (not refit and len(val.y) != group["sku_code"].nunique()):
                raise ValueError("Insufficient history for disjoint training and validation blocks")
            datasets[label] = (train, val)
        self._raw_datasets = datasets
        return datasets

    def resolved_model_type(self):
        """Return the registry key, resolving the lightweight settings alias."""
        cfg = self.settings.model
        return (
            f"lightweight_{cfg.lightweight_variant}"
            if cfg.model_type == "lightweight"
            else cfg.model_type
        )

    def model_kwargs(self):
        cfg = self.settings.model
        model_type = self.resolved_model_type()
        allowed = inspect.signature(MODEL_REGISTRY[model_type].__init__).parameters
        kwargs = {key: value for key, value in cfg.model_dump().items() if key in allowed}
        kwargs = {
            **kwargs,
            "sku_vocab_size": len(self.sku_to_index),
            "cat_features_dim": self.categorical_encoder.vocab_sizes,
            "cont_features_dim": len(self.continuous_features),
            "past_time_features_dim": 5,
            "future_time_features_dim": 4,
            "n_out": self.settings.timeseries.n_out,
            "max_past_len": self.settings.timeseries.window,
            "max_future_len": self.settings.timeseries.n_out,
        }

        if model_type == "lightweight_tcn":
            kwargs["kernel_size"] = cfg.tcn_kernel_size
        if model_type == "lightweight_mixer":
            kwargs["max_seq_len"] = self.settings.timeseries.window
        return {key: value for key, value in kwargs.items() if key in allowed}

    def build_model(self):
        if self._raw_datasets is None:
            raise ValueError("Must call create_datasets first")
        self.model = ModelWrapper(
            cluster_keys=list(self._raw_datasets),
            model_type=self.resolved_model_type(),
            **self.model_kwargs(),
        ).to(self.device)
        return self.model

    def dataloaders(self):
        cfg = self.settings.training
        return create_dataloaders(
            self._raw_datasets,
            self.categorical_encoder.get_encoded_columns(),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            seed=self.settings.seed,
            device=self.device,
        )

    def train(self, plot=False, plot_dir=None, deadline=None, scheduler_epochs: int | None = None):
        if self.model is None:
            raise ValueError("Must call build_model first")
        if self._raw_datasets is None:
            raise ValueError("Must call create_datasets first")
        cfg = self.settings.training
        train, val, _, _ = self.dataloaders()
        trainer = Trainer(
            self.model,
            train,
            val_dataloaders=val,
            num_epochs=cfg.num_epochs,
            scheduler_epochs=scheduler_epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            flatten_loss=cfg.flatten_loss,
            early_stop=EarlyStopConfig(cfg.early_stop_patience, cfg.early_stop_min_delta),
            device=self.device,
            plot_dir=plot_dir if plot else None,
            deadline=deadline,
        )
        trainer.train(plot_every_n_epochs=5)
        return trainer

    def forecast_batches(self, origin, batch_size=None):
        """Yield deterministic per-cluster inputs from weeks strictly before origin."""
        if self.model is None or self._series is None:
            raise ValueError("A fitted model and transformed history are required")
        cfg, ts = self.settings.data, self.settings.timeseries
        origin = pd.Timestamp(origin)
        if origin.weekday() != 0:
            raise ValueError("Forecast origin must be a Monday")
        weeks = pd.date_range(origin, periods=ts.n_out, freq="W-MON")
        past = pd.date_range(end=origin - pd.Timedelta(weeks=1), periods=ts.window, freq="W-MON")
        future_features = calculate_time_features(weeks).astype(np.float32)
        self.model.eval()
        for cluster, group in self._series.groupby("bins"):
            xs, cats, cont, ids = [], [], [], []
            for sku, history in group.groupby(cfg.sku_column, sort=True):
                if history.index.has_duplicates:
                    raise ValueError("forecast requires a single store per product")
                history = history.reindex(past)
                if history["qty_scaled"].isna().any():
                    raise ValueError(f"Missing input weeks for {sku} at {origin.date()}")
                xs.append(
                    history[["qty_scaled", "sku_code", *TIME_COLUMNS]].to_numpy(dtype=np.float32)
                )
                cats.append(
                    history[self.categorical_encoder.get_encoded_columns()]
                    .iloc[-1]
                    .to_numpy(dtype=np.int64)
                )
                cont.append(history[self.continuous_features].iloc[-1].to_numpy(dtype=np.float32))
                ids.append(sku)
            loader, _ = create_dataloader(
                np.asarray(xs),
                np.asarray(cats),
                np.zeros((len(ids), ts.n_out), dtype=np.float32),
                self.categorical_encoder.get_encoded_columns(),
                batch_size=batch_size or self.settings.training.batch_size,
                shuffle=False,
                future_time=np.repeat(future_features[None], len(ids), axis=0),
                cont=np.asarray(cont),
            )
            yield str(cluster), ids, loader

    def forecast(self, origin):
        """Forecast weekly units, including preparation, transfer and inverse scaling."""
        origin = pd.Timestamp(origin)
        cfg, ts = self.settings.data, self.settings.timeseries
        weeks = pd.date_range(origin, periods=ts.n_out, freq="W-MON")
        records = []
        for cluster, ids, loader in self.forecast_batches(origin):
            predictions = []
            with torch.no_grad():
                for batch in loader:
                    batch = move_batch(batch, self.device)
                    predictions.extend(
                        point_forecast(
                            self.model(str(cluster), **{k: v for k, v in batch.items() if k != "y"})
                        )
                        .cpu()
                        .numpy()
                    )
            for sku, prediction in zip(ids, predictions):
                key = sku if cfg.scaling_group == "sku" else str(cluster)
                quantities = self.scaler_manager.inverse_transform(
                    str(key), np.asarray(prediction, dtype=np.float64)
                )
                if not np.isfinite(quantities).all():
                    raise ValueError("Non-finite forecast quantities")
                for week, qty in zip(weeks, np.round(np.maximum(quantities, 0), 2)):
                    records.append(
                        (sku, origin.date().isoformat(), week.date().isoformat(), float(qty))
                    )
        return (
            pd.DataFrame(records, columns=["product_id", "origin", "week_start", "qty"])
            .sort_values(["product_id", "week_start"])
            .reset_index(drop=True)
        )

    def save(self, path: Path):
        if self.model is None:
            raise ValueError("No model to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_kwargs": self.model._model_kwargs,
                "cluster_keys": list(self.model.models),
                "model_type": self.model.model_type,
            },
            path,
        )
        self.scaler_manager.save(path.parent / "scalers.joblib")
        self.categorical_encoder.save(path.parent / "encoders.joblib")
        joblib.dump(
            {
                "sku_to_index": self.sku_to_index,
                "cluster_map": self.cluster_map,
                "clip_bounds": self.clip_bounds,
                "static": self.static,
                "cutoff": self.cutoff,
            },
            path.parent / "preprocessing.joblib",
        )
        self.settings.to_yaml(path.parent / "settings.yaml")

    def load(self, path: Path):
        checkpoint = load_checkpoint(path, map_location=self.device)
        self.model = ModelWrapper(
            cluster_keys=checkpoint["cluster_keys"],
            model_type=checkpoint["model_type"],
            **checkpoint["model_kwargs"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.scaler_manager = ScalerManager.load(path.parent / "scalers.joblib")
        self.categorical_encoder = CategoricalEncoder.load(path.parent / "encoders.joblib")
        for key, value in joblib.load(path.parent / "preprocessing.joblib").items():
            setattr(self, key, value)
        self._loaded = True

    def predict(self, confidence=0.95, plot=False, plot_dir=None, show_plots=False):
        """Compatibility entry point: point forecasts in units, without invented intervals."""
        if self.model is None:
            raise ValueError("Must load or build model first")
        if self._raw_datasets is None:
            raise ValueError("Must call create_datasets first")
        if self.settings.data.resample_period != "W-MON":
            raise ValueError("predict requires resample_period=W-MON (Monday weeks)")
        return self.forecast(self.cutoffs()[2])

    def evaluate(
        self,
        plot: bool = False,
        plot_dir: Path | None = None,
        show_plots: bool = False,
    ) -> dict:
        """Evaluate model on test data.

        Args:
            plot: Whether to generate and save evaluation plots.
            plot_dir: Directory to save plots. Defaults to model_dir/evaluation.
            show_plots: Whether to display plots interactively.

        Returns:
            Dictionary containing evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Must load or build model first")
        if self._raw_datasets is None:
            raise ValueError("Must call create_datasets first")

        from demand_forecast.core.evaluator import Evaluator
        from demand_forecast.data.dataloader import get_round_robin_iterators
        from demand_forecast.utils.metrics import init_metrics

        self.model.eval()
        encoded_features = self.categorical_encoder.get_encoded_columns()
        train_cfg = self.settings.training

        # Create dataloaders
        train_dls, test_dls, train_dss, test_dss = create_dataloaders(
            self._raw_datasets,
            encoded_features,
            batch_size=train_cfg.batch_size,
            num_workers=train_cfg.num_workers,
            seed=self.settings.seed,
            device=self.device,
        )

        # Count examples
        total_test = sum(len(ds) for ds in test_dss.values())

        # Initialize metrics
        metrics = init_metrics()

        # Determine plot directory
        eval_plot_dir = None
        if plot:
            eval_plot_dir = plot_dir or self.settings.output.model_dir / "evaluation"
            eval_plot_dir.mkdir(parents=True, exist_ok=True)

        # Create evaluator
        evaluator = Evaluator(
            model=self.model,
            criterion=torch.nn.MSELoss(),
            batch_size=train_cfg.batch_size,
            total_examples=total_test,
            flatten_loss=train_cfg.flatten_loss,
            metrics=metrics,
            plot_dir=eval_plot_dir,
        )

        # Get round robin iterator for test data
        _, dls_test = get_round_robin_iterators(train_dls, test_dls)

        # Run evaluation
        logger.info("Running evaluation on test data...")
        val_result = evaluator.validate(dls_test, plot=plot)

        # Compile results
        results = {
            "mse": val_result.mse,
            "mae": val_result.mae,
            "flatten_mse": val_result.flatten_mse,
            "flatten_mae": val_result.flatten_mae,
            "avg_loss": val_result.avg_loss,
            "total_samples": len(val_result.flatten_predictions),
            **val_result.metrics,
        }

        # Calculate additional metrics
        actuals = np.array(val_result.flatten_actuals)
        predictions = np.array(val_result.flatten_predictions)

        # MAPE (Mean Absolute Percentage Error)
        mask = actuals != 0
        if mask.any():
            results["mape"] = float(
                np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
            )
        else:
            results["mape"] = float("nan")

        # RMSE
        results["rmse"] = float(np.sqrt(results["mse"]))

        # Correlation
        if len(actuals) > 1:
            results["correlation"] = float(np.corrcoef(actuals, predictions)[0, 1])
        else:
            results["correlation"] = float("nan")

        # R-squared
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        results["r_squared"] = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else float("nan")

        logger.info(
            f"Evaluation complete: MSE={results['mse']:.4f}, MAE={results['mae']:.4f}, "
            f"RMSE={results['rmse']:.4f}, R²={results['r_squared']:.4f}"
        )

        # Generate additional plots if requested
        if plot and eval_plot_dir:
            from demand_forecast.utils.visualization import plot_prediction_quality

            # Reverse SKU mapping
            index_to_sku = {v: k for k, v in self.sku_to_index.items()}
            skus = [index_to_sku.get(int(s), f"SKU_{s}") for s in val_result.skus]

            plot_prediction_quality(
                actuals=actuals,
                predictions=predictions,
                skus=np.repeat(skus, max(1, len(actuals) // max(1, len(skus)))).tolist(),
                title="Model Evaluation - Prediction Quality",
                save_path=eval_plot_dir / "evaluation_quality.png",
                show=show_plots,
            )

            logger.info(f"Saved evaluation plots to {eval_plot_dir}")

        return results
