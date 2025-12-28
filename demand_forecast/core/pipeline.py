"""Main orchestration pipeline for demand forecasting."""

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from demand_forecast.config.settings import Settings
from demand_forecast.core.trainer import EarlyStopConfig, Trainer
from demand_forecast.data.dataloader import create_dataloaders
from demand_forecast.data.dataset import create_time_series_data
from demand_forecast.data.feature_engineering import (
    CategoricalEncoder,
    extract_metafeatures,
    get_categorical_columns,
)
from demand_forecast.data.loader import load_sales_data
from demand_forecast.data.preprocessor import (
    ScalerManager,
    create_sku_index,
    filter_skus,
    resample_series,
    scale_by_group,
)
from demand_forecast.models.wrapper import ModelWrapper
from demand_forecast.utils.checkpoint import load_checkpoint
from demand_forecast.utils.clustering import find_best_k
from demand_forecast.utils.memory import collect_garbage
from demand_forecast.utils.metrics import init_metrics
from demand_forecast.utils.outliers import remove_outliers
from demand_forecast.utils.time_features import calculate_time_features

logger = logging.getLogger(__name__)


class ForecastPipeline:
    """End-to-end pipeline for demand forecasting.

    Orchestrates data loading, preprocessing, training, and evaluation.

    Attributes:
        settings: Pipeline configuration.
        device: Computation device.
        scaler_manager: Manages scalers per cluster.
        categorical_encoder: Encodes categorical features.
        sku_to_index: Mapping from SKU names to indices.
        model: Trained model.
    """

    def __init__(self, settings: Settings):
        """Initialize the pipeline.

        Args:
            settings: Pipeline configuration.
        """
        self.settings = settings
        self.device = self._get_device()

        # Set random seeds
        self._set_seeds(settings.seed)

        # Initialize state
        self.scaler_manager = ScalerManager()
        self.categorical_encoder = CategoricalEncoder()
        self.sku_to_index: dict[str, int] = {}
        self.model: ModelWrapper | None = None
        self._series: pd.DataFrame | None = None
        self._raw_datasets: dict[int, tuple] | None = None

    def _get_device(self) -> torch.device:
        """Get computation device."""
        if self.settings.device:
            return torch.device(self.settings.device)

        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch, "accelerator") and torch.accelerator.is_available():
            return torch.accelerator.current_accelerator()
        else:
            return torch.device("cpu")

    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def load_and_preprocess(self) -> pd.DataFrame:
        """Load and preprocess data.

        Returns:
            Preprocessed DataFrame.
        """
        cfg = self.settings.data
        ts_cfg = self.settings.timeseries

        # Load data
        df = load_sales_data(
            path=cfg.input_path,
            date_column=cfg.date_column,
            sku_column=cfg.sku_column,
            quantity_column=cfg.quantity_column,
            store_column=cfg.store_column,
            product_id_column=cfg.product_id_column,
            sales_qty_column=cfg.sales_qty_column,
        )

        # Resample to specified period
        series = resample_series(
            df,
            resample_period=cfg.resample_period,
            sku_column=cfg.sku_column,
            store_column=cfg.store_column,
            quantity_column=cfg.quantity_column,
        )

        # Filter by data quality
        series = filter_skus(
            series,
            window=ts_cfg.window,
            n_out=ts_cfg.n_out,
            max_zeros_ratio=cfg.max_zeros_ratio,
            sku_column=cfg.sku_column,
            store_column=cfg.store_column,
            quantity_column=cfg.quantity_column,
        )

        # Create SKU index
        self.sku_to_index = create_sku_index(df, cfg.sku_column)
        series["sku_code"] = series[cfg.sku_column].map(self.sku_to_index)

        # Extract metafeatures and cluster
        series = self._add_clustering(series)

        # Scale quantity by cluster
        series = scale_by_group(
            series,
            self.scaler_manager,
            quantity_column=cfg.quantity_column,
        )

        # Remove outliers
        series = self._remove_outliers(series)

        # Encode categorical features
        series = self._encode_categoricals(series)

        # Add time features
        calculate_time_features(series)
        calculate_time_features(series, "future")

        series = series.convert_dtypes()
        self._series = series

        logger.info(f"Preprocessing complete: {len(series)} samples")
        return series

    def _add_clustering(self, series: pd.DataFrame) -> pd.DataFrame:
        """Add cluster assignments to series."""
        cfg = self.settings.data
        output_cfg = self.settings.output
        cache_path = output_cfg.metafeatures_path

        feature_df = extract_metafeatures(
            series,
            store_column=cfg.store_column,
            quantity_column=cfg.quantity_column,
            date_column=cfg.date_column,
            cache_path=cache_path if cache_path.exists() else None,
        )

        if not cache_path.exists():
            feature_df.to_csv(cache_path, index=False)

        # Find optimal clusters
        scaler = MinMaxScaler()
        x = scaler.fit_transform(feature_df[feature_df.columns[2:]])
        cluster_result = find_best_k(x, plot=False)

        logger.info(f"Optimal K: {cluster_result.best_k}")

        feature_df["bins"] = cluster_result.kmeans.fit_predict(x)

        # Merge cluster assignments (preserve index)
        series = series.reset_index(drop=False)
        series = series.merge(
            feature_df[["sku_code", cfg.store_column, "bins"]],
            on=["sku_code", cfg.store_column],
            how="left",
        )
        # Fill any missing cluster assignments with 0
        series["bins"] = series["bins"].fillna(0).astype(int)
        # Restore DatetimeIndex
        if cfg.date_column in series.columns:
            series = series.set_index(cfg.date_column)
        elif "index" in series.columns:
            series = series.set_index("index")

        return series

    def _remove_outliers(self, series: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from scaled quantities."""
        n_outliers = 0
        cfg = self.settings.data

        for sku in series[cfg.sku_column].unique():
            mask = series[cfg.sku_column] == sku
            data, has_outliers = remove_outliers(series.loc[mask, "qty_scaled"], n=5)
            if has_outliers:
                n_outliers += 1
                series.loc[mask, "qty_scaled"] = data

        logger.info(f"Processed outliers in {n_outliers} SKUs")
        return series

    def _encode_categoricals(self, series: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        cfg = self.settings.data

        # Use configured categorical columns or auto-detect
        if cfg.categorical_columns:
            categorical_columns = cfg.categorical_columns
        else:
            categorical_columns = get_categorical_columns(series, exclude_patterns=["sku"])

        # Use configured onehot columns or default to store column + is_promo_day if present
        if cfg.onehot_columns is not None:
            onehot_columns = cfg.onehot_columns
        else:
            onehot_columns = [cfg.store_column]
            if "is_promo_day" in series.columns:
                onehot_columns.append("is_promo_day")

        series = self.categorical_encoder.fit_transform(series, categorical_columns, onehot_columns)

        return series

    def create_datasets(self) -> dict[int, tuple]:
        """Create time series datasets for training.

        Returns:
            Dictionary mapping cluster bins to dataset tuples.
        """
        if self._series is None:
            raise ValueError("Must call load_and_preprocess first")

        ts_cfg = self.settings.timeseries
        encoded_features = self.categorical_encoder.get_encoded_columns()

        # Build series features
        series_features = ["qty_scaled", "sku_code"]
        series_features.extend(
            [c for c in self._series.select_dtypes(np.number) if "cos" in c or "sin" in c]
        )

        cfg = self.settings.data
        raw_datasets = {}
        for label, group in self._series.groupby("bins"):
            raw_datasets[label] = create_time_series_data(
                group,
                series_features,
                encoded_features,
                test_size=ts_cfg.test_size,
                window=ts_cfg.window,
                n_out=ts_cfg.n_out,
                store_column=cfg.store_column,
            )

        self._raw_datasets = raw_datasets
        return raw_datasets

    def build_model(self) -> ModelWrapper:
        """Build the model.

        Returns:
            Initialized ModelWrapper.
        """
        if self._raw_datasets is None:
            raise ValueError("Must call create_datasets first")

        cfg = self.settings.model
        ts_cfg = self.settings.timeseries
        encoded_features = self.categorical_encoder.get_encoded_columns()

        # Get feature dimensions from first dataset
        # first_dataset = next(iter(self._raw_datasets.values()))
        # sample_x = first_dataset[0][0]

        # Infer dimensions
        time_features_dim = 4  # sin/cos for week and month
        qty_features_dim = 1

        # Get categorical feature shapes (hardcoded for now, should be dynamic)
        # This would need to be extracted from the actual data
        cat_features_shapes = {col: 20 for col in encoded_features}

        model = ModelWrapper(
            cluster_keys=list(self._raw_datasets.keys()),
            sku_vocab_size=len(self.sku_to_index),
            sku_emb_dim=cfg.sku_emb_dim,
            cat_features_dim=cat_features_shapes,
            cat_emb_dims=cfg.cat_emb_dims,
            past_time_features_dim=time_features_dim + qty_features_dim,
            future_time_features_dim=time_features_dim,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            n_out=ts_cfg.n_out,
            max_past_len=ts_cfg.window,
            max_future_len=ts_cfg.n_out,
        )

        model.to(self.device)
        self.model = model

        logger.info(f"Built model with {model.num_models} cluster models")
        return model

    def train(self, plot: bool = False, plot_dir: Path | None = None) -> None:
        """Train the model.

        Args:
            plot: Whether to generate and save validation plots during training.
            plot_dir: Directory to save plots. Defaults to model_dir/training_plots.
        """
        if self.model is None:
            raise ValueError("Must call build_model first")
        if self._raw_datasets is None:
            raise ValueError("Must call create_datasets first")

        train_cfg = self.settings.training
        encoded_features = self.categorical_encoder.get_encoded_columns()

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
        total_train = sum(len(ds) for ds in train_dss.values())
        total_test = sum(len(ds) for ds in test_dss.values())

        # Initialize metrics
        metrics = init_metrics()

        # Determine plot directory
        trainer_plot_dir = None
        if plot:
            trainer_plot_dir = plot_dir or self.settings.output.model_dir / "training_plots"
            trainer_plot_dir.mkdir(parents=True, exist_ok=True)

        # Create trainer
        trainer = Trainer(
            model=self.model,
            train_dataloaders=train_dls,
            test_dataloaders=test_dls,
            num_epochs=train_cfg.num_epochs,
            batch_size=train_cfg.batch_size,
            learning_rate=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            early_stop=EarlyStopConfig(
                patience=train_cfg.early_stop_patience,
                min_delta=train_cfg.early_stop_min_delta,
            ),
            flatten_loss=train_cfg.flatten_loss,
            device=self.device,
            plot_dir=trainer_plot_dir,
        )

        trainer.set_example_counts(total_train, total_test)

        collect_garbage()

        # Train
        trainer.train(metrics=metrics)

        if plot and trainer_plot_dir:
            logger.info(f"Saved training plots to {trainer_plot_dir}")

    def save(self, path: Path) -> None:
        """Save model and artifacts.

        Args:
            path: Path to save model.
        """
        if self.model is None:
            raise ValueError("No model to save")

        torch.save(self.model, path)
        logger.info(f"Saved model to {path}")

        # Save scalers and encoders
        self.scaler_manager.save(path.parent / "scalers.joblib")
        self.categorical_encoder.save(path.parent / "encoders.joblib")

    def load(self, path: Path) -> None:
        """Load model and artifacts.

        Args:
            path: Path to load model from.
        """
        self.model = load_checkpoint(path, map_location=self.device)
        logger.info(f"Loaded model from {path}")

        # Load scalers and encoders if they exist
        scalers_path = path.parent / "scalers.joblib"
        if scalers_path.exists():
            self.scaler_manager = ScalerManager.load(scalers_path)

        encoders_path = path.parent / "encoders.joblib"
        if encoders_path.exists():
            self.categorical_encoder = CategoricalEncoder.load(encoders_path)

    def predict(
        self,
        confidence: float = 0.95,
        plot: bool = False,
        plot_dir: Path | None = None,
        show_plots: bool = False,
    ) -> pd.DataFrame:
        """Generate predictions for all SKUs.

        Args:
            confidence: Confidence interval level (e.g., 0.95 for 95% CI).
            plot: Whether to generate and save prediction quality plots.
            plot_dir: Directory to save plots. Defaults to model_dir/plots.
            show_plots: Whether to display plots interactively.

        Returns:
            DataFrame with columns: sku, cluster, prediction, lower_bound, upper_bound.
        """
        if self.model is None:
            raise ValueError("Must load or build model first")
        if self._raw_datasets is None:
            raise ValueError("Must call create_datasets first")

        self.model.eval()
        results = []
        all_actuals = []
        all_predictions = []
        all_skus = []

        encoded_features = self.categorical_encoder.get_encoded_columns()
        train_cfg = self.settings.training

        # Create dataloaders for prediction (using test data)
        train_dls, test_dls, _, _ = create_dataloaders(
            raw_datasets=self._raw_datasets,
            encoded_categorical_features=encoded_features,
            batch_size=train_cfg.batch_size,
            num_workers=train_cfg.num_workers,
            seed=self.settings.seed,
            device=self.device,
        )

        # Reverse SKU mapping
        index_to_sku = {v: k for k, v in self.sku_to_index.items()}

        with torch.no_grad():
            for cluster_id, dl in test_dls.items():
                cluster_preds = []
                cluster_actuals = []
                cluster_skus = []

                for batch in dl:
                    # Move batch to device
                    qty = batch["qty"].to(self.device)
                    past_time = batch["past_time"].to(self.device)
                    future_time = batch["future_time"].to(self.device)
                    sku = batch["sku"].to(self.device)
                    cats = {k: v.to(self.device) for k, v in batch["cats"].items()}

                    # Forward pass
                    outputs = self.model(
                        n=cluster_id,
                        qty=qty,
                        past_time=past_time,
                        future_time=future_time,
                        sku=sku,
                        cats=cats,
                    )

                    # Flatten predictions (sum over forecast horizon)
                    preds = outputs.squeeze(-1).sum(dim=-1).cpu().numpy()
                    actuals = batch["y"].sum(dim=-1).cpu().numpy()
                    sku_indices = batch["sku"].cpu().numpy()

                    cluster_preds.extend(preds.tolist())
                    cluster_actuals.extend(actuals.tolist())
                    cluster_skus.extend([index_to_sku.get(int(s), f"SKU_{s}") for s in sku_indices])

                # Calculate confidence intervals using prediction std
                preds_array = np.array(cluster_preds)
                std = preds_array.std() if len(preds_array) > 1 else 0
                z_score = 1.96 if confidence == 0.95 else 1.645  # 95% or 90% CI

                for sku_name, pred, actual in zip(cluster_skus, cluster_preds, cluster_actuals):
                    results.append(
                        {
                            "sku": sku_name,
                            "cluster": cluster_id,
                            "prediction": pred,
                            "actual": actual,
                            "lower_bound": max(0, pred - z_score * std),
                            "upper_bound": pred + z_score * std,
                        }
                    )

                all_predictions.extend(cluster_preds)
                all_actuals.extend(cluster_actuals)
                all_skus.extend(cluster_skus)

        df = pd.DataFrame(results)
        logger.info(f"Generated predictions for {len(df)} SKU samples")

        # Generate plots if requested
        if plot:
            self._generate_prediction_plots(
                actuals=np.array(all_actuals),
                predictions=np.array(all_predictions),
                skus=all_skus,
                predictions_df=df,
                plot_dir=plot_dir,
                show_plots=show_plots,
            )

        return df

    def _generate_prediction_plots(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
        skus: list[str],
        predictions_df: pd.DataFrame,
        plot_dir: Path | None = None,
        show_plots: bool = False,
    ) -> None:
        """Generate and save prediction quality plots.

        Args:
            actuals: Array of actual values.
            predictions: Array of predicted values.
            skus: List of SKU identifiers.
            predictions_df: DataFrame with prediction results.
            plot_dir: Directory to save plots.
            show_plots: Whether to display plots interactively.
        """
        from demand_forecast.utils.visualization import (
            plot_prediction_quality,
            save_prediction_report,
        )

        if plot_dir is None:
            plot_dir = self.settings.output.model_dir / "plots"

        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Generate quality analysis plot
        plot_prediction_quality(
            actuals=actuals,
            predictions=predictions,
            skus=skus,
            title="Prediction Quality Analysis",
            save_path=plot_dir / "prediction_quality.png",
            show=show_plots,
        )

        # Generate prediction report
        save_prediction_report(
            predictions_df=predictions_df,
            output_dir=plot_dir,
            prefix="predictions",
        )

        logger.info(f"Saved prediction plots to {plot_dir}")

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
                skus=skus,
                title="Model Evaluation - Prediction Quality",
                save_path=eval_plot_dir / "evaluation_quality.png",
                show=show_plots,
            )

            logger.info(f"Saved evaluation plots to {eval_plot_dir}")

        return results
