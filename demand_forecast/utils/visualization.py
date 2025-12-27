"""Visualization utilities for demand forecasting."""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def plot_clustering(
    k_range: range,
    inertia_values: list[float],
    dbi_scores: list[float],
    optimal_k: int,
) -> None:
    """Plot elbow method and Davies-Bouldin Index for K selection.

    Args:
        k_range: Range of K values tested.
        inertia_values: Inertia (within-cluster sum of squares) for each K.
        dbi_scores: Davies-Bouldin Index for each K.
        optimal_k: The selected optimal K value.
    """
    _, ax1 = plt.subplots(figsize=(10, 6))

    # Inertia plot
    ax1.plot(list(k_range), inertia_values, marker="o", label="Inertia", color="blue")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # DBI plot
    ax2 = ax1.twinx()
    ax2.plot(list(k_range), dbi_scores, marker="x", label="DBI", color="green")
    ax2.set_ylabel("Davies-Bouldin Index", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    # Highlight optimal K
    ax1.axvline(x=optimal_k, color="red", linestyle="--", label=f"Optimal K = {optimal_k}")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title("Optimal K Selection: Elbow Method & DBI")
    plt.grid()
    plt.show()


def plot_model(model: torch.nn.Module, batch: dict[str, Any]) -> Any:
    """Generate computational graph visualization for a model.

    Args:
        model: PyTorch model to visualize.
        batch: Sample batch of data.

    Returns:
        Graphviz Digraph object for rendering.

    Note:
        Requires torchviz to be installed.
    """
    from torchviz import make_dot

    qty = batch["qty"]
    past_time = batch["past_time"]
    future_time = batch["future_time"]
    sku = batch["sku"]
    cats = {key: value.to(dtype=torch.int32) for key, value in batch["cats"].items()}

    with torch.no_grad():
        model.eval()
        outputs = model("0", qty, past_time, future_time, sku, cats)

    model.train()
    return make_dot(outputs, params=dict(model.named_parameters()))


def plot_predictions(
    actuals: np.ndarray,
    predictions: np.ndarray,
    lower_bound: np.ndarray | None = None,
    upper_bound: np.ndarray | None = None,
    title: str = "Predictions vs. True Values",
) -> None:
    """Plot predictions against actual values with optional confidence bands.

    Args:
        actuals: Array of actual values.
        predictions: Array of predicted values.
        lower_bound: Lower confidence bound (optional).
        upper_bound: Upper confidence bound (optional).
        title: Plot title.
    """
    plt.figure(figsize=(15, 6))
    x = range(len(actuals))

    plt.plot(x, actuals, label="True Values", marker="o", linestyle="-")
    plt.plot(x, predictions, label="Predictions", marker="x", linestyle="--")

    delta = np.abs(predictions - actuals)
    plt.plot(x, delta, label="Delta", marker="x", linestyle=":")

    if lower_bound is not None and upper_bound is not None:
        plt.fill_between(
            x, lower_bound, upper_bound, alpha=0.2, color="grey", label="Confidence band"
        )

    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_validation_results(
    flatten_actuals: list[float],
    flatten_predictions: list[float],
    title: str = "Validation Results",
    save_path: Path | str | None = None,
    show: bool = True,
) -> None:
    """Plot validation results.

    Args:
        flatten_actuals: Flattened actual values.
        flatten_predictions: Flattened predicted values.
        title: Plot title.
        save_path: Path to save the plot. If None, plot is not saved.
        show: Whether to display the plot.
    """
    plt.figure(figsize=(20, 10))
    plt.plot(flatten_actuals, label="Actual", color="blue")
    plt.plot(flatten_predictions, label="Predicted", color="red", linestyle="dashed")
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Quantity")
    plt.legend(loc="upper right")
    plt.grid()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved validation plot to {save_path}")

    if show:
        plt.show()
    plt.close()


def plot_prediction_quality(
    actuals: np.ndarray,
    predictions: np.ndarray,
    skus: list[str] | None = None,
    title: str = "Prediction Quality Analysis",
    save_path: Path | str | None = None,
    show: bool = True,
    max_samples: int = 100,
) -> None:
    """Plot comprehensive prediction quality analysis.

    Creates a multi-panel figure with:
    - Actual vs Predicted scatter plot
    - Residual distribution
    - Prediction comparison line plot
    - Error by SKU (if provided)

    Args:
        actuals: Array of actual values.
        predictions: Array of predicted values.
        skus: List of SKU identifiers (optional).
        title: Overall plot title.
        save_path: Path to save the plot.
        show: Whether to display the plot.
        max_samples: Maximum samples to show in line plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. Scatter plot: Actual vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(actuals, predictions, alpha=0.5, s=20)
    max_val = max(actuals.max(), predictions.max())
    min_val = min(actuals.min(), predictions.min())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Actual vs Predicted")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Calculate metrics
    mse = np.mean((actuals - predictions) ** 2)
    mae = np.mean(np.abs(actuals - predictions))
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
    corr = np.corrcoef(actuals, predictions)[0, 1] if len(actuals) > 1 else 0

    ax1.text(
        0.05,
        0.95,
        f"MSE: {mse:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.1f}%\nCorr: {corr:.3f}",
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # 2. Residual distribution
    ax2 = axes[0, 1]
    residuals = predictions - actuals
    ax2.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax2.axvline(x=0, color="red", linestyle="--", label="Zero error")
    ax2.axvline(
        x=residuals.mean(), color="green", linestyle="-", label=f"Mean: {residuals.mean():.2f}"
    )
    ax2.set_xlabel("Residual (Predicted - Actual)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residual Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Line plot comparison (limited samples)
    ax3 = axes[1, 0]
    n_samples = min(len(actuals), max_samples)
    indices = np.linspace(0, len(actuals) - 1, n_samples, dtype=int)
    x = range(n_samples)

    ax3.plot(x, actuals[indices], label="Actual", marker="o", markersize=3, alpha=0.7)
    ax3.plot(x, predictions[indices], label="Predicted", marker="x", markersize=3, alpha=0.7)
    ax3.fill_between(
        x,
        actuals[indices],
        predictions[indices],
        alpha=0.2,
        color="gray",
        label="Error",
    )
    ax3.set_xlabel("Sample Index")
    ax3.set_ylabel("Value")
    ax3.set_title(f"Prediction Comparison (showing {n_samples} samples)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Error analysis by SKU or percentile
    ax4 = axes[1, 1]
    if skus is not None and len(set(skus)) > 1:
        # Group by SKU and show mean absolute error
        df = pd.DataFrame({"sku": skus, "actual": actuals, "predicted": predictions})
        df["abs_error"] = np.abs(df["predicted"] - df["actual"])
        sku_errors = df.groupby("sku")["abs_error"].mean().sort_values(ascending=False)

        # Show top 20 worst performing SKUs
        top_skus = sku_errors.head(20)
        ax4.barh(range(len(top_skus)), top_skus.values)
        ax4.set_yticks(range(len(top_skus)))
        ax4.set_yticklabels(top_skus.index, fontsize=8)
        ax4.set_xlabel("Mean Absolute Error")
        ax4.set_title("Top 20 SKUs by Error")
        ax4.invert_yaxis()
    else:
        # Show error percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        abs_errors = np.abs(predictions - actuals)
        error_percentiles = [np.percentile(abs_errors, p) for p in percentiles]

        ax4.bar([str(p) for p in percentiles], error_percentiles, color="steelblue")
        ax4.set_xlabel("Percentile")
        ax4.set_ylabel("Absolute Error")
        ax4.set_title("Error Distribution by Percentile")

    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved prediction quality plot to {save_path}")

    if show:
        plt.show()
    plt.close()


def plot_forecast_horizon(
    actuals: np.ndarray,
    predictions: np.ndarray,
    horizon: int,
    title: str = "Forecast by Horizon Step",
    save_path: Path | str | None = None,
    show: bool = True,
) -> None:
    """Plot prediction accuracy by forecast horizon step.

    Args:
        actuals: 2D array of shape (n_samples, horizon).
        predictions: 2D array of shape (n_samples, horizon).
        horizon: Number of forecast steps.
        title: Plot title.
        save_path: Path to save the plot.
        show: Whether to display the plot.
    """
    if actuals.ndim == 1:
        logger.warning("Cannot plot horizon analysis with 1D data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # MAE by horizon step
    ax1 = axes[0]
    mae_by_step = np.mean(np.abs(actuals - predictions), axis=0)
    ax1.bar(range(1, horizon + 1), mae_by_step, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Forecast Step")
    ax1.set_ylabel("Mean Absolute Error")
    ax1.set_title("MAE by Forecast Horizon")
    ax1.grid(True, alpha=0.3)

    # Correlation by horizon step
    ax2 = axes[1]
    corr_by_step = [
        np.corrcoef(actuals[:, i], predictions[:, i])[0, 1] if actuals.shape[0] > 1 else 0
        for i in range(horizon)
    ]
    ax2.plot(range(1, horizon + 1), corr_by_step, marker="o", color="green")
    ax2.set_xlabel("Forecast Step")
    ax2.set_ylabel("Correlation")
    ax2.set_title("Correlation by Forecast Horizon")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved forecast horizon plot to {save_path}")

    if show:
        plt.show()
    plt.close()


def plot_training_history(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    title: str = "Training History",
    save_path: Path | str | None = None,
    show: bool = True,
) -> None:
    """Plot training loss history.

    Args:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch (optional).
        title: Plot title.
        save_path: Path to save the plot.
        show: Whether to display the plot.
    """
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Training Loss", marker="o")

    if val_losses:
        plt.plot(epochs, val_losses, label="Validation Loss", marker="s")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved training history plot to {save_path}")

    if show:
        plt.show()
    plt.close()


def save_prediction_report(
    predictions_df: pd.DataFrame,
    output_dir: Path | str,
    prefix: str = "predictions",
) -> dict[str, Path]:
    """Generate and save a complete prediction report with visualizations.

    Args:
        predictions_df: DataFrame with columns: sku, prediction, lower_bound, upper_bound.
        output_dir: Directory to save the report files.
        prefix: Prefix for output files.

    Returns:
        Dictionary mapping report names to file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # 1. Summary statistics
    summary = {
        "total_skus": len(predictions_df),
        "mean_prediction": predictions_df["prediction"].mean(),
        "median_prediction": predictions_df["prediction"].median(),
        "std_prediction": predictions_df["prediction"].std(),
        "min_prediction": predictions_df["prediction"].min(),
        "max_prediction": predictions_df["prediction"].max(),
    }

    if "cluster" in predictions_df.columns:
        summary["num_clusters"] = predictions_df["cluster"].nunique()

    # Save summary
    summary_path = output_dir / f"{prefix}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Prediction Summary Report\n")
        f.write("=" * 50 + "\n\n")
        for key, value in summary.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
    saved_files["summary"] = summary_path

    # 2. Prediction distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1 = axes[0]
    ax1.hist(predictions_df["prediction"], bins=50, edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Prediction Value")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Prediction Distribution")
    ax1.axvline(predictions_df["prediction"].mean(), color="red", linestyle="--", label="Mean")
    ax1.axvline(predictions_df["prediction"].median(), color="green", linestyle="-", label="Median")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # By cluster if available
    ax2 = axes[1]
    if "cluster" in predictions_df.columns:
        cluster_means = predictions_df.groupby("cluster")["prediction"].mean().sort_values()
        ax2.barh(cluster_means.index.astype(str), cluster_means.values, color="steelblue")
        ax2.set_xlabel("Mean Prediction")
        ax2.set_ylabel("Cluster")
        ax2.set_title("Mean Prediction by Cluster")
    else:
        # Top predictions
        top_preds = predictions_df.nlargest(20, "prediction")
        ax2.barh(range(len(top_preds)), top_preds["prediction"].values)
        ax2.set_yticks(range(len(top_preds)))
        ax2.set_yticklabels(top_preds["sku"].values, fontsize=8)
        ax2.set_xlabel("Prediction")
        ax2.set_title("Top 20 Predictions")
        ax2.invert_yaxis()

    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    dist_path = output_dir / f"{prefix}_distribution.png"
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    saved_files["distribution"] = dist_path

    # 3. Confidence intervals plot (if available)
    if "lower_bound" in predictions_df.columns and "upper_bound" in predictions_df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))

        # Sample up to 50 SKUs for readability
        sample_df = predictions_df.head(50)
        x = range(len(sample_df))

        ax.errorbar(
            x,
            sample_df["prediction"],
            yerr=[
                sample_df["prediction"] - sample_df["lower_bound"],
                sample_df["upper_bound"] - sample_df["prediction"],
            ],
            fmt="o",
            capsize=3,
            capthick=1,
            alpha=0.7,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(sample_df["sku"], rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("SKU")
        ax.set_ylabel("Prediction")
        ax.set_title("Predictions with Confidence Intervals (first 50 SKUs)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        ci_path = output_dir / f"{prefix}_confidence_intervals.png"
        plt.savefig(ci_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_files["confidence_intervals"] = ci_path

    logger.info(f"Saved prediction report to {output_dir}")
    return saved_files
