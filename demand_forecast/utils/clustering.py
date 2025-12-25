"""Clustering utilities for time series grouping."""

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Result of K-means clustering optimization."""

    kmeans: KMeans
    best_k: int
    inertia_values: list[float]
    dbi_scores: list[float]


def find_best_k(
    series: np.ndarray,
    k_range: range | None = None,
    random_state: int = 42,
    plot: bool = False,
) -> ClusterResult:
    """Find optimal number of clusters using elbow method and Davies-Bouldin Index.

    Args:
        series: 2D array of features to cluster.
        k_range: Range of K values to try. Default is range(2, min(30, n_samples)).
        random_state: Random seed for reproducibility.
        plot: Whether to display the elbow plot.

    Returns:
        ClusterResult containing the fitted KMeans model and optimization metrics.

    Example:
        >>> from sklearn.datasets import make_blobs
        >>> X, _ = make_blobs(n_samples=100, centers=5)
        >>> result = find_best_k(X, k_range=range(2, 10))
        >>> print(f"Optimal K: {result.best_k}")
    """
    n_samples = len(series)

    # Need at least 2 samples to cluster
    if n_samples < 2:
        logger.warning(f"Only {n_samples} samples, using single cluster")
        kmeans = KMeans(n_clusters=1, random_state=random_state)
        kmeans.fit(series)
        return ClusterResult(
            kmeans=kmeans,
            best_k=1,
            inertia_values=[kmeans.inertia_],
            dbi_scores=[0.0],
        )

    # Cap k_range based on number of samples (need k < n_samples for DBI score)
    max_k = min(30, n_samples)
    if k_range is None:
        k_range = range(2, max_k)
    else:
        # Ensure k_range doesn't exceed sample count
        k_range = range(k_range.start, min(k_range.stop, max_k))

    # Need at least 2 different k values to find elbow
    if len(k_range) < 2:
        best_k = min(2, n_samples - 1) if n_samples > 2 else 1
        logger.warning(f"Too few samples ({n_samples}) for k search, using k={best_k}")
        kmeans = KMeans(n_clusters=best_k, random_state=random_state)
        kmeans.fit(series)
        return ClusterResult(
            kmeans=kmeans,
            best_k=best_k,
            inertia_values=[kmeans.inertia_],
            dbi_scores=[0.0],
        )

    inertia_values: list[float] = []
    dbi_scores: list[float] = []

    # Normalize data if not already normalized
    if series.max() > 1:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(series)
    else:
        data = series

    logger.info(
        f"Finding optimal K in range {k_range.start}-{k_range.stop - 1} (n_samples={n_samples})"
    )

    for k in tqdm(k_range, desc="Finding optimal K"):
        kmeans = KMeans(n_clusters=k, random_state=random_state, max_iter=1000, tol=1e-6)
        kmeans.fit(data)
        inertia_values.append(kmeans.inertia_)
        dbi_scores.append(davies_bouldin_score(data, kmeans.labels_))

    # Determine the "elbow point" using second derivative
    deltas = np.diff(inertia_values)
    elbow_point = int(np.argmin(np.diff(deltas)) + k_range.start)

    logger.info(f"Optimal K determined: {elbow_point}")

    if plot:
        from demand_forecast.utils.visualization import plot_clustering

        plot_clustering(k_range, inertia_values, dbi_scores, elbow_point)

    # Create final KMeans model with optimal K
    final_kmeans = KMeans(
        n_clusters=elbow_point,
        random_state=random_state,
        max_iter=1000,
        tol=1e-6,
    )

    return ClusterResult(
        kmeans=final_kmeans,
        best_k=elbow_point,
        inertia_values=inertia_values,
        dbi_scores=dbi_scores,
    )
