"""Metrics initialization and computation utilities."""

import logging

import torch
import torchmetrics

logger = logging.getLogger(__name__)


def init_metrics() -> dict[str, torchmetrics.Metric]:
    """Initialize all available regression metrics from torchmetrics.

    Returns:
        Dictionary mapping metric names to initialized Metric objects.

    Example:
        >>> metrics = init_metrics()
        >>> print(list(metrics.keys()))
        ['MeanSquaredError', 'MeanAbsoluteError', ...]
    """
    # Get all regression metric names
    metrics_names = [v for v in dir(torchmetrics.regression) if v[0].isupper()]
    metrics: dict[str, torchmetrics.Metric] = {}

    # Test tensors to verify metrics work
    target = torch.tensor([[2.5, 5, 4, 8], [3, 5, 2.5, 7]])
    preds = torch.tensor([[3, 5, 2.5, 7], [2.5, 5, 4, 8]])

    for metric_name in metrics_names:
        try:
            metric = getattr(torchmetrics, metric_name)()
            metric(preds, target)
            metrics[metric_name] = metric
        except Exception as e:
            logger.debug(f"Skipping metric {metric_name}: {e}")

    # Remove problematic metrics
    for name in ["KLDivergence", "CosineSimilarity"]:
        metrics.pop(name, None)

    logger.info(f"Initialized {len(metrics)} regression metrics")
    return metrics
