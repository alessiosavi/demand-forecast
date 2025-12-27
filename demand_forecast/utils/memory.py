"""Memory management utilities."""

import gc
import logging

import torch

logger = logging.getLogger(__name__)


def collect_garbage() -> int:
    """Force garbage collection and clear CUDA cache.

    Performs multiple garbage collection passes and clears PyTorch CUDA cache
    to free up memory during training.

    Returns:
        Total number of objects collected.

    Example:
        >>> collected = collect_garbage()
        >>> print(f"Collected {collected} objects")
    """
    total_collected = 0

    for _ in range(3):
        total_collected += gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if total_collected > 0:
        logger.debug(f"Garbage collection freed {total_collected} objects")

    return total_collected
