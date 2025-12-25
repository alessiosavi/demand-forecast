"""Checkpoint loading utilities with backward compatibility."""

import logging
import random

import dill as pickle
import numpy as np
import torch

logger = logging.getLogger(__name__)


def _seed_worker_stub(worker_id: int) -> None:
    """Stub function to replace missing seed_worker references in old checkpoints."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class _BackwardCompatibleUnpickler(pickle.Unpickler):
    """Custom unpickler that handles missing local functions from old checkpoints."""

    def find_class(self, module: str, name: str):
        """Override to handle missing seed_worker references."""
        # Handle the specific case of missing seed_worker
        if "seed_worker" in name:
            logger.warning(
                f"Found reference to old local function '{name}' in checkpoint. "
                "Using stub replacement."
            )
            return _seed_worker_stub

        return super().find_class(module, name)


def load_checkpoint(
    path: str,
    map_location: torch.device | str | None = None,
) -> any:
    """Load a checkpoint with backward compatibility for old formats.

    Handles checkpoints that contain references to local functions
    that no longer exist (e.g., seed_worker defined inside create_dataloader).

    Args:
        path: Path to the checkpoint file.
        map_location: Device to map tensors to.

    Returns:
        Loaded checkpoint object.
    """
    try:
        # First try normal loading
        return torch.load(path, map_location=map_location)
    except AttributeError as e:
        if "seed_worker" in str(e) or "local object" in str(e):
            logger.warning(
                f"Checkpoint contains unpicklable references: {e}. "
                "Attempting backward-compatible loading..."
            )
            # Use custom unpickler
            with open(path, "rb") as f:
                unpickler = _BackwardCompatibleUnpickler(f)
                checkpoint = unpickler.load()

            # Move tensors to the right device if needed
            if map_location is not None:
                checkpoint = _map_location_recursive(checkpoint, map_location)

            return checkpoint
        raise


def _map_location_recursive(obj: any, device: torch.device | str) -> any:
    """Recursively move tensors in an object to a device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: _map_location_recursive(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_map_location_recursive(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_map_location_recursive(v, device) for v in obj)
    elif isinstance(obj, torch.nn.Module):
        return obj.to(device)
    return obj
