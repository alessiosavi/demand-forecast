"""Task-local OOM back-off for validation, prediction and training micro-batches.

Training halves micro-batches down to one row while preserving effective batch
and step counts; its limit survives checkpoints and is shared with task-local refits.
"""

import gc
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class BatchLimit:
    size: int | None = None


_limits: ContextVar[dict[str, BatchLimit] | None] = ContextVar("inference_limits", default=None)


@contextmanager
def task_batch_limits():
    """Share reductions across windows/refits, resetting at the next task."""
    token = _limits.set({})
    try:
        yield
    finally:
        _limits.reset(token)


def batch_limit(kind: str) -> BatchLimit:
    limits = _limits.get()
    return BatchLimit() if limits is None else limits.setdefault(kind, BatchLimit())


def is_oom_error(exc: RuntimeError) -> bool:
    """Recognize typed CUDA OOMs and message-based runtime OOMs consistently."""
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


@torch.no_grad()
def _predict(model, cpu_batch):
    # Keep device tensors in this frame so unwinding releases them before retry.
    batch = cpu_batch.to(model.fallback_level.device)
    return model.predict(batch).cpu().numpy()


def predict_batches(model, cpu_batch, limit: BatchLimit):
    """Yield successful slices in order; preserve CPU padding and retry the same rows.

    Unchanged batches take the original prediction path. Different batch shapes
    can select different floating-point kernels; bitwise parity after OOM is not
    guaranteed, although the mathematical predictions are unchanged.
    """
    start = 0
    while start < cpu_batch.size:
        size = min(limit.size or cpu_batch.size, cpu_batch.size - start)
        chunk = (
            cpu_batch
            if start == 0 and size == cpu_batch.size
            else cpu_batch._map_tensors(lambda tensor: tensor[start : start + size])
        )
        try:
            predictions = _predict(model, chunk)
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            if size <= 8:
                raise
        else:
            yield start, chunk, predictions
            start += size
            continue
        # Outside the exception scope: the traceback no longer retains activations.
        gc.collect()
        torch.cuda.empty_cache()
        limit.size = max(8, size // 2)
        logger.warning("Inference OOM: reducing batch size from %d to %d", size, limit.size)
