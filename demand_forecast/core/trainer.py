"""Training loop for demand forecasting models."""

import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn as nn
from torch import optim

from demand_forecast.core.evaluator import Evaluator
from demand_forecast.models.losses import compute_loss, point_forecast
from demand_forecast.utils.checkpoint import load_checkpoint as safe_load_checkpoint

logger = logging.getLogger(__name__)


class TrainingCallback(Protocol):
    """Protocol for training callbacks."""

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Called at the start of each epoch."""
        ...

    def on_epoch_end(self, epoch: int, loss: float, metrics: dict[str, float]) -> None:
        """Called at the end of each epoch."""
        ...

    def on_batch_end(self, batch: int, loss: float) -> None:
        """Called at the end of each batch."""
        ...


@dataclass
class EarlyStopConfig:
    """Configuration for early stopping."""

    patience: int = 3
    min_delta: float = 1.0


@dataclass
class TrainingState:
    """State of training process."""

    best_metric: float = float("inf")
    epochs_no_improve: int = 0
    best_model_state: dict[str, Any] | None = None
    current_epoch: int = 0


@dataclass
class Trainer:
    """Trainer for demand forecasting models.

    Handles the training loop, validation, early stopping, and checkpointing.

    Attributes:
        model: The model to train.
        train_dataloaders: Dictionary of training dataloaders per cluster.
        test_dataloaders: Dictionary of test dataloaders per cluster.
        num_epochs: Number of training epochs.
        scheduler_epochs: Cosine schedule horizon (defaults to num_epochs).
        batch_size: Batch size.
        learning_rate: Learning rate.
        weight_decay: Weight decay for AdamW.
        early_stop: Early stopping configuration.
        flatten_loss: Whether to use flattened loss.
        device: Training device.
        callbacks: List of training callbacks.
    """

    model: nn.Module
    train_dataloaders: dict[str, Any]
    val_dataloaders: dict[str, Any] = field(default_factory=dict)
    test_dataloaders: dict[str, Any] = field(default_factory=dict)
    num_epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    early_stop: EarlyStopConfig = field(default_factory=EarlyStopConfig)
    flatten_loss: bool = False
    device: torch.device | None = None
    callbacks: list[TrainingCallback] = field(default_factory=list)
    plot_dir: Path | None = None

    deadline: float | None = None
    history: dict[str, list[float]] = field(
        default_factory=lambda: {"train_loss": [], "val_loss": []}
    )
    epoch_seconds: list[float] = field(default_factory=list)
    best_epoch: int = 0
    budget_exceeded: bool = False
    stop_reason: str = "max_epochs"
    scheduler_epochs: int | None = None

    # Internal state
    _optimizers: dict[str, optim.Optimizer] = field(default_factory=dict, init=False)
    _schedulers: dict[str, optim.lr_scheduler.LRScheduler] = field(default_factory=dict, init=False)
    _criterion: nn.Module = field(default_factory=nn.MSELoss, init=False)
    _state: TrainingState = field(default_factory=TrainingState, init=False)
    _evaluator: Evaluator | None = field(default=None, init=False)
    _total_train_examples: int = field(default=0, init=False)
    _total_test_examples: int = field(default=0, init=False)

    def __post_init__(self):
        """Initialize optimizers, schedulers, and evaluator."""
        if self.device is None:
            self.device = torch.device("cpu")
        if self.scheduler_epochs is None:
            self.scheduler_epochs = self.num_epochs

        # Create optimizer and scheduler per cluster (use actual cluster keys)
        for cluster_key in self.train_dataloaders.keys():
            key = str(cluster_key)
            self._optimizers[key] = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            self._schedulers[key] = optim.lr_scheduler.CosineAnnealingLR(
                self._optimizers[key], T_max=self.scheduler_epochs
            )

        self._criterion = nn.MSELoss()

    def set_example_counts(self, train_count: int, test_count: int) -> None:
        """Set the total number of training and test examples.

        Args:
            train_count: Total training examples.
            test_count: Total test examples.
        """
        self._total_train_examples = train_count
        self._total_test_examples = test_count

    def _forward_step(
        self,
        model_idx: str,
        batch: dict[str, torch.Tensor],
    ) -> tuple:
        """Execute a single forward step.

        Args:
            model_idx: Cluster model index.
            batch: Batch of data.

        Returns:
            Tuple of (loss, outputs, targets, flatten_outputs, flatten_targets).
        """
        batch = move_batch(batch, self.device)
        targets = batch["y"]
        extra = {"cont": batch["cont"]} if "cont" in batch else {}
        outputs = self.model(
            model_idx,
            batch["qty"],
            batch["past_time"],
            batch["future_time"],
            batch["sku"],
            batch["cats"],
            **extra,
        )
        outputs = point_forecast(outputs)
        loss, flatten_outputs, flatten_targets = compute_loss(
            outputs, targets, self._criterion, self.flatten_loss
        )

        return loss, outputs, targets, flatten_outputs, flatten_targets

    def train(
        self,
        metrics: dict[str, Any] | None = None,
        plot_every_n_epochs: int = 3,
    ) -> None:
        """Run the training loop.

        Args:
            metrics: Optional dictionary of torchmetrics to compute.
            plot_every_n_epochs: Plot validation results every N epochs.
        """
        self._evaluator = Evaluator(
            model=self.model,
            criterion=self._criterion,
            batch_size=self.batch_size,
            total_examples=sum(len(dl.dataset) for dl in self.val_dataloaders.values()),
            flatten_loss=self.flatten_loss,
            metrics=metrics or {},
            plot_dir=self.plot_dir,
        )
        for epoch in range(self.num_epochs):
            started = time.monotonic()
            self._state.current_epoch = epoch
            for callback in self.callbacks:
                callback.on_epoch_start(epoch, self.num_epochs)
            self.model.train()
            total_loss, count = 0.0, 0
            for key, loader in self.train_dataloaders.items():
                for i, batch in enumerate(loader):
                    loss, _, targets, _, _ = self._forward_step(key, batch)
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite training loss")
                    self._optimizers[key].zero_grad()
                    loss.backward()
                    self._optimizers[key].step()
                    total_loss += loss.item() * len(targets)
                    count += len(targets)
                    for callback in self.callbacks:
                        callback.on_batch_end(i, loss.item())
            if not count:
                raise ValueError("No training windows before cutoff")
            avg_loss = total_loss / count
            self.history["train_loss"].append(avg_loss)
            for scheduler in self._schedulers.values():
                scheduler.step()
            should_stop = False
            if self.val_dataloaders:
                batches = ({key: batch} for key, dl in self.val_dataloaders.items() for batch in dl)
                result = self._evaluator.validate(
                    batches,
                    plot=self.plot_dir is not None
                    and ((epoch + 1) % plot_every_n_epochs == 0 or epoch == self.num_epochs - 1),
                    epoch=epoch + 1,
                )
                metric = result.avg_loss
                self.history["val_loss"].append(metric)
                if metric < self._state.best_metric:
                    improvement = self._state.best_metric - metric
                    self._state.best_model_state = deepcopy(self.model.state_dict())
                    self._state.best_metric = metric
                    self.best_epoch = epoch + 1
                    if improvement > self.early_stop.min_delta:
                        self._state.epochs_no_improve = 0
                    else:
                        self._state.epochs_no_improve += 1
                else:
                    self._state.epochs_no_improve += 1
                should_stop = self._state.epochs_no_improve >= self.early_stop.patience
            else:
                self.best_epoch = epoch + 1
            elapsed = time.monotonic() - started
            self.epoch_seconds.append(elapsed)
            logger.info(
                "Epoch %d/%d: train_loss=%.6f seconds/epoch=%.3f",
                epoch + 1,
                self.num_epochs,
                avg_loss,
                elapsed,
            )
            for callback in self.callbacks:
                callback.on_epoch_end(
                    epoch,
                    avg_loss,
                    {"val_loss": self.history["val_loss"][-1]} if self.val_dataloaders else {},
                )
            if self.deadline is not None and time.monotonic() >= self.deadline:
                self.budget_exceeded = True
                self.stop_reason = "budget"
                logger.warning("Budget exceeded; stopping training after current epoch")
                break
            if should_stop:
                self.stop_reason = "early_stop"
                break
        # Restore on exhaustion as well as early stopping. Refit keeps its final weights.
        if self.val_dataloaders and self._state.best_model_state is not None:
            self.model.load_state_dict(self._state.best_model_state)
            if self.plot_dir is not None:
                batches = ({key: batch} for key, dl in self.val_dataloaders.items() for batch in dl)
                self._evaluator.validate(batches, plot=True, epoch="best")

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_states": {k: v.state_dict() for k, v in self._optimizers.items()},
            "scheduler_states": {k: v.state_dict() for k, v in self._schedulers.items()},
            "training_state": {
                "best_metric": self._state.best_metric,
                "epochs_no_improve": self._state.epochs_no_improve,
                "current_epoch": self._state.current_epoch,
            },
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.

        Args:
            path: Path to load checkpoint from.
        """
        checkpoint = safe_load_checkpoint(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        for k, state in checkpoint["optimizer_states"].items():
            self._optimizers[k].load_state_dict(state)

        for k, state in checkpoint["scheduler_states"].items():
            self._schedulers[k].load_state_dict(state)

        training_state = checkpoint["training_state"]
        self._state.best_metric = training_state["best_metric"]
        self._state.epochs_no_improve = training_state["epochs_no_improve"]
        self._state.current_epoch = training_state["current_epoch"]

        logger.info(f"Loaded checkpoint from {path}")


def move_batch(batch, device):
    """Move tensors, including nested categories, to the actual model device."""
    return {
        key: (
            {k: v.to(device) for k, v in value.items()}
            if isinstance(value, dict)
            else value.to(device)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }
