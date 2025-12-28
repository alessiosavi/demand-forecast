"""Training loop for demand forecasting models."""

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn as nn
from torch import optim
from tqdm.auto import tqdm

from demand_forecast.core.evaluator import Evaluator
from demand_forecast.data.dataloader import get_round_robin_iterators
from demand_forecast.models.losses import compute_loss
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
    test_dataloaders: dict[str, Any]
    num_epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    early_stop: EarlyStopConfig = field(default_factory=EarlyStopConfig)
    flatten_loss: bool = True
    device: torch.device | None = None
    callbacks: list[TrainingCallback] = field(default_factory=list)
    plot_dir: Path | None = None

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

        # Create optimizer and scheduler per cluster (use actual cluster keys)
        for cluster_key in self.train_dataloaders.keys():
            key = str(cluster_key)
            self._optimizers[key] = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            self._schedulers[key] = optim.lr_scheduler.CosineAnnealingLR(
                self._optimizers[key], T_max=self.num_epochs
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
        qty = batch["qty"]
        past_time = batch["past_time"]
        future_time = batch["future_time"]
        sku = batch["sku"]
        cats = {key: value.to(dtype=torch.int32) for key, value in batch["cats"].items()}
        targets = batch["y"]

        outputs = self.model(model_idx, qty, past_time, future_time, sku, cats)
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
        if metrics is None:
            metrics = {}

        total_steps = self._total_train_examples // self.batch_size

        self._evaluator = Evaluator(
            model=self.model,
            criterion=self._criterion,
            batch_size=self.batch_size,
            total_examples=self._total_test_examples,
            flatten_loss=self.flatten_loss,
            metrics=metrics,
            plot_dir=self.plot_dir,
        )

        logger.info(f"Starting training for {self.num_epochs} epochs")

        with tqdm(range(self.num_epochs), position=0, desc="Training") as pbar:
            for epoch in pbar:
                self._state.current_epoch = epoch

                # Notify callbacks
                for callback in self.callbacks:
                    callback.on_epoch_start(epoch, self.num_epochs)

                # Get fresh iterators
                dls_train, dls_test = get_round_robin_iterators(
                    self.train_dataloaders, self.test_dataloaders
                )

                epoch_loss = 0.0
                self.model.train()

                for i, item in tqdm(
                    enumerate(dls_train),
                    position=1,
                    leave=False,
                    total=total_steps,
                    desc=f"Epoch {epoch + 1}",
                ):
                    model_idx = list(item.keys())[0]
                    batch = item[model_idx]

                    loss, _, _, _, _ = self._forward_step(model_idx, batch)

                    # Backward pass
                    self._optimizers[model_idx].zero_grad()
                    loss.backward()
                    self._optimizers[model_idx].step()

                    epoch_loss += loss.item()
                    pbar.set_description(
                        f"Epoch [{epoch + 1}/{self.num_epochs}] - Loss: {(epoch_loss / (i + 1)):.4f}"
                    )

                    # Notify callbacks
                    for callback in self.callbacks:
                        callback.on_batch_end(i, loss.item())

                # Update learning rate schedulers
                for scheduler in self._schedulers.values():
                    scheduler.step()

                avg_loss = epoch_loss / total_steps
                logger.info(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_loss:.4f}")

                # Validation
                should_plot = (epoch + 1) % plot_every_n_epochs == 0
                val_result = self._evaluator.validate(dls_test, plot=should_plot, epoch=epoch + 1)

                # Notify callbacks
                for callback in self.callbacks:
                    callback.on_epoch_end(epoch, avg_loss, val_result.metrics)

                # Early stopping check
                metric = val_result.flatten_mse if self.flatten_loss else val_result.mse

                if metric < self._state.best_metric:
                    self._state.best_model_state = deepcopy(self.model.state_dict())
                    self._state.best_metric = metric
                    self._state.epochs_no_improve = 0
                else:
                    self._state.epochs_no_improve += 1
                    if self._state.epochs_no_improve >= self.early_stop.patience:
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        self.model.load_state_dict(self._state.best_model_state)
                        return

        logger.info("Training completed")

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
