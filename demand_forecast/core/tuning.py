"""Hyperparameter tuning for demand forecasting models using Optuna."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

try:
    import optuna
    from optuna.trial import Trial

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    Trial = Any  # type: ignore

from demand_forecast.models import (
    CombinedForecastLoss,
)
from demand_forecast.models.wrapper import create_model

logger = logging.getLogger(__name__)


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning.

    Attributes:
        n_trials: Number of Optuna trials to run.
        timeout: Maximum time in seconds for tuning (None = no limit).
        direction: Optimization direction ("minimize" or "maximize").
        metric: Metric to optimize ("mse", "mae", "loss").
        pruner: Optuna pruner type ("median", "hyperband", "none").
        sampler: Optuna sampler type ("tpe", "random", "cmaes").
        study_name: Name for the Optuna study.
        storage: SQLite path for study persistence (None = in-memory).
        n_jobs: Number of parallel jobs (-1 = all cores).
        seed: Random seed for reproducibility.
    """

    n_trials: int = 50
    timeout: int | None = None
    direction: Literal["minimize", "maximize"] = "minimize"
    metric: Literal["mse", "mae", "loss"] = "mse"
    pruner: Literal["median", "hyperband", "none"] = "median"
    sampler: Literal["tpe", "random", "cmaes"] = "tpe"
    study_name: str = "demand_forecast_tuning"
    storage: str | None = None
    n_jobs: int = 1
    seed: int = 42


@dataclass
class SearchSpace:
    """Defines the hyperparameter search space.

    Each attribute defines the range for a hyperparameter.
    Use None to exclude a parameter from tuning.
    """

    # Model architecture
    d_model: tuple[int, int] | None = (64, 512)
    nhead: list[int] | None = field(default_factory=lambda: [4, 8, 16])
    num_encoder_layers: tuple[int, int] | None = (2, 8)
    num_decoder_layers: tuple[int, int] | None = (2, 6)
    dim_feedforward: tuple[int, int] | None = (256, 2048)
    dropout: tuple[float, float] | None = (0.05, 0.5)

    # Embeddings
    sku_emb_dim: tuple[int, int] | None = (8, 64)
    cat_emb_dims: tuple[int, int] | None = (8, 64)

    # Training
    learning_rate: tuple[float, float] | None = (1e-6, 1e-3)
    weight_decay: tuple[float, float] | None = (1e-5, 1e-1)
    batch_size: list[int] | None = field(default_factory=lambda: [32, 64, 128, 256])

    # Model improvements
    use_rope: bool | None = True  # Include in search if True
    use_pre_layernorm: bool | None = True
    use_film_conditioning: bool | None = True
    use_improved_head: bool | None = True
    stochastic_depth_rate: tuple[float, float] | None = (0.0, 0.3)

    # Advanced model
    patch_size: list[int] | None = field(default_factory=lambda: [2, 4, 8])
    use_decomposition: bool | None = True
    decomposition_kernel: list[int] | None = field(default_factory=lambda: [13, 25, 51])

    # Lightweight model
    tcn_channels_depth: tuple[int, int] | None = (2, 5)
    tcn_channels_width: tuple[int, int] | None = (16, 128)
    tcn_kernel_size: list[int] | None = field(default_factory=lambda: [3, 5, 7])


def _check_optuna() -> None:
    """Check if Optuna is available."""
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is required for hyperparameter tuning. Install it with: pip install optuna"
        )


def create_sampler(sampler_type: str, seed: int) -> "optuna.samplers.BaseSampler":
    """Create an Optuna sampler.

    Args:
        sampler_type: Type of sampler ("tpe", "random", "cmaes").
        seed: Random seed.

    Returns:
        Optuna sampler instance.
    """
    _check_optuna()

    if sampler_type == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    elif sampler_type == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    elif sampler_type == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


def create_pruner(pruner_type: str) -> "optuna.pruners.BasePruner":
    """Create an Optuna pruner.

    Args:
        pruner_type: Type of pruner ("median", "hyperband", "none").

    Returns:
        Optuna pruner instance.
    """
    _check_optuna()

    if pruner_type == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    elif pruner_type == "hyperband":
        return optuna.pruners.HyperbandPruner(min_resource=1, max_resource=10)
    elif pruner_type == "none":
        return optuna.pruners.NopPruner()
    else:
        raise ValueError(f"Unknown pruner type: {pruner_type}")


def suggest_hyperparameters(
    trial: Trial,
    search_space: SearchSpace,
    model_type: str = "standard",
) -> dict[str, Any]:
    """Suggest hyperparameters for a trial.

    Args:
        trial: Optuna trial object.
        search_space: Search space configuration.
        model_type: Type of model being tuned.

    Returns:
        Dictionary of suggested hyperparameters.
    """
    params: dict[str, Any] = {}

    # Model architecture
    if search_space.d_model is not None:
        # Ensure d_model is divisible by potential nhead values
        d_model_base = trial.suggest_int("d_model_base", 1, 8)
        max_nhead = max(search_space.nhead) if search_space.nhead else 8
        params["d_model"] = d_model_base * max_nhead * 8  # Ensures divisibility

    if search_space.nhead is not None:
        params["nhead"] = trial.suggest_categorical("nhead", search_space.nhead)

    if search_space.num_encoder_layers is not None:
        params["num_encoder_layers"] = trial.suggest_int(
            "num_encoder_layers", *search_space.num_encoder_layers
        )

    if search_space.num_decoder_layers is not None:
        params["num_decoder_layers"] = trial.suggest_int(
            "num_decoder_layers", *search_space.num_decoder_layers
        )

    if search_space.dim_feedforward is not None:
        params["dim_feedforward"] = trial.suggest_int(
            "dim_feedforward", *search_space.dim_feedforward, step=64
        )

    if search_space.dropout is not None:
        params["dropout"] = trial.suggest_float("dropout", *search_space.dropout)

    # Embeddings
    if search_space.sku_emb_dim is not None:
        params["sku_emb_dim"] = trial.suggest_int("sku_emb_dim", *search_space.sku_emb_dim, step=8)

    if search_space.cat_emb_dims is not None:
        params["cat_emb_dims"] = trial.suggest_int(
            "cat_emb_dims", *search_space.cat_emb_dims, step=8
        )

    # Training parameters
    if search_space.learning_rate is not None:
        params["learning_rate"] = trial.suggest_float(
            "learning_rate", *search_space.learning_rate, log=True
        )

    if search_space.weight_decay is not None:
        params["weight_decay"] = trial.suggest_float(
            "weight_decay", *search_space.weight_decay, log=True
        )

    if search_space.batch_size is not None:
        params["batch_size"] = trial.suggest_categorical("batch_size", search_space.batch_size)

    # Model-specific parameters
    if model_type == "standard":
        if search_space.use_rope is not None:
            params["use_rope"] = trial.suggest_categorical("use_rope", [True, False])
        if search_space.use_pre_layernorm is not None:
            params["use_pre_layernorm"] = trial.suggest_categorical(
                "use_pre_layernorm", [True, False]
            )
        if search_space.use_film_conditioning is not None:
            params["use_film_conditioning"] = trial.suggest_categorical(
                "use_film_conditioning", [True, False]
            )
        if search_space.use_improved_head is not None:
            params["use_improved_head"] = trial.suggest_categorical(
                "use_improved_head", [True, False]
            )
        if search_space.stochastic_depth_rate is not None:
            params["stochastic_depth_rate"] = trial.suggest_float(
                "stochastic_depth_rate", *search_space.stochastic_depth_rate
            )

    elif model_type == "advanced":
        if search_space.patch_size is not None:
            params["patch_size"] = trial.suggest_categorical("patch_size", search_space.patch_size)
        if search_space.use_decomposition is not None:
            params["use_decomposition"] = trial.suggest_categorical(
                "use_decomposition", [True, False]
            )
        if search_space.decomposition_kernel is not None:
            params["decomposition_kernel"] = trial.suggest_categorical(
                "decomposition_kernel", search_space.decomposition_kernel
            )

    elif model_type in ("lightweight", "lightweight_tcn"):
        if (
            search_space.tcn_channels_depth is not None
            and search_space.tcn_channels_width is not None
        ):
            depth = trial.suggest_int("tcn_depth", *search_space.tcn_channels_depth)
            width = trial.suggest_int("tcn_width", *search_space.tcn_channels_width, step=16)
            params["tcn_channels"] = [width * (2**i) for i in range(depth)]
        if search_space.tcn_kernel_size is not None:
            params["kernel_size"] = trial.suggest_categorical(
                "tcn_kernel_size", search_space.tcn_kernel_size
            )

    return params


class HyperparameterTuner:
    """Hyperparameter tuner using Optuna.

    Example:
        >>> tuner = HyperparameterTuner(
        ...     model_type="standard",
        ...     train_dataloader=train_dl,
        ...     val_dataloader=val_dl,
        ...     base_model_kwargs={"sku_vocab_size": 100, "cat_features_dim": {"cat": 10}},
        ... )
        >>> best_params = tuner.tune(n_trials=20)
        >>> best_model = tuner.get_best_model()
    """

    def __init__(
        self,
        model_type: str,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        base_model_kwargs: dict[str, Any],
        config: TuningConfig | None = None,
        search_space: SearchSpace | None = None,
        device: torch.device | None = None,
        num_epochs_per_trial: int = 5,
        early_stop_patience: int = 2,
    ):
        """Initialize the tuner.

        Args:
            model_type: Type of model to tune ("standard", "advanced", "lightweight").
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader.
            base_model_kwargs: Base kwargs for model creation (e.g., vocab sizes).
            config: Tuning configuration.
            search_space: Hyperparameter search space.
            device: Device to train on.
            num_epochs_per_trial: Number of epochs per trial.
            early_stop_patience: Early stopping patience.
        """
        _check_optuna()

        self.model_type = model_type
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.base_model_kwargs = base_model_kwargs
        self.config = config or TuningConfig()
        self.search_space = search_space or SearchSpace()
        self.device = device or torch.device("cpu")
        self.num_epochs_per_trial = num_epochs_per_trial
        self.early_stop_patience = early_stop_patience

        self.study: optuna.Study | None = None
        self.best_model: nn.Module | None = None

    def _create_objective(self) -> Callable[[Trial], float]:
        """Create the Optuna objective function."""

        def objective(trial: Trial) -> float:
            # Suggest hyperparameters
            params = suggest_hyperparameters(trial, self.search_space, self.model_type)

            # Separate model and training params
            training_params = {}
            model_params = dict(self.base_model_kwargs)

            for key in ["learning_rate", "weight_decay", "batch_size"]:
                if key in params:
                    training_params[key] = params.pop(key)

            model_params.update(params)

            # Create model
            try:
                model = create_model(self.model_type, **model_params)
                model = model.to(self.device)
            except Exception as e:
                logger.warning(f"Failed to create model with params {model_params}: {e}")
                raise optuna.TrialPruned()

            # Setup training
            lr = training_params.get("learning_rate", 1e-4)
            wd = training_params.get("weight_decay", 1e-2)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

            # Use appropriate loss
            if self.model_type == "advanced":
                criterion = CombinedForecastLoss()
            else:
                criterion = nn.MSELoss()

            # Training loop
            best_val_metric = float("inf")
            epochs_no_improve = 0

            for epoch in range(self.num_epochs_per_trial):
                # Training
                model.train()
                train_loss = 0.0
                num_batches = 0

                for batch in self.train_dataloader:
                    # Move to device
                    batch = self._move_batch_to_device(batch)

                    # Forward
                    outputs = model(
                        batch["qty"],
                        batch["past_time"],
                        batch["future_time"],
                        batch["sku"],
                        batch["cats"],
                    )

                    # Compute loss
                    if isinstance(outputs, dict):
                        loss, _ = criterion(outputs, batch["y"])
                    else:
                        loss = criterion(outputs, batch["y"])

                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    num_batches += 1

                avg_train_loss = train_loss / max(num_batches, 1)

                # Validation
                model.eval()
                val_metric = self._evaluate(model, criterion)

                # Report to Optuna for pruning
                trial.report(val_metric, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                # Early stopping
                if val_metric < best_val_metric:
                    best_val_metric = val_metric
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.early_stop_patience:
                        break

                logger.debug(
                    f"Trial {trial.number}, Epoch {epoch + 1}: "
                    f"train_loss={avg_train_loss:.4f}, val_{self.config.metric}={val_metric:.4f}"
                )

            return best_val_metric

        return objective

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move batch tensors to device."""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, dict):
                result[key] = {k: v.to(self.device) for k, v in value.items()}
            else:
                result[key] = value
        return result

    def _evaluate(self, model: nn.Module, criterion: nn.Module) -> float:
        """Evaluate model on validation set."""
        model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = self._move_batch_to_device(batch)

                outputs = model(
                    batch["qty"],
                    batch["past_time"],
                    batch["future_time"],
                    batch["sku"],
                    batch["cats"],
                )

                targets = batch["y"]

                # Handle dict outputs
                if isinstance(outputs, dict):
                    predictions = outputs["prediction"]
                    loss, _ = criterion(outputs, targets)
                else:
                    predictions = outputs
                    loss = criterion(predictions, targets)

                total_loss += loss.item()

                # Compute MSE and MAE
                mse = ((predictions - targets) ** 2).mean().item()
                mae = (predictions - targets).abs().mean().item()
                total_mse += mse
                total_mae += mae
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_mse = total_mse / max(num_batches, 1)
        avg_mae = total_mae / max(num_batches, 1)

        if self.config.metric == "mse":
            return avg_mse
        elif self.config.metric == "mae":
            return avg_mae
        else:
            return avg_loss

    def tune(
        self,
        n_trials: int | None = None,
        timeout: int | None = None,
        show_progress_bar: bool = True,
    ) -> dict[str, Any]:
        """Run hyperparameter tuning.

        Args:
            n_trials: Number of trials (overrides config).
            timeout: Timeout in seconds (overrides config).
            show_progress_bar: Whether to show progress bar.

        Returns:
            Dictionary of best hyperparameters.
        """
        _check_optuna()

        n_trials = n_trials or self.config.n_trials
        timeout = timeout or self.config.timeout

        # Create study
        sampler = create_sampler(self.config.sampler, self.config.seed)
        pruner = create_pruner(self.config.pruner)

        self.study = optuna.create_study(
            study_name=self.config.study_name,
            storage=self.config.storage,
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        # Run optimization
        objective = self._create_objective()

        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=show_progress_bar,
        )

        logger.info(f"Best trial: {self.study.best_trial.number}")
        logger.info(f"Best value: {self.study.best_value:.6f}")
        logger.info(f"Best params: {self.study.best_params}")

        return self.study.best_params

    def get_best_model(self) -> nn.Module:
        """Get the best model from tuning.

        Returns:
            Model with best hyperparameters.
        """
        if self.study is None:
            raise RuntimeError("Must run tune() before getting best model")

        params = suggest_hyperparameters(
            self.study.best_trial,
            self.search_space,
            self.model_type,
        )

        # Filter out training params
        model_params = dict(self.base_model_kwargs)
        for key in ["learning_rate", "weight_decay", "batch_size"]:
            params.pop(key, None)
        model_params.update(params)

        model = create_model(self.model_type, **model_params)
        self.best_model = model.to(self.device)
        return self.best_model

    def get_study_dataframe(self) -> pd.DataFrame:
        """Get study results as a DataFrame."""
        if self.study is None:
            raise RuntimeError("Must run tune() before getting study dataframe")
        return self.study.trials_dataframe()

    def save_study(self, path: str | Path) -> None:
        """Save study to file.

        Args:
            path: Path to save the study.
        """
        import pickle

        if self.study is None:
            raise RuntimeError("Must run tune() before saving study")

        with open(path, "wb") as f:
            pickle.dump(self.study, f)

    @classmethod
    def load_study(cls, path: str | Path) -> "optuna.Study":
        """Load study from file.

        Args:
            path: Path to load the study from.

        Returns:
            Loaded Optuna study.
        """
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)


def quick_tune(
    model_type: str,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    base_model_kwargs: dict[str, Any],
    n_trials: int = 20,
    device: torch.device | None = None,
) -> tuple[dict[str, Any], nn.Module]:
    """Quick hyperparameter tuning with sensible defaults.

    Args:
        model_type: Type of model to tune.
        train_dataloader: Training data loader.
        val_dataloader: Validation data loader.
        base_model_kwargs: Base model kwargs.
        n_trials: Number of trials.
        device: Device to use.

    Returns:
        Tuple of (best_params, best_model).
    """
    tuner = HyperparameterTuner(
        model_type=model_type,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        base_model_kwargs=base_model_kwargs,
        device=device,
        num_epochs_per_trial=3,
    )

    best_params = tuner.tune(n_trials=n_trials)
    best_model = tuner.get_best_model()

    return best_params, best_model
