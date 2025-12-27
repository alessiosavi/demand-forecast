"""Lightweight CPU-optimized model for demand forecasting.

This module implements LightweightDemandModel, optimized for:
- Fast CPU inference (< 50ms for batch=32)
- Small model size (< 3M parameters, < 50MB fp32, < 15MB int8)
- Quantization support (static and dynamic)
- Export to ONNX and TorchScript

Architecture: Temporal Convolutional Network (TCN) with FiLM conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from demand_forecast.models.components import (
    FiLMConditioning,
    TemporalConvNet,
)


class LightweightDemandModel(nn.Module):
    """Lightweight demand forecasting model optimized for CPU inference.

    Uses a Temporal Convolutional Network (TCN) backbone with FiLM conditioning
    for incorporating static features. Designed for production deployment with
    quantization and export support.

    Architecture:
        1. Compact embeddings for SKU and categorical features
        2. TCN backbone for temporal processing
        3. FiLM conditioning to modulate temporal features with static context
        4. Linear output head

    Attributes:
        sku_embedding: Compact SKU embeddings.
        cat_embeddings: Compact categorical embeddings.
        tcn: Temporal Convolutional Network backbone.
        film: FiLM conditioning layer.
        output_head: Final prediction layer.
    """

    def __init__(
        self,
        sku_vocab_size: int,
        cat_features_dim: dict[str, int],
        sku_emb_dim: int = 16,
        cat_emb_dims: int = 16,
        tcn_channels: list[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        n_out: int = 16,
        past_time_features_dim: int = 5,  # qty + 4 time features
        future_time_features_dim: int = 4,
        use_film: bool = True,
        **kwargs,
    ):
        """Initialize the lightweight model.

        Args:
            sku_vocab_size: Number of unique SKUs.
            cat_features_dim: Dict mapping category names to vocab sizes.
            sku_emb_dim: SKU embedding dimension (keep small for speed).
            cat_emb_dims: Categorical embedding dimension.
            tcn_channels: List of TCN channel sizes. Default: [32, 64, 64].
            kernel_size: Convolution kernel size.
            dropout: Dropout rate.
            n_out: Number of forecast steps.
            past_time_features_dim: Past input dimension.
            future_time_features_dim: Future time features dimension.
            use_film: Whether to use FiLM conditioning.
        """
        super().__init__()

        if tcn_channels is None:
            tcn_channels = [32, 64, 64]

        self.n_out = n_out
        self.use_film = use_film

        # Compact embeddings
        self.sku_embedding = nn.Embedding(sku_vocab_size, sku_emb_dim)
        self.cat_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(vocab_size, cat_emb_dims)
                for name, vocab_size in cat_features_dim.items()
            }
        )

        # Static feature dimension
        total_static_dim = sku_emb_dim + cat_emb_dims * len(cat_features_dim)
        self.static_proj = nn.Linear(total_static_dim, tcn_channels[-1])

        # Input projection
        input_dim = past_time_features_dim
        self.input_proj = nn.Linear(input_dim, tcn_channels[0])

        # TCN backbone
        self.tcn = TemporalConvNet(
            input_dim=tcn_channels[0],
            channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # FiLM conditioning
        if use_film:
            self.film = FiLMConditioning(tcn_channels[-1], tcn_channels[-1])
        else:
            self.film = None

        # Future time projection and processing
        self.future_proj = nn.Linear(future_time_features_dim, tcn_channels[-1])

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(tcn_channels[-1], tcn_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tcn_channels[-1] // 2, 1),
        )

        # Store config for export
        self._config = {
            "sku_vocab_size": sku_vocab_size,
            "cat_features_dim": cat_features_dim,
            "n_out": n_out,
            "tcn_channels": tcn_channels,
        }

    def _get_static_features(
        self,
        sku: torch.Tensor,
        cats: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Extract and combine static features.

        Args:
            sku: SKU indices (batch,).
            cats: Dict of categorical features.

        Returns:
            Combined static embedding (batch, static_dim).
        """
        # SKU embedding
        sku_emb = self.sku_embedding(sku)

        # Category embeddings with max pooling
        cat_embs = []
        for name, emb_layer in self.cat_embeddings.items():
            cat_emb = torch.max(emb_layer(cats[name]), dim=1)[0]
            cat_embs.append(cat_emb)

        # Combine all static features
        if cat_embs:
            static = torch.cat([sku_emb] + cat_embs, dim=-1)
        else:
            static = sku_emb

        return self.static_proj(static)

    def forward(
        self,
        qty: torch.Tensor,
        past_time: torch.Tensor,
        future_time: torch.Tensor,
        sku: torch.Tensor,
        cats: dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            qty: Past quantities, shape (batch, past_len, 1).
            past_time: Past time features, shape (batch, past_len, time_dim).
            future_time: Future time features, shape (batch, future_len, time_dim).
            sku: SKU indices, shape (batch,).
            cats: Dict of categorical features.

        Returns:
            Predictions of shape (batch, n_out).
        """
        # Get static features
        static = self._get_static_features(sku, cats)

        # Process past inputs through TCN
        past_inputs = torch.cat([qty, past_time], dim=-1)
        past_emb = self.input_proj(past_inputs)
        tcn_output = self.tcn(past_emb)  # (batch, seq_len, channels)

        # Take the last hidden state as context
        context = tcn_output[:, -1, :]  # (batch, channels)

        # Apply FiLM conditioning if enabled
        if self.use_film and self.film is not None:
            # Expand static for sequence
            future_emb = self.future_proj(future_time)
            future_conditioned = self.film(future_emb, static)
        else:
            future_emb = self.future_proj(future_time)
            static_expanded = static.unsqueeze(1).expand(-1, future_time.size(1), -1)
            future_conditioned = future_emb + static_expanded

        # Add context from encoder
        context_expanded = context.unsqueeze(1).expand(-1, future_time.size(1), -1)
        decoder_input = future_conditioned + context_expanded

        # Output predictions
        output = self.output_head(decoder_input).squeeze(-1)  # (batch, future_len)

        # Ensure output matches n_out
        if output.size(1) != self.n_out:
            output = F.interpolate(
                output.unsqueeze(1), size=self.n_out, mode="linear", align_corners=False
            ).squeeze(1)

        return output

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self, quantized: bool = False) -> float:
        """Estimate model size in MB.

        Args:
            quantized: If True, estimate int8 size.

        Returns:
            Estimated size in MB.
        """
        param_count = self.count_parameters()
        bytes_per_param = 1 if quantized else 4  # int8 vs float32
        return param_count * bytes_per_param / (1024 * 1024)

    # ==================== Quantization Support ====================

    def prepare_for_quantization(self) -> "LightweightDemandModel":
        """Prepare model for quantization-aware training.

        Returns:
            Self with quantization stubs inserted.
        """
        torch.quantization.prepare_qat(self, inplace=True)
        return self

    def convert_to_quantized(self) -> nn.Module:
        """Convert to quantized model for inference.

        Returns:
            Quantized model.
        """
        self.eval()
        return torch.quantization.convert(self)

    @staticmethod
    def get_quantization_config() -> torch.quantization.QConfig:
        """Get recommended quantization configuration."""
        return torch.quantization.get_default_qat_qconfig("fbgemm")

    # ==================== Export Methods ====================

    def export_onnx(
        self,
        path: str,
        batch_size: int = 1,
        past_len: int = 52,
        future_len: int = 16,
        opset_version: int = 14,
    ) -> None:
        """Export model to ONNX format.

        Args:
            path: Output file path.
            batch_size: Batch size for export.
            past_len: Past sequence length.
            future_len: Future sequence length.
            opset_version: ONNX opset version.
        """
        self.eval()

        # Create dummy inputs
        dummy_qty = torch.randn(batch_size, past_len, 1)
        dummy_past_time = torch.randn(batch_size, past_len, 4)
        dummy_future_time = torch.randn(batch_size, future_len, 4)
        dummy_sku = torch.zeros(batch_size, dtype=torch.long)

        # Create dummy categorical inputs
        dummy_cats = {}
        for name, vocab_size in self._config["cat_features_dim"].items():
            dummy_cats[name] = torch.zeros(batch_size, vocab_size, dtype=torch.long)

        # Export
        torch.onnx.export(
            self,
            (dummy_qty, dummy_past_time, dummy_future_time, dummy_sku, dummy_cats),
            path,
            input_names=["qty", "past_time", "future_time", "sku", "cats"],
            output_names=["predictions"],
            dynamic_axes={
                "qty": {0: "batch_size", 1: "past_len"},
                "past_time": {0: "batch_size", 1: "past_len"},
                "future_time": {0: "batch_size", 1: "future_len"},
                "sku": {0: "batch_size"},
                "predictions": {0: "batch_size"},
            },
            opset_version=opset_version,
        )

    def to_torchscript(self, optimize: bool = True) -> torch.jit.ScriptModule:
        """Convert model to TorchScript for deployment.

        Args:
            optimize: Whether to apply optimizations.

        Returns:
            TorchScript module.
        """
        self.eval()

        # Use tracing for simpler export
        batch_size = 1
        past_len = 52
        future_len = 16

        dummy_qty = torch.randn(batch_size, past_len, 1)
        dummy_past_time = torch.randn(batch_size, past_len, 4)
        dummy_future_time = torch.randn(batch_size, future_len, 4)
        dummy_sku = torch.zeros(batch_size, dtype=torch.long)

        # Create wrapper for tracing with dict input
        class TracingWrapper(nn.Module):
            def __init__(self, model, cat_names):
                super().__init__()
                self.model = model
                self.cat_names = cat_names

            def forward(self, qty, past_time, future_time, sku, *cat_tensors):
                cats = {name: tensor for name, tensor in zip(self.cat_names, cat_tensors)}
                return self.model(qty, past_time, future_time, sku, cats)

        wrapper = TracingWrapper(self, list(self._config["cat_features_dim"].keys()))

        # Create dummy cat inputs
        cat_tensors = []
        for name, vocab_size in self._config["cat_features_dim"].items():
            cat_tensors.append(torch.zeros(batch_size, vocab_size, dtype=torch.long))

        # Trace
        traced = torch.jit.trace(
            wrapper,
            (dummy_qty, dummy_past_time, dummy_future_time, dummy_sku, *cat_tensors),
        )

        if optimize:
            traced = torch.jit.optimize_for_inference(traced)

        return traced


class LightweightMixerModel(nn.Module):
    """Alternative lightweight model using MLP-Mixer architecture.

    Even more efficient than TCN for some use cases, using only
    MLPs without convolutions.
    """

    def __init__(
        self,
        sku_vocab_size: int,
        cat_features_dim: dict[str, int],
        sku_emb_dim: int = 16,
        cat_emb_dims: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_out: int = 16,
        past_time_features_dim: int = 5,
        future_time_features_dim: int = 4,
        max_seq_len: int = 52,
        **kwargs,
    ):
        """Initialize MLP-Mixer model.

        Args:
            sku_vocab_size: Number of unique SKUs.
            cat_features_dim: Dict mapping category names to vocab sizes.
            sku_emb_dim: SKU embedding dimension.
            cat_emb_dims: Categorical embedding dimension.
            hidden_dim: Hidden dimension for mixer layers.
            num_layers: Number of mixer layers.
            dropout: Dropout rate.
            n_out: Number of forecast steps.
            past_time_features_dim: Past input dimension.
            future_time_features_dim: Future time features dimension.
            max_seq_len: Maximum sequence length.
        """
        super().__init__()

        self.n_out = n_out
        self.hidden_dim = hidden_dim

        # Embeddings
        self.sku_embedding = nn.Embedding(sku_vocab_size, sku_emb_dim)
        self.cat_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(vocab_size, cat_emb_dims)
                for name, vocab_size in cat_features_dim.items()
            }
        )

        total_static_dim = sku_emb_dim + cat_emb_dims * len(cat_features_dim)
        self.static_proj = nn.Linear(total_static_dim, hidden_dim)

        # Input projection
        self.input_proj = nn.Linear(past_time_features_dim, hidden_dim)

        # Mixer layers
        self.mixer_layers = nn.ModuleList(
            [MixerBlock(hidden_dim, max_seq_len, dropout) for _ in range(num_layers)]
        )

        # Output
        self.output_proj = nn.Linear(hidden_dim, n_out)

    def forward(
        self,
        qty: torch.Tensor,
        past_time: torch.Tensor,
        future_time: torch.Tensor,
        sku: torch.Tensor,
        cats: dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass."""
        # Static features
        sku_emb = self.sku_embedding(sku)
        cat_embs = [
            torch.max(self.cat_embeddings[name](cats[name]), dim=1)[0]
            for name in self.cat_embeddings.keys()
        ]
        static = self.static_proj(torch.cat([sku_emb] + cat_embs, dim=-1))

        # Input
        past_inputs = torch.cat([qty, past_time], dim=-1)
        x = self.input_proj(past_inputs)

        # Add static context
        x = x + static.unsqueeze(1)

        # Mixer layers
        for layer in self.mixer_layers:
            x = layer(x)

        # Global average pooling and output
        x = x.mean(dim=1)  # (batch, hidden_dim)
        return self.output_proj(x)  # (batch, n_out)


class MixerBlock(nn.Module):
    """MLP-Mixer block with token and channel mixing."""

    def __init__(self, hidden_dim: int, seq_len: int, dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Token mixing (across time)
        self.token_mixing = nn.Sequential(
            nn.Linear(seq_len, seq_len * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len * 2, seq_len),
            nn.Dropout(dropout),
        )

        # Channel mixing (across features)
        self.channel_mixing = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, hidden_dim).

        Returns:
            Output of same shape.
        """
        # Token mixing
        residual = x
        x = self.norm1(x)
        x = x.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        x = self.token_mixing(x)
        x = x.transpose(1, 2)  # (batch, seq_len, hidden_dim)
        x = x + residual

        # Channel mixing
        residual = x
        x = self.norm2(x)
        x = self.channel_mixing(x)
        x = x + residual

        return x
