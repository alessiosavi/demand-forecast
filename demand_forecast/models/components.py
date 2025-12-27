"""Neural network components for Transformer models."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer models.

    Adds position information to input embeddings using sine and cosine
    functions of different frequencies.

    Attributes:
        pe: Buffer containing precomputed positional encodings.
    """

    def __init__(self, d_model: int, max_len: int = 1000):
        """Initialize positional encoding.

        Args:
            d_model: Model dimension (must match input embedding dimension).
            max_len: Maximum sequence length to precompute.
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Input with positional encoding added.
        """
        return x + self.pe[:, : x.size(1), :]


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for better position extrapolation.

    RoPE encodes position information by rotating query and key vectors,
    providing better length generalization than sinusoidal embeddings.

    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, max_len: int = 2048, base: float = 10000.0):
        """Initialize RoPE.

        Args:
            dim: Dimension per attention head (d_model // nhead).
            max_len: Maximum sequence length.
            base: Base for frequency computation.
        """
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base

        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin cache
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build cos/sin cache for given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cos/sin embeddings for sequence length.

        Args:
            seq_len: Current sequence length.

        Returns:
            Tuple of (cos, sin) tensors of shape (seq_len, dim).
        """
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor of shape (batch, heads, seq_len, head_dim).
        k: Key tensor of shape (batch, heads, seq_len, head_dim).
        cos: Cosine embeddings of shape (seq_len, head_dim).
        sin: Sine embeddings of shape (seq_len, head_dim).

    Returns:
        Tuple of rotated (query, key) tensors.
    """

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    # Reshape cos/sin for broadcasting: (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for non-linear processing with skip connections.

    GRN applies non-linear transformations with gating mechanism and residual
    connections, allowing the network to suppress unnecessary components.

    Reference: Temporal Fusion Transformers (https://arxiv.org/abs/1912.09363)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        context_dim: int | None = None,
        dropout: float = 0.1,
    ):
        """Initialize GRN.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension (defaults to input_dim).
            context_dim: Optional context vector dimension for conditioning.
            dropout: Dropout rate.
        """
        super().__init__()
        self.output_dim = output_dim or input_dim

        # Primary transformation
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.output_dim)

        # Context projection (optional)
        self.context_proj = None
        if context_dim is not None:
            self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False)

        # Gating mechanism
        self.gate = nn.Linear(self.output_dim, self.output_dim)

        # Skip connection
        self.skip = nn.Linear(input_dim, self.output_dim) if input_dim != self.output_dim else None

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.output_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., input_dim).
            context: Optional context tensor of shape (..., context_dim).

        Returns:
            Output tensor of shape (..., output_dim).
        """
        # Skip connection
        residual = self.skip(x) if self.skip is not None else x

        # Primary path
        hidden = self.fc1(x)
        if self.context_proj is not None and context is not None:
            hidden = hidden + self.context_proj(context)
        hidden = F.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)

        # Gating
        gate = torch.sigmoid(self.gate(hidden))
        output = gate * hidden

        # Residual and normalization
        return self.layer_norm(output + residual)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for learning feature importance.

    VSN learns to weight different input features based on their relevance,
    providing interpretable feature selection.

    Reference: Temporal Fusion Transformers (https://arxiv.org/abs/1912.09363)
    """

    def __init__(
        self,
        input_dim: int,
        num_inputs: int,
        hidden_dim: int,
        context_dim: int | None = None,
        dropout: float = 0.1,
    ):
        """Initialize VSN.

        Args:
            input_dim: Dimension of each input feature.
            num_inputs: Number of input variables to select from.
            hidden_dim: Hidden dimension for GRN processing.
            context_dim: Optional context dimension for conditioning.
            dropout: Dropout rate.
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.input_dim = input_dim

        # GRN for each input variable
        self.input_grns = nn.ModuleList(
            [
                GatedResidualNetwork(input_dim, hidden_dim, input_dim, dropout=dropout)
                for _ in range(num_inputs)
            ]
        )

        # Softmax weights for variable selection
        self.weight_grn = GatedResidualNetwork(
            input_dim * num_inputs,
            hidden_dim,
            num_inputs,
            context_dim=context_dim,
            dropout=dropout,
        )

    def forward(
        self,
        inputs: list[torch.Tensor],
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            inputs: List of input tensors, each of shape (..., input_dim).
            context: Optional context tensor for conditioning.

        Returns:
            Tuple of (weighted_output, selection_weights).
            - weighted_output: Shape (..., input_dim)
            - selection_weights: Shape (..., num_inputs)
        """
        # Process each input through its GRN
        processed = [grn(inp) for grn, inp in zip(self.input_grns, inputs)]
        processed_stack = torch.stack(processed, dim=-2)  # (..., num_inputs, input_dim)

        # Compute selection weights
        flattened = torch.cat(inputs, dim=-1)  # (..., num_inputs * input_dim)
        weights = self.weight_grn(flattened, context)
        weights = F.softmax(weights, dim=-1)  # (..., num_inputs)

        # Weighted combination
        weights_expanded = weights.unsqueeze(-1)  # (..., num_inputs, 1)
        weighted_output = (processed_stack * weights_expanded).sum(dim=-2)

        return weighted_output, weights


class PatchEmbedding(nn.Module):
    """Patch embedding for time series (PatchTST-style).

    Divides time series into patches and projects them to embedding space,
    enabling better capture of local temporal patterns.

    Reference: https://arxiv.org/abs/2211.14730
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        patch_size: int = 4,
        stride: int | None = None,
    ):
        """Initialize patch embedding.

        Args:
            input_dim: Number of input channels/features.
            d_model: Output embedding dimension.
            patch_size: Size of each patch.
            stride: Stride between patches (defaults to patch_size for non-overlapping).
        """
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride or patch_size

        # Linear projection from patch to embedding
        self.proj = nn.Linear(input_dim * patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Patch embeddings of shape (batch, num_patches, d_model).
        """
        batch_size, seq_len, input_dim = x.shape

        # Pad if necessary
        pad_len = (self.patch_size - seq_len % self.patch_size) % self.patch_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        # Unfold into patches: (batch, num_patches, patch_size, input_dim)
        x = x.unfold(1, self.patch_size, self.stride)
        # Flatten patches: (batch, num_patches, patch_size * input_dim)
        x = x.reshape(batch_size, -1, self.patch_size * input_dim)

        # Project and normalize
        return self.norm(self.proj(x))


class SeriesDecomposition(nn.Module):
    """Series decomposition using moving average (Autoformer-style).

    Separates time series into trend and seasonal components using
    moving average filtering.

    Reference: https://arxiv.org/abs/2106.13008
    """

    def __init__(self, kernel_size: int = 25):
        """Initialize series decomposition.

        Args:
            kernel_size: Moving average kernel size (should be odd).
        """
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.avg_pool = nn.AvgPool1d(
            kernel_size, stride=1, padding=padding, count_include_pad=False
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, features).

        Returns:
            Tuple of (seasonal, trend) components, each same shape as input.
        """
        # Pool expects (batch, channels, length)
        x_permuted = x.permute(0, 2, 1)
        trend = self.avg_pool(x_permuted)

        # Handle edge case where output length differs
        if trend.size(-1) != x.size(1):
            trend = F.interpolate(trend, size=x.size(1), mode="linear", align_corners=False)

        trend = trend.permute(0, 2, 1)
        seasonal = x - trend

        return seasonal, trend


class FiLMConditioning(nn.Module):
    """Feature-wise Linear Modulation (FiLM) for conditional processing.

    Modulates features using learned scale (gamma) and shift (beta)
    parameters derived from a context vector.

    Reference: https://arxiv.org/abs/1709.07871
    """

    def __init__(self, d_model: int, context_dim: int):
        """Initialize FiLM conditioning.

        Args:
            d_model: Feature dimension to modulate.
            context_dim: Context vector dimension.
        """
        super().__init__()
        self.gamma = nn.Linear(context_dim, d_model)
        self.beta = nn.Linear(context_dim, d_model)

        # Initialize gamma close to 1, beta close to 0
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            x: Input features of shape (batch, seq_len, d_model).
            context: Context vector of shape (batch, context_dim).

        Returns:
            Modulated features of shape (batch, seq_len, d_model).
        """
        gamma = self.gamma(context).unsqueeze(1)  # (batch, 1, d_model)
        beta = self.beta(context).unsqueeze(1)  # (batch, 1, d_model)
        return gamma * x + beta


class StochasticDepth(nn.Module):
    """Stochastic depth for regularization via random layer dropping.

    During training, randomly drops entire residual branches with a given
    probability. At inference, all layers are active.

    Reference: https://arxiv.org/abs/1603.09382
    """

    def __init__(self, drop_prob: float = 0.1):
        """Initialize stochastic depth.

        Args:
            drop_prob: Probability of dropping the layer during training.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth.

        Args:
            x: Residual branch output to potentially drop.
            residual: Skip connection tensor.

        Returns:
            Combined output with stochastic depth applied.
        """
        if not self.training or self.drop_prob == 0.0:
            return residual + x

        keep_prob = 1.0 - self.drop_prob
        # Sample a single random value per batch element
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device, dtype=x.dtype))
        # Scale to maintain expected value
        return residual + x * mask / keep_prob


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention that returns attention weights for interpretability.

    Standard multi-head attention with optional RoPE and explicit attention
    weight output for model explainability.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        use_rope: bool = True,
        max_len: int = 2048,
    ):
        """Initialize interpretable attention.

        Args:
            d_model: Model dimension.
            nhead: Number of attention heads.
            dropout: Dropout rate for attention weights.
            use_rope: Whether to use rotary position embeddings.
            max_len: Maximum sequence length for RoPE.
        """
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim, max_len)
        else:
            self.rope = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            query: Query tensor of shape (batch, tgt_len, d_model).
            key: Key tensor of shape (batch, src_len, d_model).
            value: Value tensor of shape (batch, src_len, d_model).
            attn_mask: Optional attention mask.
            return_attention: Whether to return attention weights.

        Returns:
            Tuple of (output, attention_weights).
            - output: Shape (batch, tgt_len, d_model)
            - attention_weights: Shape (batch, nhead, tgt_len, src_len) or None
        """
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]

        # Project to Q, K, V
        q = self.q_proj(query).view(batch_size, tgt_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, src_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, src_len, self.nhead, self.head_dim).transpose(1, 2)

        # Apply RoPE if enabled
        if self.use_rope and self.rope is not None:
            cos, sin = self.rope(max(tgt_len, src_len))
            q, k = apply_rotary_pos_emb(q, k, cos[:tgt_len], sin[:src_len])

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights_dropped, v)

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)
        )
        output = self.out_proj(attn_output)

        if return_attention:
            return output, attn_weights
        return output, None


class TemporalBlock(nn.Module):
    """Temporal convolutional block with dilated causal convolutions.

    A single block in a Temporal Convolutional Network (TCN), using
    dilated causal convolutions for capturing long-range dependencies.

    Reference: https://arxiv.org/abs/1803.01271
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        """Initialize temporal block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            dilation: Dilation factor for the convolution.
            dropout: Dropout rate.
        """
        super().__init__()
        # Causal padding: (kernel_size - 1) * dilation
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )

        self.chomp1 = padding  # Amount to trim for causal convolution
        self.chomp2 = padding

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming normal."""
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, seq_len).

        Returns:
            Output tensor of shape (batch, out_channels, seq_len).
        """
        # First conv block
        out = self.conv1(x)
        out = out[:, :, : -self.chomp1] if self.chomp1 > 0 else out  # Causal trim
        out = self.relu(out)
        out = self.dropout(out)

        # Second conv block
        out = self.conv2(out)
        out = out[:, :, : -self.chomp2] if self.chomp2 > 0 else out  # Causal trim
        out = self.relu(out)
        out = self.dropout(out)

        # Residual connection
        residual = self.downsample(x) if self.downsample is not None else x
        return self.relu(out + residual)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network (TCN).

    A stack of temporal blocks with exponentially increasing dilation,
    providing a large receptive field with efficient computation.
    """

    def __init__(
        self,
        input_dim: int,
        channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        """Initialize TCN.

        Args:
            input_dim: Number of input features.
            channels: List of channel sizes for each layer.
            kernel_size: Convolution kernel size.
            dropout: Dropout rate.
        """
        super().__init__()
        layers = []
        num_levels = len(channels)

        for i in range(num_levels):
            dilation = 2**i
            in_ch = input_dim if i == 0 else channels[i - 1]
            out_ch = channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Output tensor of shape (batch, seq_len, channels[-1]).
        """
        # TCN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        out = self.network(x)
        return out.transpose(1, 2)
