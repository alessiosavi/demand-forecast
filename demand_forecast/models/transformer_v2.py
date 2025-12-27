"""Research-grade Transformer model for demand forecasting.

This module implements AdvancedDemandForecastModelV2, incorporating state-of-the-art
techniques from recent time series forecasting literature:
- Temporal Fusion Transformer (TFT): Variable Selection Networks, Gated Residual Networks
- PatchTST: Patch-based tokenization for local pattern capture
- Autoformer: Series decomposition (trend/seasonality separation)
- RoPE: Rotary Position Embeddings for better length generalization
- Probabilistic forecasting: Quantile outputs for uncertainty estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from demand_forecast.models.components import (
    FiLMConditioning,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    PatchEmbedding,
    SeriesDecomposition,
    StochasticDepth,
    VariableSelectionNetwork,
)


class PreNormEncoderLayer(nn.Module):
    """Transformer encoder layer with Pre-LayerNorm and optional RoPE."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_rope: bool = True,
        max_len: int = 2048,
        stochastic_depth_rate: float = 0.0,
    ):
        super().__init__()
        self.self_attn = InterpretableMultiHeadAttention(
            d_model, nhead, dropout, use_rope=use_rope, max_len=max_len
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.stochastic_depth = (
            StochasticDepth(stochastic_depth_rate) if stochastic_depth_rate > 0 else None
        )

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Pre-LN: normalize before attention
        src_normed = self.norm1(src)
        attn_out, attn_weights = self.self_attn(
            src_normed,
            src_normed,
            src_normed,
            attn_mask=src_mask,
            return_attention=return_attention,
        )
        attn_out = self.dropout1(attn_out)

        # Residual with optional stochastic depth
        if self.stochastic_depth is not None:
            src = self.stochastic_depth(attn_out, src)
        else:
            src = src + attn_out

        # FFN with Pre-LN
        src_normed = self.norm2(src)
        ffn_out = self.linear2(self.dropout(F.gelu(self.linear1(src_normed))))
        ffn_out = self.dropout2(ffn_out)

        if self.stochastic_depth is not None:
            src = self.stochastic_depth(ffn_out, src)
        else:
            src = src + ffn_out

        return src, attn_weights


class PreNormDecoderLayer(nn.Module):
    """Transformer decoder layer with Pre-LayerNorm and optional RoPE."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_rope: bool = True,
        max_len: int = 2048,
        stochastic_depth_rate: float = 0.0,
    ):
        super().__init__()
        self.self_attn = InterpretableMultiHeadAttention(
            d_model, nhead, dropout, use_rope=use_rope, max_len=max_len
        )
        self.cross_attn = InterpretableMultiHeadAttention(
            d_model,
            nhead,
            dropout,
            use_rope=False,
            max_len=max_len,  # No RoPE for cross-attn
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.stochastic_depth = (
            StochasticDepth(stochastic_depth_rate) if stochastic_depth_rate > 0 else None
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # Self-attention with Pre-LN
        tgt_normed = self.norm1(tgt)
        self_attn_out, self_attn_weights = self.self_attn(
            tgt_normed,
            tgt_normed,
            tgt_normed,
            attn_mask=tgt_mask,
            return_attention=return_attention,
        )
        self_attn_out = self.dropout1(self_attn_out)

        if self.stochastic_depth is not None:
            tgt = self.stochastic_depth(self_attn_out, tgt)
        else:
            tgt = tgt + self_attn_out

        # Cross-attention with Pre-LN
        tgt_normed = self.norm2(tgt)
        cross_attn_out, cross_attn_weights = self.cross_attn(
            tgt_normed,
            memory,
            memory,
            return_attention=return_attention,
        )
        cross_attn_out = self.dropout2(cross_attn_out)

        if self.stochastic_depth is not None:
            tgt = self.stochastic_depth(cross_attn_out, tgt)
        else:
            tgt = tgt + cross_attn_out

        # FFN with Pre-LN
        tgt_normed = self.norm3(tgt)
        ffn_out = self.linear2(self.dropout(F.gelu(self.linear1(tgt_normed))))
        ffn_out = self.dropout3(ffn_out)

        if self.stochastic_depth is not None:
            tgt = self.stochastic_depth(ffn_out, tgt)
        else:
            tgt = tgt + ffn_out

        return tgt, self_attn_weights, cross_attn_weights


class StaticEnrichmentLayer(nn.Module):
    """Static enrichment using GRN for incorporating static context into temporal features."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.grn = GatedResidualNetwork(
            input_dim=d_model,
            hidden_dim=hidden_dim,
            output_dim=d_model,
            context_dim=d_model,
            dropout=dropout,
        )

    def forward(self, temporal: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """Enrich temporal features with static context.

        Args:
            temporal: Temporal features (batch, seq_len, d_model).
            static: Static context (batch, d_model).

        Returns:
            Enriched temporal features (batch, seq_len, d_model).
        """
        batch_size, seq_len, _ = temporal.shape
        static_expanded = static.unsqueeze(1).expand(-1, seq_len, -1)
        return self.grn(temporal, static_expanded)


class AdvancedDemandForecastModelV2(nn.Module):
    """Research-grade demand forecasting model with state-of-the-art techniques.

    This model incorporates:
    - Variable Selection Networks for feature importance
    - Patch embedding for local temporal pattern capture
    - Series decomposition for trend/seasonality separation
    - Rotary Position Embeddings (RoPE)
    - Pre-LayerNorm for training stability
    - Probabilistic forecasting with quantile outputs
    - Interpretable attention weights

    The model outputs a dictionary with predictions, quantiles, and optional
    decomposition components for full interpretability.
    """

    def __init__(
        self,
        sku_vocab_size: int,
        cat_features_dim: dict[str, int],
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        n_out: int = 16,
        # Embedding dimensions
        sku_emb_dim: int = 32,
        cat_emb_dims: int = 32,
        # Time features
        past_time_features_dim: int = 5,  # qty + 4 time features
        future_time_features_dim: int = 4,
        # Advanced features
        patch_size: int = 4,
        num_quantiles: int = 3,
        quantiles: list[float] | None = None,
        use_decomposition: bool = True,
        decomposition_kernel: int = 25,
        hidden_continuous_size: int = 64,
        # Architecture options
        max_past_len: int = 100,
        max_future_len: int = 50,
        stochastic_depth_rate: float = 0.1,
        use_patch_embedding: bool = True,
        **kwargs,
    ):
        """Initialize the advanced model.

        Args:
            sku_vocab_size: Number of unique SKUs.
            cat_features_dim: Dict mapping category names to vocab sizes.
            d_model: Transformer model dimension.
            nhead: Number of attention heads.
            num_encoder_layers: Number of encoder layers.
            num_decoder_layers: Number of decoder layers.
            dim_feedforward: FFN dimension.
            dropout: Dropout rate.
            n_out: Number of forecast steps.
            sku_emb_dim: SKU embedding dimension.
            cat_emb_dims: Categorical embedding dimension.
            past_time_features_dim: Past input dimension (qty + time features).
            future_time_features_dim: Future time features dimension.
            patch_size: Size of patches for patch embedding.
            num_quantiles: Number of quantile outputs.
            quantiles: Specific quantile values (default: [0.1, 0.5, 0.9]).
            use_decomposition: Whether to use series decomposition.
            decomposition_kernel: Kernel size for moving average decomposition.
            hidden_continuous_size: Hidden size for continuous variable processing.
            max_past_len: Maximum past sequence length.
            max_future_len: Maximum future sequence length.
            stochastic_depth_rate: Drop probability for stochastic depth.
            use_patch_embedding: Whether to use patch embedding.
        """
        super().__init__()

        # Configuration
        self.d_model = d_model
        self.n_out = n_out
        self.num_quantiles = num_quantiles
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.use_decomposition = use_decomposition
        self.use_patch_embedding = use_patch_embedding
        self.patch_size = patch_size

        # Embeddings
        self.sku_embedding = nn.Embedding(sku_vocab_size, sku_emb_dim)
        self.cat_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(vocab_size, cat_emb_dims)
                for name, vocab_size in cat_features_dim.items()
            }
        )

        # Static feature dimension
        num_static_features = 1 + len(cat_features_dim)  # SKU + categories

        # Variable Selection Network for static features
        self.static_vsn = VariableSelectionNetwork(
            input_dim=max(sku_emb_dim, cat_emb_dims),
            num_inputs=num_static_features,
            hidden_dim=hidden_continuous_size,
            dropout=dropout,
        )
        self.static_proj = nn.Linear(max(sku_emb_dim, cat_emb_dims), d_model)

        # Input projections
        self.past_proj = nn.Linear(past_time_features_dim, d_model)
        self.future_proj = nn.Linear(future_time_features_dim, d_model)

        # Patch embedding (optional)
        if use_patch_embedding:
            self.patch_embed = PatchEmbedding(
                input_dim=d_model,
                d_model=d_model,
                patch_size=patch_size,
            )
        else:
            self.patch_embed = None

        # Series decomposition (optional)
        if use_decomposition:
            self.decomposition = SeriesDecomposition(kernel_size=decomposition_kernel)
            self.trend_proj = nn.Linear(d_model, d_model)
            self.seasonal_proj = nn.Linear(d_model, d_model)
        else:
            self.decomposition = None

        # FiLM conditioning for static → temporal
        self.encoder_film = FiLMConditioning(d_model, d_model)
        self.decoder_film = FiLMConditioning(d_model, d_model)

        # Transformer Encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                PreNormEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    use_rope=True,
                    max_len=max(max_past_len, max_future_len),
                    stochastic_depth_rate=stochastic_depth_rate * (i + 1) / num_encoder_layers,
                )
                for i in range(num_encoder_layers)
            ]
        )

        # Static enrichment layer (between encoder and decoder)
        self.static_enrichment = StaticEnrichmentLayer(
            d_model=d_model,
            hidden_dim=hidden_continuous_size,
            dropout=dropout,
        )

        # Transformer Decoder layers
        self.decoder_layers = nn.ModuleList(
            [
                PreNormDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    use_rope=True,
                    max_len=max(max_past_len, max_future_len),
                    stochastic_depth_rate=stochastic_depth_rate * (i + 1) / num_decoder_layers,
                )
                for i in range(num_decoder_layers)
            ]
        )

        # Output heads
        self.point_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self.quantile_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_quantiles),
        )

        # Decomposition output heads (if enabled)
        if use_decomposition:
            self.trend_head = nn.Linear(d_model, 1)
            self.seasonal_head = nn.Linear(d_model, 1)

        # Final layer norms
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for decoder self-attention."""
        mask = torch.triu(torch.ones(sz, sz, device=device) * float("-inf"), diagonal=1)
        return mask

    def _process_static_features(
        self,
        sku: torch.Tensor,
        cats: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process static features through VSN.

        Returns:
            Tuple of (static_embedding, feature_weights).
        """
        # SKU embedding
        sku_emb = self.sku_embedding(sku)  # (batch, sku_emb_dim)

        # Category embeddings (max pool over vocabulary)
        cat_emb_list = []
        for name, emb_layer in self.cat_embeddings.items():
            cat_emb = torch.max(emb_layer(cats[name]), dim=1)[0]
            cat_emb_list.append(cat_emb)

        # Pad to same dimension for VSN
        max_dim = max(sku_emb.size(-1), cat_emb_list[0].size(-1) if cat_emb_list else 0)
        sku_padded = F.pad(sku_emb, (0, max_dim - sku_emb.size(-1)))

        static_inputs = [sku_padded]
        for cat_emb in cat_emb_list:
            cat_padded = F.pad(cat_emb, (0, max_dim - cat_emb.size(-1)))
            static_inputs.append(cat_padded)

        # Variable selection
        static_selected, feature_weights = self.static_vsn(static_inputs)
        static = self.static_proj(static_selected)

        return static, feature_weights

    def forward(
        self,
        qty: torch.Tensor,
        past_time: torch.Tensor,
        future_time: torch.Tensor,
        sku: torch.Tensor,
        cats: dict[str, torch.Tensor],
        return_attention: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            qty: Past quantities, shape (batch, past_len, 1).
            past_time: Past time features, shape (batch, past_len, time_dim).
            future_time: Future time features, shape (batch, future_len, time_dim).
            sku: SKU indices, shape (batch,).
            cats: Dict of categorical features.
            return_attention: Whether to return attention weights.

        Returns:
            Dict with keys:
                - 'prediction': Point predictions (batch, n_out)
                - 'quantiles': Quantile predictions (batch, n_out, num_quantiles)
                - 'trend': Trend component (batch, n_out) [if use_decomposition]
                - 'seasonality': Seasonal component (batch, n_out) [if use_decomposition]
                - 'feature_weights': Static feature importance (batch, num_features)
                - 'attention_weights': Attention maps [if return_attention]
        """
        device = qty.device

        # Process static features
        static, feature_weights = self._process_static_features(sku, cats)

        # Process past inputs
        past_inputs = torch.cat([qty, past_time], dim=-1)
        past_emb = self.past_proj(past_inputs)

        # Optional series decomposition
        if self.use_decomposition and self.decomposition is not None:
            seasonal, trend = self.decomposition(past_emb)
            past_emb = self.seasonal_proj(seasonal) + self.trend_proj(trend)

        # Optional patch embedding
        if self.use_patch_embedding and self.patch_embed is not None:
            past_emb = self.patch_embed(past_emb)

        # Apply FiLM conditioning with static features
        encoder_input = self.encoder_film(past_emb, static)

        # Encoder forward
        encoder_attentions = []
        for layer in self.encoder_layers:
            encoder_input, attn = layer(encoder_input, return_attention=return_attention)
            if return_attention and attn is not None:
                encoder_attentions.append(attn)

        encoder_output = self.encoder_norm(encoder_input)

        # Static enrichment
        encoder_output = self.static_enrichment(encoder_output, static)

        # Decoder input
        future_emb = self.future_proj(future_time)
        decoder_input = self.decoder_film(future_emb, static)

        # Causal mask for decoder
        tgt_len = future_time.size(1)
        tgt_mask = self._generate_causal_mask(tgt_len, device)

        # Decoder forward
        decoder_attentions = []
        for layer in self.decoder_layers:
            decoder_input, self_attn, cross_attn = layer(
                decoder_input, encoder_output, tgt_mask=tgt_mask, return_attention=return_attention
            )
            if return_attention and self_attn is not None:
                decoder_attentions.append((self_attn, cross_attn))

        decoder_output = self.decoder_norm(decoder_input)

        # Output heads
        point_pred = self.point_head(decoder_output).squeeze(-1)  # (batch, future_len)
        quantile_pred = self.quantile_head(decoder_output)  # (batch, future_len, num_quantiles)

        # Ensure output matches n_out
        if point_pred.size(1) != self.n_out:
            point_pred = F.interpolate(
                point_pred.unsqueeze(1), size=self.n_out, mode="linear", align_corners=False
            ).squeeze(1)
            quantile_pred = F.interpolate(
                quantile_pred.transpose(1, 2), size=self.n_out, mode="linear", align_corners=False
            ).transpose(1, 2)

        # Build output dictionary
        outputs = {
            "prediction": point_pred,
            "quantiles": quantile_pred,
            "feature_weights": feature_weights,
        }

        # Add decomposition outputs if enabled
        if self.use_decomposition:
            trend_out = self.trend_head(decoder_output).squeeze(-1)
            seasonal_out = self.seasonal_head(decoder_output).squeeze(-1)

            if trend_out.size(1) != self.n_out:
                trend_out = F.interpolate(
                    trend_out.unsqueeze(1), size=self.n_out, mode="linear", align_corners=False
                ).squeeze(1)
                seasonal_out = F.interpolate(
                    seasonal_out.unsqueeze(1), size=self.n_out, mode="linear", align_corners=False
                ).squeeze(1)

            outputs["trend"] = trend_out
            outputs["seasonality"] = seasonal_out

        # Add attention weights if requested
        if return_attention:
            outputs["encoder_attention"] = encoder_attentions
            outputs["decoder_attention"] = decoder_attentions

        return outputs

    def get_prediction_intervals(
        self,
        outputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract prediction intervals from model outputs.

        Args:
            outputs: Model output dictionary.

        Returns:
            Tuple of (lower_bound, median, upper_bound) tensors.
        """
        quantiles = outputs["quantiles"]
        lower = quantiles[..., 0]
        median = quantiles[..., 1]
        upper = quantiles[..., 2]
        return lower, median, upper
