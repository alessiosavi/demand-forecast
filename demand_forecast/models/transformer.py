"""Transformer-based demand forecasting model."""

import torch
import torch.nn as nn

from demand_forecast.models.components import (
    FiLMConditioning,
    PositionalEncoding,
    RotaryPositionEmbedding,
    StochasticDepth,
)


class AdvancedDemandForecastModel(nn.Module):
    """Transformer encoder-decoder model for demand forecasting.

    Architecture:
        1. Static embeddings (SKU + categories) projected to d_model
        2. Encoder processes past time series with cross-attention to static
        3. Decoder generates future predictions with causal masking

    Attributes:
        sku_embedding: Embedding layer for SKU indices.
        cat_embeddings: ModuleDict of categorical embeddings.
        static_proj: Linear projection for combined static features.
        past_proj: Linear projection for past time features.
        future_proj: Linear projection for future time features.
        pos_enc: Positional encoding for encoder.
        dec_pos_enc: Positional encoding for decoder.
        transformer_encoder: Transformer encoder stack.
        transformer_decoder: Transformer decoder stack.
        reg_head: Output regression head.
    """

    def __init__(
        self,
        sku_vocab_size: int,
        sku_emb_dim: int,
        cat_features_dim: dict[str, int],
        cat_emb_dims: int,
        past_time_features_dim: int,
        future_time_features_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        n_out: int,
        max_past_len: int = 100,
        max_future_len: int = 50,
        # New parameters for improved architecture
        use_rope: bool = False,
        use_pre_layernorm: bool = False,
        use_film_conditioning: bool = False,
        use_improved_head: bool = False,
        stochastic_depth_rate: float = 0.0,
        **kwargs,
    ):
        """Initialize the demand forecast model.

        Args:
            sku_vocab_size: Number of unique SKUs.
            sku_emb_dim: SKU embedding dimension.
            cat_features_dim: Dictionary mapping category names to vocab sizes.
            cat_emb_dims: Categorical embedding dimension.
            past_time_features_dim: Dimension of past time features.
            future_time_features_dim: Dimension of future time features.
            d_model: Transformer model dimension.
            nhead: Number of attention heads.
            num_encoder_layers: Number of encoder layers.
            num_decoder_layers: Number of decoder layers.
            dim_feedforward: Feedforward network dimension.
            dropout: Dropout rate.
            n_out: Number of output time steps.
            max_past_len: Maximum past sequence length.
            max_future_len: Maximum future sequence length.
            use_rope: Use Rotary Position Embeddings instead of sinusoidal.
            use_pre_layernorm: Use Pre-LayerNorm for better training stability.
            use_film_conditioning: Use FiLM for static feature integration.
            use_improved_head: Use improved output head with GELU.
            stochastic_depth_rate: Drop rate for stochastic depth (0 = disabled).
        """
        super().__init__()

        # Store configuration
        self.use_rope = use_rope
        self.use_pre_layernorm = use_pre_layernorm
        self.use_film_conditioning = use_film_conditioning
        self.use_improved_head = use_improved_head
        self.stochastic_depth_rate = stochastic_depth_rate
        self.d_model = d_model
        self.nhead = nhead
        self.n_out = n_out

        # SKU Embedding
        self.sku_embedding = nn.Embedding(sku_vocab_size, sku_emb_dim)

        # Categorical Feature Embeddings
        self.cat_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(vocab_size, cat_emb_dims)
                for name, vocab_size in cat_features_dim.items()
            }
        )

        # Total static embedding dimension
        total_cat_emb_dim = cat_emb_dims * len(cat_features_dim) + sku_emb_dim

        # Projection layers
        self.static_proj = nn.Linear(total_cat_emb_dim, d_model)
        self.past_proj = nn.Linear(past_time_features_dim, d_model)
        self.future_proj = nn.Linear(future_time_features_dim, d_model)

        # Positional encodings (RoPE or sinusoidal)
        if use_rope:
            head_dim = d_model // nhead
            self.rope = RotaryPositionEmbedding(head_dim, max_len=max(max_past_len, max_future_len))
            self.pos_enc = None
            self.dec_pos_enc = None
        else:
            self.rope = None
            self.pos_enc = PositionalEncoding(d_model, max_len=max_past_len)
            self.dec_pos_enc = PositionalEncoding(d_model, max_len=max_future_len)

        # FiLM conditioning for static features
        if use_film_conditioning:
            self.encoder_film = FiLMConditioning(d_model, d_model)
            self.decoder_film = FiLMConditioning(d_model, d_model)
        else:
            self.encoder_film = None
            self.decoder_film = None

        # Transformer Encoder with Pre-LayerNorm option
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=use_pre_layernorm,  # Pre-LN when True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Transformer Decoder with Pre-LayerNorm option
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=use_pre_layernorm,  # Pre-LN when True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Stochastic depth for encoder/decoder
        if stochastic_depth_rate > 0:
            self.stochastic_depth = StochasticDepth(stochastic_depth_rate)
        else:
            self.stochastic_depth = None

        # Output head (improved or simple)
        if use_improved_head:
            self.reg_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )
        else:
            self.reg_head = nn.Linear(d_model, n_out)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder.

        Args:
            sz: Sequence length.

        Returns:
            Upper triangular mask with -inf for future positions.
        """
        mask = torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)
        return mask

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
            qty: Past quantity values, shape (batch_size, past_len, 1).
            past_time: Past time features, shape (batch_size, past_len, time_dim).
            future_time: Future time features, shape (batch_size, future_len, time_dim).
            sku: SKU indices, shape (batch_size,).
            cats: Dictionary of categorical features, each shape (batch_size, vocab_size).

        Returns:
            Predicted quantities, shape (batch_size, n_out).
        """
        # SKU Embedding
        sku_emb = self.sku_embedding(sku)

        # Categorical Embeddings (max pooling over vocabulary)
        cat_embs = [
            torch.max(emb(cats[name]), dim=1)[0] for name, emb in self.cat_embeddings.items()
        ]
        cat_embs = torch.cat(cat_embs, dim=1)

        # Combined static features
        static_emb = torch.cat([sku_emb, cat_embs], dim=1)
        static = self.static_proj(static_emb)  # [batch_size, d_model]

        # Encoder: Process past time-series
        past_inputs = torch.cat([qty, past_time], dim=-1)
        past_emb = self.past_proj(past_inputs)

        # Apply static conditioning (FiLM or simple addition)
        if self.use_film_conditioning and self.encoder_film is not None:
            encoder_input = self.encoder_film(past_emb, static)
        else:
            static_repeated = static.unsqueeze(1).repeat(1, past_inputs.size(1), 1)
            encoder_input = past_emb + static_repeated

        # Apply positional encoding (sinusoidal, RoPE handled internally by attention)
        if self.pos_enc is not None:
            encoder_input = self.pos_enc(encoder_input)

        encoder_output = self.transformer_encoder(encoder_input)

        # Decoder: Process future time features with cross-attention
        future_emb = self.future_proj(future_time)

        # Apply static conditioning to decoder
        if self.use_film_conditioning and self.decoder_film is not None:
            decoder_input = self.decoder_film(future_emb, static)
        else:
            dec_static_repeated = static.unsqueeze(1).repeat(1, future_time.size(1), 1)
            decoder_input = future_emb + dec_static_repeated

        # Apply positional encoding
        if self.dec_pos_enc is not None:
            decoder_input = self.dec_pos_enc(decoder_input)

        tgt_mask = self._generate_square_subsequent_mask(future_time.size(1)).to(future_time.device)
        decoder_output = self.transformer_decoder(
            tgt=decoder_input, memory=encoder_output, tgt_mask=tgt_mask
        )

        # Output
        if self.use_improved_head:
            # Improved head outputs per-timestep (1 value each), take last n_out
            reg_output = self.reg_head(decoder_output).squeeze(-1)
            # If decoder outputs more than n_out, take last n_out steps
            if reg_output.size(-1) > self.n_out:
                reg_output = reg_output[:, -self.n_out :]
        else:
            # Simple head: use last decoder position to predict all n_out values
            last_hidden = decoder_output[:, -1, :]  # [batch, d_model]
            reg_output = self.reg_head(last_hidden)  # [batch, n_out]

        return reg_output
