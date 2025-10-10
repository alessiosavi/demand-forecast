import math
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

import datasets


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class AdvancedDemandForecastModel(nn.Module):
    def __init__(
        self,
        sku_vocab_size: int,
        sku_emb_dim: int,
        cat_features_dim: Dict[str, int],
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
        **kwargs,
    ):
        super(AdvancedDemandForecastModel, self).__init__()

        # SKU Embedding
        self.sku_embedding = nn.Embedding(sku_vocab_size, sku_emb_dim)

        # Categorical Feature Embeddings
        self.cat_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(vocab_size, cat_emb_dims)
                for name, vocab_size in cat_features_dim.items()
            }
        )

        # Total static embedding dim
        total_cat_emb_dim = cat_emb_dims * len(cat_features_dim) + sku_emb_dim

        # Projection for static features
        self.static_proj = nn.Linear(total_cat_emb_dim, d_model)

        # Projection for past time-series inputs (qty + past_time)
        self.past_proj = nn.Linear(past_time_features_dim, d_model)

        # Projection for future time features
        self.future_proj = nn.Linear(future_time_features_dim, d_model)

        # Positional encodings
        self.pos_enc = PositionalEncoding(d_model, max_len=max_past_len)
        self.dec_pos_enc = PositionalEncoding(d_model, max_len=max_future_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Output heads
        self.reg_head = nn.Linear(d_model, n_out)  # Regression output per future step

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)
        return mask

    def forward(self, qty, past_time, future_time, sku, cats, **kwargs):
        # SKU Embedding
        sku_emb = self.sku_embedding(sku)
        # Categorical Embeddings (using max pooling as in original)
        cat_embs = [
            torch.max(emb(cats[name]), dim=1)[0]
            for name, emb in self.cat_embeddings.items()
        ]
        cat_embs = torch.cat(cat_embs, dim=1)
        # Combined static features
        static_emb = torch.cat([sku_emb, cat_embs], dim=1)
        static = self.static_proj(static_emb)  # [batch_size, d_model]

        # Encoder: Process past time-series
        past_inputs = torch.cat(
            [qty, past_time], dim=-1
        )  # [batch_size, past_len, past_input_dim]
        past_emb = self.past_proj(past_inputs)  # [batch_size, past_len, d_model]

        static_repeated = static.unsqueeze(1).repeat(1, past_inputs.size(1), 1)
        encoder_input = self.pos_enc(past_emb + static_repeated)

        encoder_output = self.transformer_encoder(encoder_input)

        # Decoder: Process future time features with cross-attention to encoder
        future_emb = self.future_proj(future_time)  # [batch_size, future_len, d_model]
        dec_static_repeated = static.unsqueeze(1).repeat(1, future_time.size(1), 1)
        decoder_input = self.dec_pos_enc(future_emb + dec_static_repeated)
        tgt_mask = self._generate_square_subsequent_mask(future_time.size(1)).to(
            future_time.device
        )
        decoder_output = self.transformer_decoder(
            tgt=decoder_input, memory=encoder_output, tgt_mask=tgt_mask
        )

        # Outputs
        reg_output = self.reg_head(decoder_output).squeeze(-1)  # [batch_size, n_out]

        return reg_output


class ModelWrapper(nn.Module):
    def __init__(self, n: int, **kwargs):
        super(ModelWrapper, self).__init__()

        self.models = nn.ModuleDict(
            {f"{i}": AdvancedDemandForecastModel(**kwargs) for i in range(n)}
        )

    def forward(self, n, qty, past_time, future_time, sku, cats, **kwargs):
        model = self.models[n]
        return model(qty, past_time, future_time, sku, cats, **kwargs)


def init_metrics():
    metrics_names = [v for v in dir(torchmetrics.regression) if v[0].isupper()]
    metrics = {}
    target = torch.tensor([[2.5, 5, 4, 8], [3, 5, 2.5, 7]])
    preds = torch.tensor([[3, 5, 2.5, 7], [2.5, 5, 4, 8]])
    for metric_name in metrics_names:
        try:
            metric = getattr(torchmetrics, metric_name)()
            metric(preds, target)
            metrics[metric_name] = metric
        except Exception as e:
            print("Skipping ", metric_name, e)
    del metrics["KLDivergence"]
    del metrics["CosineSimilarity"]
    return metrics


def core(model_idx, batch, regression_criterion, models, flatten=True):
    qty = batch["qty"]
    past_time = batch["past_time"]
    future_time = batch["future_time"]
    sku = batch["sku"]
    # Maybe it is not necessary. To save memory, we save the categorical matrix data as boolean (True/False instead of 1/0)
    cats = {key: value.to(dtype=torch.int32) for key, value in batch["cats"].items()}
    targets = batch["y"]

    outputs = models(model_idx, qty, past_time, future_time, sku, cats)
    # Use the sum of each value to reduce "global batch distance" from targets (from `[batch_size, n_out]` to `[batch_size]`)
    if flatten:
        flatten_outputs = torch.sum(outputs, dim=-1)
        flatten_targets = torch.sum(targets, dim=-1)
    else:
        flatten_outputs = outputs
        flatten_targets = targets

    loss = regression_criterion(flatten_outputs, flatten_targets)

    return loss, outputs, targets, flatten_outputs, flatten_targets


# Validation on the test dataset
def validate_model(
    models,
    dataloader,
    regression_criterion,
    batch_size,
    total_examples_test,
    metrics,
    plot=False,
):
    models.eval()
    total_loss = 0.0
    flatten_predictions, flatten_actuals, predictions, actuals = [], [], [], []
    _skus = []
    total_steps = total_examples_test // batch_size

    with torch.no_grad():
        for item in tqdm(dataloader, total=total_steps, leave=False):
            model_idx = list(item.keys())[0]
            batch = item[model_idx]
            loss, outputs, targets, flatten_outputs, flatten_targets = core(
                model_idx, batch, regression_criterion, models
            )
            total_loss += loss.item()

            # Store predictions and actual values
            flatten_predictions.extend(flatten_outputs.squeeze().detach().cpu().numpy())
            flatten_actuals.extend(flatten_targets.detach().cpu().numpy())
            predictions.extend(outputs.squeeze().detach().cpu().numpy())
            actuals.extend(targets.detach().cpu().numpy())
            _skus.extend(batch["sku"].detach().cpu().numpy())

    avg_loss = total_loss / total_steps

    # Calculate performance metrics
    _actuals = np.array(actuals)
    _predictions = np.array(predictions)
    res = _actuals - _predictions

    _flatten_actuals = np.array(flatten_actuals)
    _flatten_predictions = np.array(flatten_predictions)
    flatten_res = _flatten_actuals - _flatten_predictions

    mse = np.mean(res**2)
    mae = np.mean(np.abs(res))
    flatten_mse = np.mean(flatten_res**2)
    flatten_mae = np.mean(np.abs(flatten_res))

    # Plot predictions vs actuals
    s_res = f"Loss: {avg_loss:.4f} MSE: {mse:.4f} MAE: {mae:.4f} FLAT_MSE: {flatten_mse:.4f} FLAT_MAE: {flatten_mae:.4f}"
    if plot:
        plt.figure(figsize=(20, 10))
        plt.plot(flatten_actuals, label="Actual", color="blue")
        plt.plot(
            flatten_predictions, label="Predicted", color="red", linestyle="dashed"
        )
        plt.title(s_res)
        plt.xlabel("Sample Index")
        plt.ylabel("Quantity")
        plt.legend(loc="upper right")
        plt.grid()
        plt.show()

    print(f"Validation Results:\n{s_res}")
    _p, _a = torch.as_tensor(flatten_predictions), torch.as_tensor(flatten_actuals)
    _res_metric = {}
    for metric_name, metric in metrics.items():
        try:
            _res_metric[metric_name] = metric(_p, _a).item()
        except Exception:
            print("skipping", metric_name)

    return {
        "predictions": predictions,
        "actuals": actuals,
        "flatten_predictions": flatten_predictions,
        "flatten_actuals": flatten_actuals,
        "skus": _skus,
        "avg_loss": avg_loss,
        "mse": mse,
        "mae": mae,
        "flatten_mse": flatten_mse,
        "flatten_mae": flatten_mae,
        "metrics": _res_metric,
    }


# Training Loop
def train_model(
    models,
    dataloader_train,
    dataloader_test,
    regression_criterion,
    optimizers: Dict[str, torch.optim.Optimizer],
    schedulers: Dict[str, torch.optim.lr_scheduler.LRScheduler],
    num_epochs,
    batch_size,
    early_stop: Dict[str, int],
    plot_n_epochs=3,
    total_examples_train=0,
    total_examples_test=0,
    flatten=True,
    metrics={},
):

    total_steps = total_examples_train // batch_size

    best_metric = float("inf")
    epochs_no_improve = 0
    best_model = None
    with tqdm(range(num_epochs), position=0) as pbar:
        for epoch in pbar:
            dls_train, dls_test = datasets.get_data(dataloader_train, dataloader_test)
            epoch_loss = 0.0
            models.train()
            for i, item in tqdm(
                enumerate(dls_train), position=1, leave=False, total=total_steps
            ):
                model_idx = list(item.keys())[0]
                batch = item[model_idx]
                loss, _, _, _, _ = core(
                    model_idx, batch, regression_criterion, models, flatten
                )

                # Backward pass and optimization
                optimizers[model_idx].zero_grad()
                loss.backward()
                optimizers[model_idx].step()
                epoch_loss += loss.item()
                pbar.set_description(
                    f"Epoch [{epoch + 1}/{num_epochs}] - LOSS: {(epoch_loss / (i+1)):.4f}"
                )

            for k in schedulers:
                schedulers[k].step()
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / total_steps:.4f}"
            )
            val_metrics = validate_model(
                models=models,
                dataloader=dls_test,
                regression_criterion=regression_criterion,
                batch_size=batch_size,
                total_examples_test=total_examples_test,
                metrics=metrics,
                plot=(epoch + 1) % plot_n_epochs == 0,
            )
            metric = None
            if flatten:
                metric = val_metrics["flatten_mse"]
            else:
                metric = val_metrics["mse"]

            if metric < best_metric:
                best_model = deepcopy(models.state_dict())
                best_metric = metric
                # if flatten_mse < (best_metric - early_stop["min_delta"]):
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop["patience"]:
                    print(f"Early stopping triggered after {epoch + 1} epochs!")
                    models.load_state_dict(best_model)  # load the best model
                    return
