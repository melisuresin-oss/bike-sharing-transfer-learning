"""GRU-based next-hour demand forecaster.

Architecture, per the proposal: a shared GRU processes the 24-hour demand
sequence -- one scalar (departures) per timestep, shape (batch, 24, 1). Its
final hidden state is concatenated with the 10 static covariates (constant
per sample, not time-varying, so they never go through the GRU itself) and
passed through a linear output layer to predict next-hour departures.

Two lessons from a Colab run baked in here as non-negotiable defaults:

1. No activation (softplus, ReLU, ...) on the output. The model predicts in
   *scaled* space (src/features/windowing.py's per-city demand scaler),
   where the scaled city mean sits at 0. A softplus output floors
   predictions at 0 in scaled space -- which is well *above* the city's
   mean for a sparse demand distribution -- so the model becomes
   structurally unable to predict below-average demand. Negativity in the
   final, inverse-transformed raw prediction is a display/evaluation
   concern (clip after inverse_transform_target, in src/training/train.py),
   never a constraint baked into the network.
2. This module defines the network only -- no loss function, no scaling or
   unscaling, no training loop. That lives in src/training/train.py, kept
   separate because the same model gets trained under three different
   regimes (target-only, pooled, source-pretrained-then-fine-tuned) that
   share the architecture but differ in what data feeds it and when.
"""
import torch
import torch.nn as nn


class GRUForecaster(nn.Module):
    def __init__(
        self,
        n_covariates: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        # nn.GRU's own dropout only applies *between* stacked layers, and
        # torch warns (harmlessly, but noisily) if passed with num_layers=1.
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        gru_output_size = hidden_size * (2 if bidirectional else 1)
        self.head_dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(gru_output_size + n_covariates, 1)

    def forward(self, sequence: torch.Tensor, covariates: torch.Tensor) -> torch.Tensor:
        """
        sequence: (batch, seq_len, 1) -- the 24-hour demand window, oldest first
        covariates: (batch, n_covariates) -- static, non-time-varying

        Returns (batch,): raw linear output in SCALED space. No activation --
        see the module docstring.
        """
        _, h_n = self.gru(sequence)
        # h_n: (num_layers * num_directions, batch, hidden_size). Concatenate
        # the last layer's forward (and backward, if bidirectional) hidden
        # state -- matches gru_output_size above.
        if self.gru.bidirectional:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            last_hidden = h_n[-1]
        combined = self.head_dropout(torch.cat([last_hidden, covariates], dim=-1))
        return self.output_layer(combined).squeeze(-1)


def build_model_from_config(config: dict, n_covariates: int) -> GRUForecaster:
    model_cfg = config["model"]
    return GRUForecaster(
        n_covariates=n_covariates,
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        bidirectional=model_cfg.get("bidirectional", False),
    )
