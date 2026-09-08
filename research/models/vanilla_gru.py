from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import (
    NeuralModelConfig,
    NeuralOutput,
    sanitize_value_mask_pair,
    to_neural_output,
    validate_redundant_mask,
)
from .heads import NonnegativePredictionHead


class VanillaGRUCell(nn.Module):
    """Explicit registered GRU cell with reset/update conventions from V2.1."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        width = self.input_size + self.hidden_size
        self.weight_reset = nn.Parameter(torch.empty(width, self.hidden_size))
        self.weight_update = nn.Parameter(torch.empty(width, self.hidden_size))
        self.weight_candidate = nn.Parameter(torch.empty(width, self.hidden_size))
        self.bias_reset = nn.Parameter(torch.zeros(self.hidden_size))
        self.bias_update = nn.Parameter(torch.zeros(self.hidden_size))
        self.bias_candidate = nn.Parameter(torch.zeros(self.hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in (self.weight_reset, self.weight_update, self.weight_candidate):
            nn.init.xavier_uniform_(weight)
        for bias in (self.bias_reset, self.bias_update, self.bias_candidate):
            nn.init.zeros_(bias)

    def forward(self, x: Tensor, hidden: Tensor, *, return_gates: bool = False):
        if x.shape[:-1] != hidden.shape[:-1]:
            raise ValueError("Input and hidden leading dimensions must match")
        joined = torch.cat([x, hidden], dim=-1)
        reset = torch.sigmoid(joined @ self.weight_reset + self.bias_reset)
        update = torch.sigmoid(joined @ self.weight_update + self.bias_update)
        candidate_joined = torch.cat([x, reset * hidden], dim=-1)
        candidate = torch.tanh(candidate_joined @ self.weight_candidate + self.bias_candidate)
        next_hidden = update * hidden + (1.0 - update) * candidate
        if return_gates:
            return next_hidden, {"reset": reset, "update": update, "candidate": candidate}
        return next_hidden


class VanillaGRU(nn.Module):
    """Station-independent one-layer recurrent control with no graph object."""

    def __init__(self, config: NeuralModelConfig):
        super().__init__()
        if config.model_type != "vanilla_gru":
            raise ValueError("VanillaGRU requires model_type='vanilla_gru'")
        self.config = config
        self.input_dropout = nn.Dropout(config.dropout)
        self.cell = VanillaGRUCell(config.input_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.dropout)
        self.head = NonnegativePredictionHead(
            config.hidden_size, config.context_size, config.output_mode
        )

    def forward(
        self,
        x_hist: Tensor,
        x_week: Tensor,
        x_static: Tensor,
        x_calendar: Tensor,
        m_hist: Tensor | None = None,
    ) -> NeuralOutput:
        if x_hist.ndim != 3 or x_hist.shape[1:] != (24, 2):
            raise ValueError("Vanilla X_hist must be [B,24,2]")
        validate_redundant_mask(x_hist, m_hist)
        # Loader columns are lag 1..24; recurrence runs chronologically t-24..t-1.
        recurrent_input = self.input_dropout(
            torch.flip(sanitize_value_mask_pair(x_hist), dims=(1,))
        )
        hidden = x_hist.new_zeros((x_hist.shape[0], self.config.hidden_size))
        for step in range(24):
            hidden = self.cell(recurrent_input[:, step, :], hidden)
        week = sanitize_value_mask_pair(x_week)
        if week.shape != (x_hist.shape[0], 2):
            raise ValueError("Vanilla X_week must be [B,2]")
        if x_static.shape != (x_hist.shape[0], 2) or x_calendar.shape != (x_hist.shape[0], 6):
            raise ValueError("Vanilla static/calendar shapes violate the loader contract")
        context = torch.cat([week, x_static, x_calendar], dim=-1)
        head = self.head(self.output_dropout(hidden), context)
        return to_neural_output(head, hidden)
