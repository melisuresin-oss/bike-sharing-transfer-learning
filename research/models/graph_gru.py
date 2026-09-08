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


def graph_aggregate(adjacency: Tensor, features: Tensor) -> Tensor:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency must be square")
    if features.shape[-2] != adjacency.shape[0]:
        raise ValueError("Adjacency node count differs from features")
    if adjacency.layout != torch.strided:
        raise ValueError("The validated Stage-1 kernel expects dense normalized adjacency")
    return torch.matmul(adjacency, features)


def graph_propagate(adjacency: Tensor, features: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    return graph_aggregate(adjacency, features) @ weight + bias


class GraphGRUCell(nn.Module):
    """Symmetric-GCN propagation inside each registered GRU gate."""

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

    def forward(self, x: Tensor, hidden: Tensor, adjacency: Tensor, *, return_gates: bool = False):
        if x.ndim != 3 or hidden.ndim != 3 or x.shape[:2] != hidden.shape[:2]:
            raise ValueError("Graph cell expects X [B,N,F] and H [B,N,H]")
        joined = torch.cat([x, hidden], dim=-1)
        # Reset and update use the identical A_hat[joined] term.  Reuse that
        # algebraically common subexpression; gate equations and parameters are
        # unchanged while one dense graph aggregation per hour is eliminated.
        aggregated_joined = graph_aggregate(adjacency, joined)
        reset = torch.sigmoid(aggregated_joined @ self.weight_reset + self.bias_reset)
        update = torch.sigmoid(aggregated_joined @ self.weight_update + self.bias_update)
        candidate_joined = torch.cat([x, reset * hidden], dim=-1)
        candidate = torch.tanh(
            graph_propagate(
                adjacency, candidate_joined, self.weight_candidate, self.bias_candidate
            )
        )
        next_hidden = update * hidden + (1.0 - update) * candidate
        if return_gates:
            return next_hidden, {"reset": reset, "update": update, "candidate": candidate}
        return next_hidden


class GraphGRU(nn.Module):
    def __init__(self, config: NeuralModelConfig):
        super().__init__()
        if config.model_type != "graph_gru":
            raise ValueError("GraphGRU requires model_type='graph_gru'")
        self.config = config
        self.input_dropout = nn.Dropout(config.dropout)
        self.cell = GraphGRUCell(config.input_size, config.hidden_size)
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
        adjacency: Tensor,
        m_hist: Tensor | None = None,
    ) -> NeuralOutput:
        if x_hist.ndim != 4 or x_hist.shape[1] != 24 or x_hist.shape[-1] != 2:
            raise ValueError("Graph X_hist must be [B,24,N,2]")
        validate_redundant_mask(x_hist, m_hist)
        batch, _, nodes, _ = x_hist.shape
        if adjacency.shape != (nodes, nodes):
            raise ValueError("Adjacency does not match graph batch node count")
        # Loader columns are lag 1..24; recurrence runs chronologically t-24..t-1.
        recurrent_input = self.input_dropout(
            torch.flip(sanitize_value_mask_pair(x_hist), dims=(1,))
        )
        hidden = x_hist.new_zeros((batch, nodes, self.config.hidden_size))
        for step in range(24):
            hidden = self.cell(recurrent_input[:, step, :, :], hidden, adjacency)
        week = sanitize_value_mask_pair(x_week)
        if week.shape != (batch, nodes, 2):
            raise ValueError("Graph X_week must be [B,N,2]")
        if x_static.shape == (nodes, 2):
            static = x_static.unsqueeze(0).expand(batch, -1, -1)
        elif x_static.shape == (batch, nodes, 2):
            static = x_static
        else:
            raise ValueError("Graph X_static must be [N,2] or [B,N,2]")
        if x_calendar.shape != (batch, 6):
            raise ValueError("Graph X_calendar must be [B,6]")
        calendar = x_calendar[:, None, :].expand(-1, nodes, -1)
        context = torch.cat([week, static, calendar], dim=-1)
        head = self.head(self.output_dropout(hidden), context)
        return to_neural_output(head, hidden)
