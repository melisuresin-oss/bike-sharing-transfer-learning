from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


OutputMode = Literal["raw_count", "log1p_target"]


@dataclass(frozen=True)
class PredictionHeadOutput:
    prediction: Tensor
    count_prediction: Tensor


class NonnegativePredictionHead(nn.Module):
    """One shared linear node-wise head followed by the registered Softplus."""

    def __init__(self, hidden_size: int, context_size: int = 10, output_mode: OutputMode = "raw_count"):
        super().__init__()
        if output_mode not in ("raw_count", "log1p_target"):
            raise ValueError(f"Unregistered output mode: {output_mode}")
        self.hidden_size = int(hidden_size)
        self.context_size = int(context_size)
        self.output_mode: OutputMode = output_mode
        self.linear = nn.Linear(self.hidden_size + self.context_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, hidden: Tensor, context: Tensor) -> PredictionHeadOutput:
        if hidden.shape[:-1] != context.shape[:-1]:
            raise ValueError("Hidden and context leading dimensions must match")
        if hidden.shape[-1] != self.hidden_size or context.shape[-1] != self.context_size:
            raise ValueError("Hidden/context channel count violates the frozen head contract")
        prediction = self.softplus(self.linear(torch.cat([hidden, context], dim=-1))).squeeze(-1)
        count_prediction = (
            prediction
            if self.output_mode == "raw_count"
            else torch.expm1(prediction).clamp_min(0.0)
        )
        return PredictionHeadOutput(prediction=prediction, count_prediction=count_prediction)
