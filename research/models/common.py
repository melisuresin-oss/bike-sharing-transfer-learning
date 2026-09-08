from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from .heads import OutputMode, PredictionHeadOutput


@dataclass(frozen=True)
class NeuralModelConfig:
    model_type: str
    input_size: int = 2
    hidden_size: int = 64
    context_size: int = 10
    dropout: float = 0.10
    output_mode: OutputMode = "raw_count"
    recurrent_layers: int = 1

    def __post_init__(self) -> None:
        if self.model_type not in ("vanilla_gru", "graph_gru"):
            raise ValueError(f"Unknown model type: {self.model_type}")
        if self.input_size != 2 or self.context_size != 10 or self.recurrent_layers != 1:
            raise ValueError("V2.1 fixes input=2, context=10, and one recurrent layer")
        if self.hidden_size <= 0 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("Invalid hidden size or dropout")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NeuralOutput:
    prediction: Tensor
    count_prediction: Tensor
    hidden: Tensor


def sanitize_value_mask_pair(pair: Tensor) -> Tensor:
    """Enforce placeholder semantics while retaining the indicator as a channel."""
    if pair.shape[-1] != 2:
        raise ValueError("Value/mask pairs must have exactly two channels")
    mask = pair[..., 1].to(dtype=pair.dtype)
    return torch.stack([pair[..., 0] * mask, mask], dim=-1)


def validate_redundant_mask(pair: Tensor, mask: Tensor | None) -> None:
    """M_hist is a contract check, not a third recurrent feature channel."""
    if mask is None:
        return
    expected = pair[..., 1] > 0.5
    if expected.shape != mask.shape or not torch.equal(expected, mask.to(dtype=torch.bool)):
        raise ValueError("M_hist disagrees with the observation-indicator channel in X_hist")


def to_neural_output(head: PredictionHeadOutput, hidden: Tensor) -> NeuralOutput:
    return NeuralOutput(
        prediction=head.prediction,
        count_prediction=head.count_prediction,
        hidden=hidden,
    )
