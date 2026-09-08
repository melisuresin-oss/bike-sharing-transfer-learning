from __future__ import annotations

from torch import nn

from .common import NeuralModelConfig
from .graph_gru import GraphGRU
from .vanilla_gru import VanillaGRU


def build_model(config: NeuralModelConfig) -> nn.Module:
    if config.model_type == "vanilla_gru":
        return VanillaGRU(config)
    if config.model_type == "graph_gru":
        return GraphGRU(config)
    raise ValueError(f"Unknown registered model: {config.model_type}")


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
