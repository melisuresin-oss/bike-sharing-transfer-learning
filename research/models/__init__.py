"""Registered V2.1 neural forecasting architectures."""

from .graph_gru import GraphGRU
from .vanilla_gru import VanillaGRU

__all__ = ["GraphGRU", "VanillaGRU"]
