from __future__ import annotations

from typing import Any

import numpy as np
import torch

from research.data.graph_dataset import GraphWindowBatch
from research.data.window_dataset import StationWindowBatch


def _tensor(array: np.ndarray, *, dtype=None, device: str | torch.device = "cpu"):
    return torch.as_tensor(array, dtype=dtype, device=device)


def station_batch_to_torch(batch: StationWindowBatch, device="cpu") -> dict[str, Any]:
    return {
        "model_inputs": {
            "x_hist": _tensor(batch.x_hist, dtype=torch.float32, device=device),
            "m_hist": _tensor(batch.m_hist, dtype=torch.bool, device=device),
            "x_week": _tensor(batch.x_week, dtype=torch.float32, device=device),
            "x_static": _tensor(batch.x_static, dtype=torch.float32, device=device),
            "x_calendar": _tensor(batch.x_calendar, dtype=torch.float32, device=device),
        },
        "target": None if batch.y is None else _tensor(batch.y, dtype=torch.float32, device=device),
        "mask": _tensor(batch.m_target, dtype=torch.bool, device=device),
        "city_ids": _tensor(batch.city_ids, dtype=torch.int64, device=device),
        "station_ids": _tensor(batch.station_ids, dtype=torch.int64, device=device),
    }


def graph_batch_to_torch(batch: GraphWindowBatch, adjacency, device="cpu") -> dict[str, Any]:
    return {
        "model_inputs": {
            "x_hist": _tensor(batch.x_hist, dtype=torch.float32, device=device),
            "m_hist": _tensor(batch.m_hist, dtype=torch.bool, device=device),
            "x_week": _tensor(batch.x_week, dtype=torch.float32, device=device),
            "x_static": _tensor(batch.x_static, dtype=torch.float32, device=device),
            "x_calendar": _tensor(batch.x_calendar, dtype=torch.float32, device=device),
            "adjacency": _tensor(adjacency, dtype=torch.float32, device=device),
        },
        "target": None if batch.y is None else _tensor(batch.y, dtype=torch.float32, device=device),
        "mask": _tensor(batch.m_target, dtype=torch.bool, device=device),
        "city_ids": torch.full((len(batch.timestamps),), batch.city_id, dtype=torch.int64, device=device),
        "station_ids": _tensor(batch.station_ids, dtype=torch.int64, device=device),
    }
