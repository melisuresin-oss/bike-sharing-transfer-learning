from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch

from research.development.stage2_graphgru_selection import (
    assert_batch_device,
    configure_deterministic_device,
)
from research.models.common import NeuralModelConfig
from research.models.model_factory import build_model
from research.training.trainer import NeuralTrainer, OptimizerConfig, TrainerConfig


def fixed_graph_batch(device: torch.device, *, nodes: int = 8, batch_size: int = 2):
    adjacency = torch.eye(nodes, dtype=torch.float32, device=device)
    values = torch.arange(
        batch_size * 24 * nodes, dtype=torch.float32, device=device
    ).reshape(batch_size, 24, nodes)
    values = torch.remainder(values, 11.0) / 10.0
    observed = torch.ones_like(values)
    x_hist = torch.stack((values, observed), dim=-1)
    week_values = values[:, 0, :]
    x_week = torch.stack((week_values, torch.ones_like(week_values)), dim=-1)
    x_static = torch.stack(
        (
            torch.linspace(-1.0, 1.0, nodes, device=device),
            torch.linspace(1.0, -1.0, nodes, device=device),
        ),
        dim=-1,
    )
    x_calendar = torch.zeros((batch_size, 6), dtype=torch.float32, device=device)
    target = torch.remainder(
        torch.arange(batch_size * nodes, dtype=torch.float32, device=device), 5.0
    ).reshape(batch_size, nodes)
    batch = {
        "model_inputs": {
            "x_hist": x_hist,
            "m_hist": x_hist[..., 1] > 0.5,
            "x_week": x_week,
            "x_static": x_static,
            "x_calendar": x_calendar,
            "adjacency": adjacency,
        },
        "target": target,
        "mask": torch.ones_like(target, dtype=torch.bool),
        "city_ids": torch.zeros(batch_size, dtype=torch.int64, device=device),
        "station_ids": torch.arange(nodes, dtype=torch.int64, device=device),
    }
    assert_batch_device(batch, device)
    return batch


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def run_fixed_training_fixture(
    requested_device: str, *, seed: int = 17, updates: int = 4
) -> dict[str, Any]:
    device, deterministic = configure_deterministic_device(seed, requested_device)
    config = NeuralModelConfig(
        model_type="graph_gru",
        input_size=2,
        hidden_size=8,
        context_size=10,
        dropout=0.0,
        output_mode="log1p_target",
        recurrent_layers=1,
    )
    model = build_model(config).to(device)
    batch = fixed_graph_batch(device)
    trainer = NeuralTrainer(
        model,
        OptimizerConfig(learning_rate=1e-3, weight_decay=1e-4),
        TrainerConfig(
            seed=seed,
            output_mode="log1p_target",
            gradient_clip_global_norm=1.0,
            checkpoint_mode="final",
        ),
    )
    losses: list[float] = []
    for _ in range(int(updates)):
        losses.append(trainer.train_step(batch).loss)
    model.eval()
    with torch.no_grad():
        prediction = model(**batch["model_inputs"]).count_prediction.detach().cpu().numpy()
    return {
        "prediction": prediction,
        "state_sha256": _state_hash(model),
        "losses": np.asarray(losses, dtype=np.float64),
        "resolved_device": str(device),
        "deterministic_settings": deterministic,
        "parameter_devices": sorted({str(item.device) for item in model.parameters()}),
    }
