from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]


def neural_code_hash() -> str:
    digest = hashlib.sha256()
    for folder in ("models", "training", "graphs"):
        for path in sorted((ROOT / "research" / folder).rglob("*.py")):
            digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointMetadata:
    architecture_config: dict[str, Any]
    optimizer_config: dict[str, Any]
    graph_config: dict[str, Any] | None
    protocol_version: str
    feature_schema_sha256: str
    city_roster_sha256: str
    seed: int
    epoch: int
    step: int
    scale_loss_mode: str
    code_sha256: str


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: CheckpointMetadata,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metadata": asdict(metadata),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path.stat().st_size


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    expected_feature_schema_sha256: str,
    expected_city_roster_sha256: str,
    expected_graph_sha256: str | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    if metadata["protocol_version"] != "2.1":
        raise RuntimeError("Checkpoint protocol version is incompatible")
    if metadata["feature_schema_sha256"] != expected_feature_schema_sha256:
        raise RuntimeError("Checkpoint feature schema hash is incompatible")
    if metadata["city_roster_sha256"] != expected_city_roster_sha256:
        raise RuntimeError("Checkpoint station roster hash is incompatible")
    actual_graph = (metadata.get("graph_config") or {}).get("adjacency_hash_sha256")
    if expected_graph_sha256 is not None and actual_graph != expected_graph_sha256:
        raise RuntimeError("Checkpoint graph hash is incompatible")
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return metadata
