from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import nn

from .losses import log1p_target_mae, raw_count_mae


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "AdamW"
    learning_rate: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        if self.name != "AdamW":
            raise ValueError("V2.1 registers AdamW only")


@dataclass(frozen=True)
class TrainerConfig:
    seed: int
    output_mode: Literal["raw_count", "log1p_target"]
    gradient_clip_global_norm: float = 1.0
    checkpoint_mode: Literal["final", "best_fixed_validation"] = "final"


@dataclass(frozen=True)
class StepLog:
    step: int
    loss: float
    valid_targets: int
    gradient_norm_before_clip: float
    learning_rate: float


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )


class NeuralTrainer:
    """Generic fixed-step trainer; it contains no DANN or hidden early stopping."""

    def __init__(
        self,
        model: nn.Module,
        optimizer_config: OptimizerConfig,
        trainer_config: TrainerConfig,
    ) -> None:
        self.model = model
        self.optimizer_config = optimizer_config
        self.config = trainer_config
        self.optimizer = build_optimizer(model, optimizer_config)
        self.step = 0
        self.logs: list[StepLog] = []

    def _loss(self, prediction, target, mask):
        if self.config.output_mode == "raw_count":
            return raw_count_mae(prediction, target, mask)
        return log1p_target_mae(prediction, target, mask)

    def train_step(self, batch: dict[str, Any]) -> StepLog:
        if batch.get("target") is None:
            raise ValueError("Training batch has no labels")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(**batch["model_inputs"])
        loss = self._loss(output.prediction, batch["target"], batch["mask"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        gradients = [parameter.grad for parameter in self.model.parameters() if parameter.grad is not None]
        if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError("Missing or nonfinite model gradients")
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip_global_norm
        )
        self.optimizer.step()
        self.step += 1
        log = StepLog(
            step=self.step,
            loss=float(loss.detach()),
            valid_targets=int(batch["mask"].sum().item()),
            gradient_norm_before_clip=float(norm),
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
        )
        self.logs.append(log)
        return log

    @torch.no_grad()
    def evaluate_loss(self, batch: dict[str, Any]) -> float:
        if batch.get("target") is None:
            raise ValueError("Evaluation batch has no labels")
        self.model.eval()
        output = self.model(**batch["model_inputs"])
        return float(self._loss(output.prediction, batch["target"], batch["mask"]))

    def structured_log(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self.logs]
