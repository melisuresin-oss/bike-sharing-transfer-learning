"""Final-only eight-source GRL model; Stage-4 development code is untouched."""
from __future__ import annotations

import torch
from torch import nn
from research.models.graph_gru import GraphGRU
from research.models.common import NeuralModelConfig
from research.training.losses import log1p_target_mae
from . import core

class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, coefficient):
        ctx.coefficient = float(coefficient)
        return value.view_as(value)
    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None

def pool_current_mask(hidden, mask):
    if hidden.ndim != 3 or mask.dtype != torch.bool or mask.shape != hidden.shape[:2]:
        raise ValueError("current station mask must be bool [B,N]")
    denominator = mask.sum(1)
    if (denominator == 0).any():
        raise ValueError("empty current target mask")
    return (hidden * mask.unsqueeze(-1)).sum(1) / denominator.unsqueeze(-1)

class FinalSourceInvariantGraphGRU(nn.Module):
    domain_map = dict(core.DOMAIN_MAP)
    def __init__(self):
        super().__init__()
        config = NeuralModelConfig(model_type="graph_gru", hidden_size=32, dropout=0.0,
                                   output_mode="log1p_target")
        self.forecast = GraphGRU(config)
        self.discriminator = nn.Sequential(nn.Linear(32, 64), nn.ReLU(),
                                           nn.Dropout(0.10), nn.Linear(64, 8))

    def objectives(self, model_inputs, target, mask, domain_class):
        if not isinstance(domain_class, int) or not 0 <= domain_class < 8:
            raise ValueError("source domain class must be in [0,8)")
        output = self.forecast(**model_inputs)
        pooled = pool_current_mask(output.hidden, mask)
        logits = self.discriminator(ReverseGradient.apply(pooled, 0.5))
        labels = torch.full((len(pooled),), domain_class, dtype=torch.long,
                            device=pooled.device)
        forecast = log1p_target_mae(output.prediction, target, mask)
        domain = nn.functional.cross_entropy(logits, labels)
        return forecast, domain

def source_step(model, optimizer, batch, domain_class):
    model.train(); optimizer.zero_grad(set_to_none=True)
    forecast, domain = model.objectives(batch["model_inputs"], batch["target"],
                                        batch["mask"], domain_class)
    loss = forecast + domain
    if not torch.isfinite(loss): raise FloatingPointError("nonfinite GRL objective")
    loss.backward()
    gradients = [p.grad for p in model.parameters()]
    if any(g is None or not torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError("missing or nonfinite GRL gradient")
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {"forecast_loss": float(forecast.detach()), "domain_ce": float(domain.detach()),
            "lambda": 0.5, "gradient_norm_before_clip": float(norm)}

