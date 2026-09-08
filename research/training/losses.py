from __future__ import annotations

import torch
from torch import Tensor


class ZeroValidTargetsError(ValueError):
    pass


def masked_mae(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("Prediction, target, and mask shapes must match")
    valid = mask.to(dtype=torch.bool) & torch.isfinite(target)
    denominator = valid.sum()
    if int(denominator.item()) == 0:
        raise ZeroValidTargetsError("Masked MAE has zero valid targets")
    errors = torch.where(valid, torch.abs(prediction - target), torch.zeros_like(prediction))
    return errors.sum() / denominator.to(dtype=prediction.dtype)


def raw_count_mae(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    return masked_mae(prediction, target, mask)


def log1p_target_mae(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    valid = mask.to(dtype=torch.bool) & torch.isfinite(target)
    transformed = torch.where(valid, torch.log1p(target.clamp_min(0.0)), torch.zeros_like(target))
    return masked_mae(prediction, transformed, valid)


def city_balanced_masked_mae(
    prediction: Tensor, target: Tensor, mask: Tensor, city_ids: Tensor
) -> Tensor:
    """Average station-hour MAE within city, then equally across valid cities."""
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("Prediction, target, and mask shapes must match")
    if city_ids.ndim != 1 or city_ids.shape[0] != prediction.shape[0]:
        raise ValueError("city_ids must identify the leading batch dimension")
    city_losses = []
    for city_id in torch.unique(city_ids, sorted=True):
        rows = city_ids == city_id
        try:
            city_losses.append(masked_mae(prediction[rows], target[rows], mask[rows]))
        except ZeroValidTargetsError:
            continue
    if not city_losses:
        raise ZeroValidTargetsError("No city has a valid target")
    return torch.stack(city_losses).mean()
