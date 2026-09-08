from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Hashable, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    n: int
    mae: float | None
    rmse: float | None
    wape: float | None
    station_macro_mae: float | None
    stations_in_macro: int

    def as_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def compute_metrics(
    target: Sequence[float],
    prediction: Sequence[float],
    station_ids: Sequence[int],
    *,
    station_min_hours: int = 24,
) -> MetricResult:
    y = np.asarray(target, dtype=np.float64)
    yhat = np.asarray(prediction, dtype=np.float64)
    stations = np.asarray(station_ids, dtype=np.int64)
    if not (len(y) == len(yhat) == len(stations)):
        raise ValueError("target, prediction, and station_ids must have equal length")
    valid = np.isfinite(y) & np.isfinite(yhat)
    if not valid.any():
        return MetricResult(0, None, None, None, None, 0)
    y = y[valid]
    yhat = yhat[valid]
    stations = stations[valid]
    error = np.abs(y - yhat)
    mae = float(np.mean(error))
    rmse = float(np.sqrt(np.mean(np.square(y - yhat))))
    denominator = float(np.sum(y))
    wape = None if denominator == 0.0 else float(np.sum(error) / denominator)
    station_maes = []
    for station_id in np.unique(stations):
        station_error = error[stations == station_id]
        if len(station_error) >= station_min_hours:
            station_maes.append(float(np.mean(station_error)))
    station_macro = float(np.mean(station_maes)) if station_maes else None
    return MetricResult(
        n=len(y),
        mae=mae,
        rmse=rmse,
        wape=wape,
        station_macro_mae=station_macro,
        stations_in_macro=len(station_maes),
    )


def matched_keys(
    left_keys: Iterable[Hashable], right_keys: Iterable[Hashable]
) -> list[Hashable]:
    left = list(left_keys)
    right = list(right_keys)
    if len(left) != len(set(left)) or len(right) != len(set(right)):
        raise ValueError("Paired comparison inputs contain duplicate keys")
    return sorted(set(left).intersection(right), key=str)


def equal_city_macro(city_metrics: Sequence[MetricResult], field: str = "mae") -> float | None:
    values = [getattr(item, field) for item in city_metrics]
    available = [float(value) for value in values if value is not None]
    return float(np.mean(available)) if available else None
