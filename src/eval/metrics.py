"""Regression metrics shared by baselines and models, always in original units.

Callers are expected to inverse-transform any scaled predictions/targets
before calling these -- these functions never scale or unscale anything
themselves.
"""
import numpy as np


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true, y_pred) -> float:
    """Weighted Absolute Percentage Error: sum(|y_true - y_pred|) / sum(|y_true|).

    Returns NaN (never raises, never inf) when the true values sum to zero --
    e.g. a station-hour slice with no demand at all -- since the ratio is
    undefined there, not zero or infinite.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true).sum()
    if denominator == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denominator)


def regression_metrics(y_true, y_pred) -> dict:
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "wape": wape(y_true, y_pred)}
