import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.metrics import mae, regression_metrics, rmse, wape


def test_mae_basic():
    assert mae([1, 2, 3], [1, 2, 3]) == 0.0
    assert mae([0, 0, 0], [1, 2, 3]) == 2.0


def test_rmse_basic():
    assert rmse([0, 0], [3, 4]) == pytest.approx(5.0 / np.sqrt(2))


def test_wape_basic():
    y_true = [10, 20, 30]
    y_pred = [12, 18, 33]
    expected = (2 + 2 + 3) / (10 + 20 + 30)
    assert wape(y_true, y_pred) == expected


def test_wape_returns_nan_not_inf_or_crash_when_true_values_sum_to_zero():
    # A station-hour slice with genuinely no demand at all: sum(|y_true|)==0.
    y_true = [0, 0, 0]
    y_pred = [0, 1, 2]
    result = wape(y_true, y_pred)
    assert np.isnan(result)


def test_wape_zero_denominator_does_not_raise_even_with_nonzero_predictions():
    # Same as above but phrased as a smoke test: this must not raise
    # ZeroDivisionError or produce inf.
    result = wape(np.zeros(50), np.random.default_rng(0).uniform(0, 5, 50))
    assert np.isnan(result)
    assert not np.isinf(result)


def test_regression_metrics_bundles_all_three():
    out = regression_metrics([1, 2, 3], [1, 2, 4])
    assert set(out.keys()) == {"mae", "rmse", "wape"}
    assert out["mae"] == mae([1, 2, 3], [1, 2, 4])
