import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.baselines import (
    fit_historical_average,
    hour_of_week,
    predict_historical_average,
    predict_persistence,
)


def ts(s):
    return pd.Timestamp(s, tz="UTC")


def test_hour_of_week_is_weekday_times_24_plus_hour():
    # 2023-01-02 is a Monday (weekday 0).
    assert hour_of_week(pd.Series([ts("2023-01-02 05:00")])).iloc[0] == 5
    # 2023-01-03 is a Tuesday (weekday 1) -> 24 + 5 = 29.
    assert hour_of_week(pd.Series([ts("2023-01-03 05:00")])).iloc[0] == 29


def test_historical_average_predicts_the_station_hour_of_week_mean():
    train = pd.DataFrame(
        {
            "station_id": [1, 1, 1],
            "target_hour": [ts("2023-01-02 05:00"), ts("2023-01-09 05:00"), ts("2023-01-16 05:00")],  # all Mondays 05:00
            "target_raw": [10.0, 20.0, 30.0],
        }
    )
    model = fit_historical_average(train)
    eval_df = pd.DataFrame({"station_id": [1], "target_hour": [ts("2023-01-23 05:00")]})  # another Monday 05:00
    pred = predict_historical_average(model, eval_df)
    assert pred[0] == pytest.approx(20.0)


def test_historical_average_falls_back_to_station_mean_for_unseen_hour_of_week():
    train = pd.DataFrame(
        {
            "station_id": [1, 1],
            "target_hour": [ts("2023-01-02 05:00"), ts("2023-01-09 05:00")],  # only Monday 05:00 ever seen
            "target_raw": [10.0, 30.0],
        }
    )
    model = fit_historical_average(train)
    # Same station, but a Tuesday 09:00 -- never seen for this station.
    eval_df = pd.DataFrame({"station_id": [1], "target_hour": [ts("2023-01-03 09:00")]})
    pred = predict_historical_average(model, eval_df)
    assert pred[0] == pytest.approx(20.0)  # station-level mean of 10 and 30


def test_historical_average_falls_back_to_global_mean_for_unseen_station():
    train = pd.DataFrame(
        {
            "station_id": [1, 2],
            "target_hour": [ts("2023-01-02 05:00"), ts("2023-01-02 05:00")],
            "target_raw": [10.0, 30.0],
        }
    )
    model = fit_historical_average(train)
    eval_df = pd.DataFrame({"station_id": [999], "target_hour": [ts("2023-01-02 05:00")]})
    pred = predict_historical_average(model, eval_df)
    assert pred[0] == pytest.approx(20.0)  # global mean of 10 and 30


def test_persistence_recovers_raw_current_hour_value_through_the_scaler():
    scalers = {"demand": {"mean": 5.0, "std": 2.0}}
    raw_lag_0 = np.array([0.0, 5.0, 13.0])
    scaled_lag_0 = (raw_lag_0 - 5.0) / 2.0
    eval_df = pd.DataFrame({"lag_0": scaled_lag_0})
    pred = predict_persistence(eval_df, scalers)
    np.testing.assert_allclose(pred, raw_lag_0, rtol=1e-9, atol=1e-9)
