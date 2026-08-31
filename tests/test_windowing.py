import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.windowing import (
    apply_budget_cutoff,
    apply_scalers,
    assign_split,
    build_supervised_samples,
    fit_scalers,
    inverse_transform_target,
)

CITY = 1


def make_panel(n_hours, start="2023-01-01", station_ids=(1,), departures=None, eligible=True, capacity=10.0):
    hours = pd.date_range(start, periods=n_hours, freq="h", tz="UTC")
    rows = []
    for sid in station_ids:
        rows.append(
            pd.DataFrame(
                {
                    "station_id": sid,
                    "hour": hours,
                    "departures": departures if departures is not None else (np.arange(n_hours) % 5),
                    "eligible": eligible,
                    "capacity": capacity,
                    "city_id": CITY,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def make_stations(station_ids, lat=43.0, lon=-2.9):
    return pd.DataFrame({"station_id": list(station_ids), "city_id": CITY, "lat": lat, "lon": lon})


N = 24 * 10  # 10 days, comfortably more than the 168h lag requirement


def test_window_requires_all_24_input_hours_eligible():
    # Hour 200 becomes ineligible. It sits inside the 24h window of every
    # input row t in [200, 223] (t-23..t includes 200), so all of them must
    # be dropped -- while t=180 (in the valid t>=167 range, but neither
    # touching the window nor needing hour 200 as its lag-167) is untouched.
    hours = pd.date_range("2023-01-01", periods=N, freq="h", tz="UTC")
    eligible = np.ones(N, dtype=bool)
    eligible[200] = False
    panel = make_panel(N, eligible=eligible)
    samples, *_ = build_supervised_samples(panel, make_stations([1]))
    present_hours = set(samples["hour"])
    for t in range(200, 224):
        assert hours[t] not in present_hours
    assert hours[180] in present_hours


def test_target_hour_must_be_eligible():
    # Hour 201 becomes ineligible. Input row t=200 targets hour 201, and
    # t=200's own 24h window (177..200) never touches 201 -- so t=200's
    # exclusion here is specifically the target-eligibility check, not the
    # window check (which is exercised separately above).
    hours = pd.date_range("2023-01-01", periods=N, freq="h", tz="UTC")
    eligible = np.ones(N, dtype=bool)
    eligible[201] = False
    panel = make_panel(N, eligible=eligible)
    samples, *_ = build_supervised_samples(panel, make_stations([1]))
    assert hours[200] not in set(samples["hour"])


def test_previous_week_lag_hour_must_be_eligible():
    # Hour 10 becomes ineligible. Input row t=177 needs hour 10 (=177-167)
    # as its previous-week lag; neither t=177's own 24h window (154..177)
    # nor its target (178) touch hour 10, isolating the lag-eligibility check.
    hours = pd.date_range("2023-01-01", periods=N, freq="h", tz="UTC")
    eligible = np.ones(N, dtype=bool)
    eligible[10] = False
    panel = make_panel(N, eligible=eligible)
    samples, *_ = build_supervised_samples(panel, make_stations([1]))
    assert hours[177] not in set(samples["hour"])


def test_covariate_values_match_config_semantics():
    panel = make_panel(N)
    samples, sequence_columns, feature_columns = build_supervised_samples(panel, make_stations([1]))
    assert feature_columns == [
        "previous_week_lag",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "is_weekend",
        "station_lat_norm",
        "station_lon_norm",
        "rack_capacity",
        "citywide_demand_recent",
    ]
    assert len(sequence_columns) == 24
    # lag_0 (most recent, last in the chronological sequence) equals
    # departures at t; sequence_columns is ordered oldest -> newest.
    row = samples.iloc[0]
    assert row[sequence_columns[-1]] == row["lag_0"]
    assert row[sequence_columns[0]] == row["lag_23"]


def test_split_drops_samples_whose_window_crosses_a_split_boundary():
    # Validation must be longer than the 168h lag window, or *every*
    # validation target would cross the boundary and the split would be
    # empty by construction -- that's not what this test is checking.
    hours = pd.date_range("2023-01-01", periods=24 * 60, freq="h", tz="UTC")
    panel = make_panel(len(hours))
    samples, *_ = build_supervised_samples(panel, make_stations([1]))

    split_config = {
        "train": {"start": "2023-01-01", "end": "2023-01-15"},
        "validation": {"start": "2023-01-15", "end": "2023-02-05"},
        "test": {"start": "2023-02-05", "end": "2023-02-19"},
    }
    labels = assign_split(samples, split_config)

    # A target right at validation's start needs lookback reaching back
    # into the training period -- its window crosses the boundary, so it
    # must belong to neither split.
    boundary_target = pd.Timestamp("2023-01-15 02:00", tz="UTC")
    boundary_rows = samples[samples["target_hour"] == boundary_target]
    assert len(boundary_rows) == 1
    assert pd.isna(labels.loc[boundary_rows.index[0]])

    # A target 10 days into validation, whose full week-plus lookback also
    # sits inside validation, must be labeled validation.
    safe_target = pd.Timestamp("2023-01-25 12:00", tz="UTC")
    safe_rows = samples[samples["target_hour"] == safe_target]
    assert len(safe_rows) == 1
    assert labels.loc[safe_rows.index[0]] == "validation"


def test_budget_cutoff_takes_the_last_n_days_not_the_first():
    hours = pd.date_range("2023-01-01", periods=24 * 20, freq="h", tz="UTC")
    train_samples = pd.DataFrame({"target_hour": hours})
    train_end = hours[-1] + pd.Timedelta(hours=1)  # end is exclusive, one hour past the last sample

    cropped = apply_budget_cutoff(train_samples, 7, train_end)
    assert cropped["target_hour"].min() == train_end - pd.Timedelta(days=7)
    assert cropped["target_hour"].max() == hours[-1]

    first_seven_days_cutoff = hours[0] + pd.Timedelta(days=7)
    assert (cropped["target_hour"] >= first_seven_days_cutoff).all(), (
        "budget cutoff must keep hours near train_end, not near the start of the train split"
    )

    full = apply_budget_cutoff(train_samples, "full", train_end)
    assert len(full) == len(train_samples)


def test_scaler_uses_only_train_split_no_leakage_from_validation_or_test():
    hours = pd.date_range("2023-01-01", periods=24 * 30, freq="h", tz="UTC")
    # train demand centered around 10, validation/test demand around 1000 --
    # if either leaked into the scaler, the fitted mean would be far higher.
    departures = np.where(hours < pd.Timestamp("2023-01-15", tz="UTC"), 10, 1000)
    panel = make_panel(len(hours), departures=departures)
    samples, sequence_columns, _ = build_supervised_samples(panel, make_stations([1]))

    split_config = {
        "train": {"start": "2023-01-01", "end": "2023-01-15"},
        "validation": {"start": "2023-01-15", "end": "2023-01-22"},
        "test": {"start": "2023-01-22", "end": "2023-01-31"},
    }
    samples["split"] = assign_split(samples, split_config)
    train_samples = samples[samples["split"] == "train"]
    assert len(train_samples) > 0
    assert (train_samples["target"] < 100).all()  # sanity: this really is the low-demand period

    scalers = fit_scalers(train_samples, sequence_columns)
    assert scalers["demand"]["mean"] < 100, "scaler mean leaked validation/test's much higher demand"


def test_inverse_transform_recovers_raw_target_exactly():
    panel = make_panel(N)
    samples, sequence_columns, _ = build_supervised_samples(panel, make_stations([1]))
    scalers = fit_scalers(samples, sequence_columns)
    scaled = apply_scalers(samples, scalers, sequence_columns)

    recovered = inverse_transform_target(scaled["target"], scalers)
    np.testing.assert_allclose(recovered, scaled["target_raw"].to_numpy(), rtol=1e-9, atol=1e-9)
