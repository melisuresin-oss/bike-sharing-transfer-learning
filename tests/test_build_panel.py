import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build_panel import (
    build_station_panel,
    compute_bracket_flags,
    resolve_station_capacity,
    summarize_panel,
)

CITY = 1
START = "2023-01-01"
END = "2023-01-03"  # two days = 48 hourly rows per station


def ts(hour_str):
    return pd.Timestamp(hour_str, tz="UTC")


def make_stations(station_ids, bike_racks):
    return pd.DataFrame({"station_id": station_ids, "city_id": CITY, "bike_racks": bike_racks})


def make_status(rows):
    df = pd.DataFrame(rows, columns=["station_id", "timestamp", "bikes_available", "free_racks", "is_maintenance"])
    df["city_id"] = CITY
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def make_trips(rows):
    df = pd.DataFrame(rows, columns=["start_station_id", "start_time"])
    df["city_id"] = CITY
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    return df


def test_quiet_night_hours_are_bracketed_not_eliminated():
    # Station 1 only reports at 06:00 and 09:00 -- nothing changed in between,
    # so status has no rows for 07:00/08:00. Those hours must still be
    # bracketed (both observations are 3h away, well inside the 24h window).
    grid = pd.DataFrame(
        {"station_id": [1, 1, 1, 1], "hour": [ts("2023-01-01 05:00"), ts("2023-01-01 07:00"), ts("2023-01-01 08:00"), ts("2023-01-01 10:00")]}
    )
    status = make_status(
        [
            [1, "2023-01-01 06:00", 5, 5, False],
            [1, "2023-01-01 09:00", 5, 5, False],
        ]
    )
    bracketed = compute_bracket_flags(grid, status, window_hours=24)
    assert bracketed[1]  # 07:00, between the two observations
    assert bracketed[2]  # 08:00, between the two observations
    assert not bracketed[0]  # 05:00 is before the first observation -> not bracketed
    assert not bracketed[3]  # 10:00 is after the last observation -> not bracketed


def test_naive_within_hour_rule_would_have_failed_this_case():
    # Sanity check that the scenario above is a real regression case: no
    # status row falls inside 07:00 or 08:00 themselves.
    status = make_status([[1, "2023-01-01 06:00", 5, 5, False], [1, "2023-01-01 09:00", 5, 5, False]])
    in_07 = status["timestamp"].between(ts("2023-01-01 07:00"), ts("2023-01-01 08:00"), inclusive="left")
    in_08 = status["timestamp"].between(ts("2023-01-01 08:00"), ts("2023-01-01 09:00"), inclusive="left")
    assert not in_07.any()
    assert not in_08.any()


def test_city_coverage_survives_a_quiet_hour_but_catches_an_outage():
    stations = make_stations([1, 2], [10, 10])
    # Both stations report every ~3h until 06:00, then a 60h feed outage
    # before the next reading -- far longer than the 24h bracket window on
    # either side, so no hour in the middle of it can be bracketed from
    # either direction.
    status = make_status(
        [
            [1, "2023-01-01 00:00", 5, 5, False],
            [1, "2023-01-01 03:00", 5, 5, False],
            [1, "2023-01-01 06:00", 5, 5, False],
            [1, "2023-01-03 18:00", 5, 5, False],
            [2, "2023-01-01 00:00", 5, 5, False],
            [2, "2023-01-01 03:00", 5, 5, False],
            [2, "2023-01-01 06:00", 5, 5, False],
            [2, "2023-01-03 18:00", 5, 5, False],
        ]
    )
    trips = make_trips([])
    panel = build_station_panel(CITY, trips, stations, status, START, END)

    # 04:00 sits between the 03:00 and 06:00 readings -- a quiet hour, not an
    # outage -- so it must still be bracketed.
    quiet_hour = panel[panel["hour"] == ts("2023-01-01 04:00")]
    assert (quiet_hour["bracketed"]).all()
    assert (quiet_hour["city_coverage_fraction"] == 1.0).all()
    assert quiet_hour["eligible"].all()

    # 2023-01-02 12:00 is the midpoint of the 60h outage: 30h from the last
    # reading on both sides, so neither direction is within the 24h window.
    outage_hour = panel[panel["hour"] == ts("2023-01-02 12:00")]
    assert not outage_hour["bracketed"].any()
    assert (outage_hour["city_coverage_fraction"] == 0.0).all()
    assert not outage_hour["eligible"].any()


def test_capacity_falls_back_to_observed_bikes_plus_free_racks():
    stations = make_stations([1, 2], [np.nan, 0])
    status = make_status(
        [
            [1, "2023-01-01 00:00", 12, 8, False],
            [1, "2023-01-01 06:00", 15, 5, False],  # max observed = 20
            [2, "2023-01-01 00:00", 3, 17, False],  # max observed = 20
        ]
    )
    capacity = resolve_station_capacity(stations, status)
    assert capacity.loc[1] == 20
    assert capacity.loc[2] == 20


def test_maintenance_hours_are_excluded_when_configured():
    stations = make_stations([1], [10])
    status = make_status(
        [
            [1, "2023-01-01 00:00", 5, 5, False],
            [1, "2023-01-01 05:00", 0, 0, True],
            [1, "2023-01-01 08:00", 5, 5, False],
        ]
    )
    trips = make_trips([])
    panel = build_station_panel(CITY, trips, stations, status, START, END, exclude_maintenance=True)
    under_maintenance = panel[(panel["hour"] >= ts("2023-01-01 05:00")) & (panel["hour"] < ts("2023-01-01 08:00"))]
    assert under_maintenance["is_maintenance"].all()
    assert not under_maintenance["eligible"].any()


def test_summary_counts_positive_and_zero_hours_among_eligible_only():
    stations = make_stations([1], [10])
    status = make_status([[1, h, 5, 5, False] for h in pd.date_range(START, END, freq="3h", tz="UTC")])
    trips = make_trips([[1, "2023-01-01 07:30"], [1, "2023-01-01 07:45"]])
    panel = build_station_panel(CITY, trips, stations, status, START, END)
    summary = summarize_panel(panel, "TestCity")

    assert summary["station_hours_total"] == 48
    assert summary["positive_demand_hours"] == 1  # both trips fall in the 07:00 hour
    assert summary["verified_zero_hours"] == summary["station_hours_eligible"] - 1
    assert summary["median_capacity"] == 10
