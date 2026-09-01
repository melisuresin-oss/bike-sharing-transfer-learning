import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build_panel import (
    build_station_panel,
    compute_bracket_flags,
    compute_maintenance_flags,
    load_interim_tables,
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


def test_bracket_flags_handle_missing_neighbor_at_dataset_edges():
    # A station-hour at the very start or end of the observation window has
    # no status observation before it (start) or after it (end) -- that's
    # unavoidable at the dataset's edges and must resolve to False, not
    # raise. This is the exact real-data scenario that crashed with a NaT
    # comparison in production.
    grid = pd.DataFrame(
        {
            "station_id": [1, 1],
            "hour": [ts("2023-01-01 00:00"), ts("2023-01-02 23:00")],
        }
    )
    status = make_status(
        [
            [1, "2023-01-01 02:00", 5, 5, False],  # first observation is *after* the first grid hour
            [1, "2023-01-02 20:00", 5, 5, False],  # last observation is *before* the last grid hour
        ]
    )
    bracketed = compute_bracket_flags(grid, status, window_hours=24)
    assert bracketed.dtype == bool
    assert not pd.isna(bracketed).any()
    assert not bracketed[0]  # dataset start: no observation before it
    assert not bracketed[1]  # dataset end: no observation after it


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


def test_maintenance_flag_looks_at_end_of_hour_not_any_record_within_it():
    # Station 1 blips into maintenance mid-hour and is back to normal well
    # before the hour closes -- its departures during that hour are still
    # valid. Station 2 goes into maintenance mid-hour and is still there
    # when the hour closes -- that hour should be excluded. Checking "any
    # record during the hour" would have flagged both.
    grid = pd.DataFrame({"station_id": [1, 2], "hour": [ts("2023-01-01 05:00"), ts("2023-01-01 05:00")]})
    status = make_status(
        [
            [1, "2023-01-01 04:00", 5, 5, False],
            [1, "2023-01-01 05:20", 0, 0, True],
            [1, "2023-01-01 05:40", 5, 5, False],  # back to normal before 06:00
            [2, "2023-01-01 04:00", 5, 5, False],
            [2, "2023-01-01 05:10", 0, 0, True],  # still under maintenance at 06:00
        ]
    )
    maintenance = compute_maintenance_flags(grid, status)
    assert not maintenance[0]  # station 1: blip within the hour, back to normal by hour's end
    assert maintenance[1]  # station 2: still under maintenance when the hour closes


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
    # Maintenance starts at 05:00 and ends exactly at 08:00. Hours 05:00 and
    # 06:00 are still under maintenance when they close (at 06:00/07:00) --
    # excluded. Hour 07:00 closes at 08:00, exactly when the station goes
    # back into service, so its departures are valid again.
    excluded_hours = panel[panel["hour"].isin([ts("2023-01-01 05:00"), ts("2023-01-01 06:00")])]
    assert excluded_hours["is_maintenance"].all()
    assert not excluded_hours["eligible"].any()

    back_in_service_hour = panel[panel["hour"] == ts("2023-01-01 07:00")]
    assert not back_in_service_hour["is_maintenance"].any()


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


def test_build_station_panel_raises_when_trip_join_matches_nothing():
    # All trips point at a station_id absent from the grid -- e.g. what a
    # silent dtype mismatch on the join key would look like. Real data
    # occasionally has a stray trip at an unknown station, which should just
    # be ignored; the failure mode this guards against is *every* trip
    # missing, which signals a broken join rather than one bad row.
    stations = make_stations([1], [10])
    status = make_status([[1, "2023-01-01 00:00", 5, 5, False]])
    trips = make_trips([[999, "2023-01-01 01:00"]])
    with pytest.raises(ValueError):
        build_station_panel(CITY, trips, stations, status, START, END)


def test_load_interim_tables_matches_real_schema_and_dtype_mismatch(tmp_path):
    # Mirrors the real files: stations.id (int) vs trips.station_id_start
    # (float), station_status with no city_id, and a bikes_available_to_rent
    # column that is lower than bikes and must not be used for capacity.
    stations = pd.DataFrame(
        {"id": [1, 2], "city_id": [CITY, CITY], "bike_racks": [15, 0], "lon": [1.0, 1.1], "lat": [2.0, 2.1]}
    )
    status = pd.DataFrame(
        {
            "station_id": [1, 1, 2],
            "time": pd.to_datetime(["2023-01-01 00:00", "2023-01-01 03:00", "2023-01-01 00:00"], utc=True),
            "bikes": [5, 6, 3],
            "bikes_available_to_rent": [4, 5, 2],
            "free_racks": [10, 9, 12],
            "maintenance": [False, False, False],
        }
    )
    trips = pd.DataFrame(
        {
            "city_id": [CITY, CITY],
            "time_start": pd.to_datetime(["2023-01-01 01:00", "2023-01-01 01:30"], utc=True),
            "station_id_start": [1.0, 1.0],
            "station_id_end": [2.0, 2.0],
            "city": ["TestSystem", "TestSystem"],
        }
    )
    stations.to_parquet(tmp_path / "stations.parquet")
    status.to_parquet(tmp_path / "station_status.parquet")
    trips.to_parquet(tmp_path / "trips.parquet")

    loaded_trips, loaded_stations, loaded_status = load_interim_tables(tmp_path)

    assert loaded_trips["start_station_id"].dtype == loaded_stations["station_id"].dtype
    assert (loaded_status["city_id"] == CITY).all()

    panel = build_station_panel(CITY, loaded_trips, loaded_stations, loaded_status, "2023-01-01", "2023-01-02")
    assert panel["departures"].sum() == 2  # float vs int station_id must still join correctly

    # station 2's bike_racks is 0 -> capacity falls back to bikes + free_racks
    # (15), not bikes_available_to_rent + free_racks (14).
    assert panel.loc[panel["station_id"] == 2, "capacity"].iloc[0] == 15


def test_load_interim_tables_raises_when_status_city_id_cannot_be_derived(tmp_path):
    stations = pd.DataFrame({"id": [1], "city_id": [CITY], "bike_racks": [10], "lon": [0.0], "lat": [0.0]})
    status = pd.DataFrame(
        {
            "station_id": [999],  # absent from stations -- e.g. a dtype mismatch
            "time": pd.to_datetime(["2023-01-01 00:00"], utc=True),
            "bikes": [5],
            "bikes_available_to_rent": [4],
            "free_racks": [5],
            "maintenance": [False],
        }
    )
    trips = pd.DataFrame(
        {
            "city_id": [CITY],
            "time_start": pd.to_datetime(["2023-01-01 00:00"], utc=True),
            "station_id_start": [1.0],
            "station_id_end": [1.0],
            "city": ["TestSystem"],
        }
    )
    stations.to_parquet(tmp_path / "stations.parquet")
    status.to_parquet(tmp_path / "station_status.parquet")
    trips.to_parquet(tmp_path / "trips.parquet")

    with pytest.raises(ValueError):
        load_interim_tables(tmp_path)
