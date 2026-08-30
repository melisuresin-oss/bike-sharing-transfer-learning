"""Build the hourly per-station departure panel and its coverage mask.

Expected interim parquet files under `data.interim_dir` (see configs/default.yaml):
    trips_filtered.parquet    city_id, start_station_id, start_time (UTC)
    stations_filtered.parquet station_id, city_id, latitude, longitude, bike_racks
    status_filtered.parquet   station_id, city_id, timestamp (UTC), bikes_available,
                              free_racks, is_maintenance

`status_filtered.parquet` is not produced by src/filter_cities.py yet -- that
script only filters cities/trips/stations. It needs the same city-based
filter applied to the raw station-status export before this script can run
against real data.

status is change-triggered: a station only gets a new row when something
about it changes, so most hours have no row for a given station even while
the feed is healthy. The three helpers below turn that into a coverage mask:

  * compute_bracket_flags -- a station-hour is trustworthy if it sits within
    `bracket_window_hours` of a status observation on both sides, not if a
    row exists *inside* the hour. Requiring a row inside the hour would
    discard quiet night hours where nothing changed.
  * compute_city_coverage_fraction -- reuses those per-station bracket flags
    to ask "was the city's feed alive around this hour", rather than "did
    the city log a change during this hour" (which fails for the same
    quiet-hour reason). A real feed outage still shows up here because it
    drags every active station's bracket flag down for the hours around it.
  * resolve_station_capacity -- falls back to the observed max(bikes +
    free_racks) per station when metadata capacity is missing or zero
    (this happens for Freiburg's bike_racks column in the raw export).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def resolve_station_capacity(stations: pd.DataFrame, status: pd.DataFrame) -> pd.Series:
    capacity = stations.set_index("station_id")["bike_racks"].astype(float)
    observed_capacity = (
        (status["bikes_available"].fillna(0) + status["free_racks"].fillna(0))
        .groupby(status["station_id"])
        .max()
    )
    missing = capacity.isna() | (capacity <= 0)
    capacity[missing] = capacity[missing].index.map(observed_capacity)
    return capacity


def build_hourly_grid(station_ids, start, end) -> pd.DataFrame:
    hours = pd.date_range(start, end, freq="h", inclusive="left", tz="UTC")
    return pd.MultiIndex.from_product(
        [sorted(station_ids), hours], names=["station_id", "hour"]
    ).to_frame(index=False)


def compute_departures(trips: pd.DataFrame) -> pd.DataFrame:
    hour = trips["start_time"].dt.floor("h")
    return (
        trips.assign(hour=hour)
        .groupby(["start_station_id", "hour"])
        .size()
        .rename("departures")
        .reset_index()
        .rename(columns={"start_station_id": "station_id"})
    )


def _asof_lookup(grid: pd.DataFrame, status: pd.DataFrame, columns, direction) -> pd.DataFrame:
    status_sorted = status.sort_values("timestamp")
    lookup = grid[["station_id", "hour"]].reset_index(drop=True)
    lookup_sorted = lookup.sort_values("hour")
    merged = pd.merge_asof(
        lookup_sorted,
        status_sorted[["station_id", "timestamp", *columns]],
        left_on="hour",
        right_on="timestamp",
        by="station_id",
        direction=direction,
    )
    return merged.set_index(lookup_sorted.index).reindex(lookup.index)


def compute_bracket_flags(grid: pd.DataFrame, status: pd.DataFrame, window_hours: float) -> np.ndarray:
    window = pd.Timedelta(hours=window_hours).to_timedelta64()
    hours = grid["hour"].to_numpy()
    prev_obs = _asof_lookup(grid, status, [], "backward")["timestamp"].to_numpy()
    next_obs = _asof_lookup(grid, status, [], "forward")["timestamp"].to_numpy()
    gap_before = hours - prev_obs
    gap_after = next_obs - hours
    return (gap_before <= window) & (gap_after <= window)


def compute_city_coverage_fraction(grid: pd.DataFrame) -> pd.Series:
    return grid.groupby("hour")["bracketed"].mean()


def compute_maintenance_flags(grid: pd.DataFrame, status: pd.DataFrame) -> np.ndarray:
    maintenance = _asof_lookup(grid, status, ["is_maintenance"], "backward")["is_maintenance"]
    return maintenance.fillna(False).to_numpy().astype(bool)


def build_station_panel(
    city_id,
    trips: pd.DataFrame,
    stations: pd.DataFrame,
    status: pd.DataFrame,
    start,
    end,
    bracket_window_hours: float = 24,
    min_city_coverage_fraction: float = 0.5,
    exclude_maintenance: bool = True,
) -> pd.DataFrame:
    city_trips = trips[trips["city_id"] == city_id]
    city_stations = stations[stations["city_id"] == city_id]
    city_status = status[status["city_id"] == city_id]

    active_station_ids = city_stations["station_id"].unique()
    if len(active_station_ids) == 0:
        raise ValueError(f"No stations found for city_id={city_id}")

    capacity = resolve_station_capacity(city_stations, city_status)

    grid = build_hourly_grid(active_station_ids, start, end)
    grid = grid.merge(compute_departures(city_trips), on=["station_id", "hour"], how="left")
    grid["departures"] = grid["departures"].fillna(0).astype(int)

    grid["bracketed"] = compute_bracket_flags(grid, city_status, bracket_window_hours)
    coverage = compute_city_coverage_fraction(grid)
    grid["city_coverage_fraction"] = grid["hour"].map(coverage)
    grid["is_maintenance"] = compute_maintenance_flags(grid, city_status)

    grid["eligible"] = grid["bracketed"] & (grid["city_coverage_fraction"] >= min_city_coverage_fraction)
    if exclude_maintenance:
        grid["eligible"] &= ~grid["is_maintenance"]

    grid["capacity"] = grid["station_id"].map(capacity)
    grid["city_id"] = city_id
    return grid


def summarize_panel(panel: pd.DataFrame, city_name: str) -> dict:
    total = len(panel)
    eligible = panel["eligible"]
    n_eligible = int(eligible.sum())
    return {
        "city": city_name,
        "station_hours_total": total,
        "station_hours_eligible": n_eligible,
        "eligibility_pct": round(100 * n_eligible / total, 2) if total else float("nan"),
        "positive_demand_hours": int((eligible & (panel["departures"] > 0)).sum()),
        "verified_zero_hours": int((eligible & (panel["departures"] == 0)).sum()),
        "median_capacity": panel.loc[panel["capacity"].notna(), "capacity"].median(),
    }


def load_interim_tables(interim_dir: Path):
    interim_dir = Path(interim_dir)
    trips = pd.read_parquet(interim_dir / "trips_filtered.parquet")
    stations = pd.read_parquet(interim_dir / "stations_filtered.parquet")
    status = pd.read_parquet(interim_dir / "status_filtered.parquet")
    trips["start_time"] = pd.to_datetime(trips["start_time"], utc=True)
    status["timestamp"] = pd.to_datetime(status["timestamp"], utc=True)
    return trips, stations, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    trips, stations, status = load_interim_tables(config["data"]["interim_dir"])
    out_dir = Path(config["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_params = config["coverage_mask"]
    start, end = config["data"]["observation_start"], config["data"]["observation_end"]

    summaries = []
    for city in config["cities"]:
        panel = build_station_panel(
            city["id"],
            trips,
            stations,
            status,
            start,
            end,
            bracket_window_hours=mask_params["bracket_window_hours"],
            min_city_coverage_fraction=mask_params["min_city_coverage_fraction"],
            exclude_maintenance=mask_params["exclude_maintenance"],
        )
        panel.to_parquet(out_dir / f"panel_{city['name'].lower()}.parquet", index=False)
        summaries.append(summarize_panel(panel, city["name"]))

    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
