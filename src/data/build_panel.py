"""Build the hourly per-station departure panel and its coverage mask.

Real interim files under `data.interim_dir` (see configs/default.yaml), already
filtered to our 4 cities -- the raw dataset has since been deleted, so
src/filter_cities.py can no longer be run; it stays in the repo only as
documentation of how these were produced:

    stations.parquet       id, city_id, bike_racks, lon, lat, ... (502 rows)
    station_status.parquet station_id, time, bikes, bikes_available_to_rent,
                            free_racks, maintenance (5,645,485 rows; no city_id)
    trips.parquet          city_id, time_start, station_id_start,
                            station_id_end, city (2,703,248 rows)

`load_interim_tables` renames these to the internal names the rest of this
module uses (station_id, timestamp, bikes_available, is_maintenance,
start_time, start_station_id) and handles three real-data gotchas:

  * station_status has no city_id -- it's derived by joining station_id
    against stations.
  * trips.station_id_start is stored as float while stations.id is integer;
    joining on mismatched dtypes silently returns zero matches instead of
    raising, so both are cast to the same nullable Int64 before any merge.
  * bikes_available_to_rent excludes bikes marked broken/reserved, so it
    understates real capacity -- the capacity fallback uses bikes (the raw
    physical count) instead.

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
  * compute_maintenance_flags -- checks the maintenance state as of the END
    of each hour, not its start or "any record during the hour": a station
    that flips into maintenance and back out within the hour is in service
    by the time the hour closes, so its departures during that hour are
    still valid. A city can also turn this filter off entirely via
    coverage_mask.trust_maintenance_flag in its config entry, for cases
    like Bilbao where the raw maintenance field looks unreliable (63.5% of
    its status rows read maintenance=True).
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
    # Cross-merge instead of MultiIndex.from_product: the latter coerces
    # station_ids through a plain Python list, which silently drops a
    # nullable Int64 dtype back to int64 and then breaks merge_asof's
    # dtype-matching against Int64 station/status tables.
    hours = pd.DataFrame({"hour": pd.date_range(start, end, freq="h", inclusive="left", tz="UTC")})
    stations = pd.DataFrame({"station_id": pd.Series(station_ids).drop_duplicates().sort_values().reset_index(drop=True)})
    return stations.merge(hours, how="cross")


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
    # A station-hour at the very start or end of the dataset has no status
    # observation on one side, so the asof lookup below returns NaT there --
    # that's expected, not an error, and must resolve to "not bracketed".
    #
    # Subtracting via .to_numpy() first turns tz-aware timestamps into an
    # object-dtype array of Timestamp/NaT; comparing that against a
    # numpy.timedelta64 then falls back to per-element Python comparisons,
    # which is exactly the path where NaT vs. a scalar can raise instead of
    # returning False. Keeping the subtraction and comparison as plain
    # pandas Series operations keeps everything in a proper timedelta64
    # dtype end to end, where NaT comparisons are always False by
    # construction -- fillna(False) below is a defensive backstop, not the
    # only thing standing between this and a crash.
    window = pd.Timedelta(hours=window_hours)
    hours = grid["hour"].reset_index(drop=True)
    prev_obs = _asof_lookup(grid, status, [], "backward")["timestamp"]
    next_obs = _asof_lookup(grid, status, [], "forward")["timestamp"]

    gap_before = hours - prev_obs
    gap_after = next_obs - hours
    bracketed = (gap_before <= window) & (gap_after <= window)
    return bracketed.fillna(False).to_numpy()


def compute_city_coverage_fraction(grid: pd.DataFrame) -> pd.Series:
    return grid.groupby("hour")["bracketed"].mean()


def compute_maintenance_flags(grid: pd.DataFrame, status: pd.DataFrame) -> np.ndarray:
    # Look up the state as of the END of each hour, not its start: a station
    # that enters and exits maintenance within the hour is back in service
    # by the time the hour closes, so departures recorded during that hour
    # are still valid. Only a station still under maintenance when the hour
    # closes should have the hour excluded.
    hour_end = grid[["station_id", "hour"]].assign(hour=grid["hour"] + pd.Timedelta(hours=1))
    maintenance = _asof_lookup(hour_end, status, ["is_maintenance"], "backward")["is_maintenance"]
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
    if len(city_trips) > 0 and grid["departures"].isna().all():
        raise ValueError(
            f"Merging {len(city_trips)} trips onto the station-hour grid for city_id={city_id} "
            "matched 0 rows -- check station_id dtypes/values"
        )
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
    stations = pd.read_parquet(interim_dir / "stations.parquet").rename(columns={"id": "station_id"})
    status = pd.read_parquet(interim_dir / "station_status.parquet").rename(
        columns={"time": "timestamp", "bikes": "bikes_available", "maintenance": "is_maintenance"}
    )
    trips = pd.read_parquet(interim_dir / "trips.parquet").rename(
        columns={"time_start": "start_time", "station_id_start": "start_station_id"}
    )

    # station_id dtypes must match exactly before any join: trips stores
    # start_station_id as float, stations.station_id as integer, and a
    # dtype mismatch makes pandas merges return zero matches silently.
    stations["station_id"] = stations["station_id"].astype("Int64")
    status["station_id"] = status["station_id"].astype("Int64")
    trips["start_station_id"] = trips["start_station_id"].astype("Int64")

    # station_status carries no city_id of its own; derive it from stations.
    station_city = stations.set_index("station_id")["city_id"]
    status["city_id"] = status["station_id"].map(station_city)
    if len(status) > 0 and status["city_id"].notna().sum() == 0:
        raise ValueError(
            "Deriving city_id for station_status matched 0 of "
            f"{len(status)} rows against stations.station_id -- check for a dtype or key mismatch"
        )

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
        # A city can opt out of the maintenance filter entirely (see
        # coverage_mask.trust_maintenance_flag in the city config) if its
        # maintenance field turns out to be unreliable.
        exclude_maintenance = mask_params["exclude_maintenance"] and city.get("trust_maintenance_flag", True)
        panel = build_station_panel(
            city["id"],
            trips,
            stations,
            status,
            start,
            end,
            bracket_window_hours=mask_params["bracket_window_hours"],
            min_city_coverage_fraction=mask_params["min_city_coverage_fraction"],
            exclude_maintenance=exclude_maintenance,
        )
        panel.to_parquet(out_dir / f"panel_{city['name'].lower()}.parquet", index=False)
        summaries.append(summarize_panel(panel, city["name"]))

    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
