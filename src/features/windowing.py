"""Build supervised training samples from the coverage-masked panel.

Input: the per-city panel parquet files written by src/data/build_panel.py
(station_id, hour, departures, eligible, capacity, city_id) plus the
stations table (for static station lat/lon), from data.interim_dir.

Each sample is centered on an input hour t and predicts departures at t+1:

    sequence: departures at t-23, t-22, ..., t (24 values, oldest first --
              this is the order a GRU would consume them in, and is a
              module-level choice since the proposal/config only pin the
              *covariate* order, not the sequence order)
    covariates (10, in the exact order of config sample.covariates):
        previous_week_lag      departures at (t+1 - 168h), i.e. the same
                                clock hour one week before the target
        hour_sin, hour_cos     calendar encoding of t's local hour
        weekday_sin, weekday_cos, is_weekend   calendar encoding of t's weekday
        station_lat_norm, station_lon_norm     static, from stations.parquet
        rack_capacity           static, from the panel's capacity column
        citywide_demand_recent  sum of departures across all of the city's
                                 stations at hour t (not filtered by
                                 eligibility -- see build_supervised_samples)
    target: raw departures at t+1

Eligibility chain (all required, or the sample is dropped -- this is the
part most likely to silently produce leaked or garbage samples if any one
link is skipped):
    1. the target hour (t+1) must be eligible in the panel
    2. all 24 hours in the input window (t-23..t) must be eligible
    3. the previous-week-lag hour (t+1 - 168h) must be eligible

This only works because the panel is a *complete* hourly grid per station
(build_hourly_grid fills every hour, eligible or not) -- so a plain
groupby("station_id")[...].shift(k) already gives exactly the value k hours
before the current row, with no gaps to trip over.

Splitting is chronological per configs/default.yaml's split.*.{start,end}.
A sample belongs to a split only if its *entire* required range -- from the
previous-week-lag hour (target_hour - 168h) through the target hour -- sits
inside that split's [start, end). A sample whose window straddles a split
boundary (e.g. an early-validation target whose lookback reaches back into
the training period) is dropped rather than assigned to either split, to
avoid leaking information across splits.

Budget cutoffs (1/7/30 days, "full") subset the *training* split only, kept
from the END of the training period backward -- "last 7 days" means the 7
days immediately before the training split's end, not the first 7. All
budget variants and the validation/test splits are scaled using the same
per-city scaler, fit once on the full (uncropped) training split.

Scaling: per city, one shared mean/std for all raw departure-count
quantities (the 24-value sequence, previous_week_lag, citywide_demand_recent,
and the target itself, since they're literally the same physical quantity
measured at different times), and separate mean/std scalers for
station_lat_norm, station_lon_norm, and rack_capacity. All scalers are fit
from the training split only. The raw (unscaled) target is kept alongside
the scaled one so evaluation metrics can be computed in original units via
inverse_transform_target.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

WEEK_HOURS = 24 * 7


def _mean_std(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean = float(values.mean()) if len(values) else 0.0
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    # A degenerate (zero or undefined) std would divide samples by zero or
    # NaN downstream -- fall back to an identity-ish scale of 1 instead.
    if not np.isfinite(std) or std == 0.0:
        std = 1.0
    return mean, std


def sequence_column_names(demand_lookback_hours: int) -> list:
    """Chronological order (oldest first) for feeding a GRU: t-23, ..., t-1, t.

    Exposed separately (rather than inlined in build_supervised_samples) so
    other code -- src/training/train.py builds tensors from these same
    columns -- can name them without duplicating or drifting from this
    convention.
    """
    return [f"lag_{k}" for k in range(demand_lookback_hours - 1, -1, -1)]


def build_supervised_samples(
    panel: pd.DataFrame, stations: pd.DataFrame, demand_lookback_hours: int = 24
):
    panel = panel.sort_values(["station_id", "hour"]).reset_index(drop=True)
    grouped = panel.groupby("station_id", sort=False)

    samples = pd.DataFrame(index=panel.index)
    samples["station_id"] = panel["station_id"]
    samples["hour"] = panel["hour"]
    samples["target_hour"] = panel["hour"] + pd.Timedelta(hours=1)

    # lag_k = departures at t-k. window_eligible requires all of t..t-23.
    window_eligible = panel["eligible"].astype(bool).copy()
    for k in range(demand_lookback_hours):
        samples[f"lag_{k}"] = grouped["departures"].shift(k)
        if k > 0:
            window_eligible &= grouped["eligible"].shift(k).fillna(False).astype(bool)
    sequence_columns = sequence_column_names(demand_lookback_hours)

    samples["previous_week_lag"] = grouped["departures"].shift(WEEK_HOURS - 1)
    lag_week_eligible = grouped["eligible"].shift(WEEK_HOURS - 1).fillna(False).astype(bool)

    samples["target"] = grouped["departures"].shift(-1)
    target_hour_eligible = grouped["eligible"].shift(-1).fillna(False).astype(bool)

    hour_of_day = panel["hour"].dt.hour
    weekday = panel["hour"].dt.dayofweek
    samples["hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24)
    samples["hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24)
    samples["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    samples["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    samples["is_weekend"] = (weekday >= 5).astype(float)

    station_attrs = stations.set_index("station_id")
    samples["station_lat_norm"] = panel["station_id"].map(station_attrs["lat"]).astype(float)
    samples["station_lon_norm"] = panel["station_id"].map(station_attrs["lon"]).astype(float)
    samples["rack_capacity"] = panel["capacity"].astype(float)

    # Citywide total at hour t, summed over all of the city's stations
    # regardless of their own eligibility -- requiring every station in the
    # city to be simultaneously eligible would make this covariate almost
    # never available. This is a deliberate simplification: ineligible
    # station-hours contribute their filled-zero departure count.
    city_hourly_total = panel.groupby("hour")["departures"].sum()
    samples["citywide_demand_recent"] = panel["hour"].map(city_hourly_total).astype(float)

    eligible = window_eligible & lag_week_eligible & target_hour_eligible
    samples = samples.loc[eligible].reset_index(drop=True)

    feature_columns = [
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
    # rack_capacity can be NaN if a station never resolved a capacity value
    # (see resolve_station_capacity in build_panel.py) -- such samples can't
    # be fed to a model, so they're dropped here rather than left to corrupt
    # training downstream.
    samples = samples.dropna(subset=[*sequence_columns, *feature_columns, "target"]).reset_index(drop=True)

    return samples, sequence_columns, feature_columns


def assign_split(samples: pd.DataFrame, split_config: dict) -> pd.Series:
    earliest_required_hour = samples["target_hour"] - pd.Timedelta(hours=WEEK_HOURS)
    labels = pd.Series(pd.NA, index=samples.index, dtype=object)
    for name, bounds in split_config.items():
        start = pd.Timestamp(bounds["start"], tz="UTC")
        end = pd.Timestamp(bounds["end"], tz="UTC")
        # The whole required range -- lag-week hour through target hour --
        # must sit inside this split, or the sample is left unassigned
        # (dropped) rather than crossing into a neighboring split.
        in_split = (earliest_required_hour >= start) & (samples["target_hour"] < end)
        labels = labels.mask(in_split, name)
    return labels


def apply_budget_cutoff(train_samples: pd.DataFrame, budget_days, train_end: pd.Timestamp) -> pd.DataFrame:
    if budget_days == "full":
        return train_samples
    cutoff = train_end - pd.Timedelta(days=budget_days)
    return train_samples[train_samples["target_hour"] >= cutoff].reset_index(drop=True)


def fit_scalers(train_samples: pd.DataFrame, sequence_columns: list) -> dict:
    demand_pool = np.concatenate(
        [
            train_samples[sequence_columns].to_numpy().ravel(),
            train_samples["previous_week_lag"].to_numpy(),
            train_samples["citywide_demand_recent"].to_numpy(),
            train_samples["target"].to_numpy(),
        ]
    )
    demand_mean, demand_std = _mean_std(demand_pool)
    lat_mean, lat_std = _mean_std(train_samples["station_lat_norm"])
    lon_mean, lon_std = _mean_std(train_samples["station_lon_norm"])
    cap_mean, cap_std = _mean_std(train_samples["rack_capacity"])
    return {
        "demand": {"mean": demand_mean, "std": demand_std},
        "station_lat_norm": {"mean": lat_mean, "std": lat_std},
        "station_lon_norm": {"mean": lon_mean, "std": lon_std},
        "rack_capacity": {"mean": cap_mean, "std": cap_std},
    }


def apply_scalers(samples: pd.DataFrame, scalers: dict, sequence_columns: list) -> pd.DataFrame:
    scaled = samples.copy()
    demand = scalers["demand"]
    for col in [*sequence_columns, "previous_week_lag", "citywide_demand_recent"]:
        scaled[col] = (samples[col] - demand["mean"]) / demand["std"]
    for col in ["station_lat_norm", "station_lon_norm", "rack_capacity"]:
        s = scalers[col]
        scaled[col] = (samples[col] - s["mean"]) / s["std"]
    scaled["target_raw"] = samples["target"]
    scaled["target"] = (samples["target"] - demand["mean"]) / demand["std"]
    return scaled


def inverse_transform_target(scaled_target, scalers: dict):
    demand = scalers["demand"]
    return np.asarray(scaled_target) * demand["std"] + demand["mean"]


def build_city_dataset(panel: pd.DataFrame, stations: pd.DataFrame, config: dict) -> dict:
    lookback = config["sample"]["demand_lookback_hours"]
    covariate_order = config["sample"]["covariates"]

    samples, sequence_columns, feature_columns = build_supervised_samples(panel, stations, lookback)
    if set(covariate_order) != set(feature_columns):
        raise ValueError(
            f"configs/*.yaml sample.covariates {covariate_order} does not match "
            f"the covariates windowing.py builds {feature_columns}"
        )
    samples["split"] = assign_split(samples, config["split"])
    samples = samples[samples["split"].notna()].reset_index(drop=True)

    train_samples = samples[samples["split"] == "train"].reset_index(drop=True)
    scalers = fit_scalers(train_samples, sequence_columns)

    splits = {
        name: apply_scalers(samples[samples["split"] == name].reset_index(drop=True), scalers, sequence_columns)
        for name in ["train", "validation", "test"]
    }

    train_end = pd.Timestamp(config["split"]["train"]["end"], tz="UTC")
    budgets = {}
    for budget_days in config["budgets"]["target_history_days"]:
        cropped = apply_budget_cutoff(train_samples, budget_days, train_end)
        budgets[budget_days] = apply_scalers(cropped, scalers, sequence_columns)

    return {
        "splits": splits,
        "train_budgets": budgets,
        "scalers": scalers,
        "sequence_columns": sequence_columns,
        "covariate_order": covariate_order,
    }


def _output_columns(dataset: dict) -> list:
    return ["station_id", "hour", "target_hour", *dataset["sequence_columns"], *dataset["covariate_order"], "target", "target_raw", "split"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    interim_dir = Path(config["data"]["interim_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    out_dir = processed_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    stations = pd.read_parquet(interim_dir / "stations.parquet").rename(columns={"id": "station_id"})

    counts = []
    for city in config["cities"]:
        city_slug = city["name"].lower()
        panel = pd.read_parquet(processed_dir / f"panel_{city_slug}.parquet")
        dataset = build_city_dataset(panel, stations, config)

        for split_name, split_samples in dataset["splits"].items():
            split_samples[_output_columns(dataset)].to_parquet(
                out_dir / f"{city_slug}_{split_name}.parquet", index=False
            )
            counts.append({"city": city["name"], "budget": split_name, "samples": len(split_samples)})

        for budget_days, budget_samples in dataset["train_budgets"].items():
            budget_samples[_output_columns(dataset)].to_parquet(
                out_dir / f"{city_slug}_train_budget_{budget_days}.parquet", index=False
            )
            counts.append({"city": city["name"], "budget": f"train_budget_{budget_days}", "samples": len(budget_samples)})

        with open(out_dir / f"{city_slug}_scalers.json", "w") as f:
            json.dump(dataset["scalers"], f, indent=2)

    print(pd.DataFrame(counts).to_string(index=False))


if __name__ == "__main__":
    main()
