"""Historical-average and persistence baselines, evaluated in original units.

Both read the samples src/features/windowing.py already wrote to
data/processed/samples/ -- <city>_train_budget_<budget>.parquet for fitting
and <city>_test.parquet for evaluation -- rather than rebuilding samples from
the panel.

Windowing's saved parquet files store demand-related features (the lag
sequence, previous_week_lag, citywide_demand_recent, target) scaled with each
city's shared "demand" scaler, plus the target's raw value under
target_raw. Persistence needs the CURRENT hour's raw departure count
(lag_0), which is only saved in scaled form -- it's recovered by
inverse-transforming lag_0 through that same demand scaler (saved by
windowing.py as <city>_scalers.json), rather than re-deriving it from the
panel.

historical_average is trained per budget (its whole point is to show how
much history a simple model needs); persistence needs no training at all,
so its budget loop just repeats the same prediction under each budget label
for a consistent output table.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Running this file directly (`python src/models/baselines.py`, matching how
# build_panel.py and windowing.py are invoked) only puts src/models/ on
# sys.path, not the repo root -- the imports below need the root there too.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.metrics import regression_metrics
from src.features.windowing import inverse_transform_target


def hour_of_week(timestamps: pd.Series) -> pd.Series:
    return timestamps.dt.dayofweek * 24 + timestamps.dt.hour


def fit_historical_average(train: pd.DataFrame) -> dict:
    """Mean raw target per (station_id, hour_of_week), with a per-station
    mean and a global mean as fallbacks for combinations never seen in
    training (see predict_historical_average)."""
    hw = hour_of_week(train["target_hour"])
    df = train[["station_id", "target_raw"]].assign(hour_of_week=hw)
    return {
        "station_hour": df.groupby(["station_id", "hour_of_week"])["target_raw"].mean(),
        "station": df.groupby("station_id")["target_raw"].mean(),
        "global": float(df["target_raw"].mean()),
    }


def predict_historical_average(model: dict, eval_df: pd.DataFrame) -> np.ndarray:
    lookup = eval_df[["station_id"]].assign(hour_of_week=hour_of_week(eval_df["target_hour"]))
    lookup = lookup.merge(
        model["station_hour"].rename("pred_station_hour").reset_index(), on=["station_id", "hour_of_week"], how="left"
    )
    lookup = lookup.merge(model["station"].rename("pred_station").reset_index(), on="station_id", how="left")
    pred = lookup["pred_station_hour"].fillna(lookup["pred_station"]).fillna(model["global"])
    return pred.to_numpy()


def predict_persistence(eval_df: pd.DataFrame, scalers: dict) -> np.ndarray:
    return inverse_transform_target(eval_df["lag_0"], scalers)


def evaluate_city(city_slug: str, samples_dir: Path, budgets: list) -> list:
    with open(samples_dir / f"{city_slug}_scalers.json") as f:
        scalers = json.load(f)
    test = pd.read_parquet(samples_dir / f"{city_slug}_test.parquet")
    y_true = test["target_raw"].to_numpy()

    persistence_pred = predict_persistence(test, scalers)
    persistence_metrics = regression_metrics(y_true, persistence_pred)

    rows = []
    for budget in budgets:
        train = pd.read_parquet(samples_dir / f"{city_slug}_train_budget_{budget}.parquet")
        ha_pred = predict_historical_average(fit_historical_average(train), test)
        ha_metrics = regression_metrics(y_true, ha_pred)

        rows.append({"city": city_slug, "budget": budget, "method": "historical_average", "n_train": len(train), "n_test": len(test), **ha_metrics})
        # persistence doesn't use train data or depend on budget, but is
        # still reported once per budget for a consistent comparison table.
        rows.append({"city": city_slug, "budget": budget, "method": "persistence", "n_train": len(train), "n_test": len(test), **persistence_metrics})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    samples_dir = Path(config["data"]["processed_dir"]) / "samples"
    budgets = config["budgets"]["target_history_days"]

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for city in config["cities"]:
        all_rows.extend(evaluate_city(city["name"].lower(), samples_dir, budgets))

    results = pd.DataFrame(all_rows)
    results.to_csv(results_dir / "baseline_metrics.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
