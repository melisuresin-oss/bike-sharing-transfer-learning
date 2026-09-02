import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.run_experiment import main

SEQ_COLS = [f"lag_{k}" for k in range(23, -1, -1)]
COV_COLS = [
    "previous_week_lag", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "is_weekend", "station_lat_norm", "station_lon_norm", "rack_capacity", "citywide_demand_recent",
]
BUDGETS = [1, "full"]
SOURCE_SLUGS = ["bilbao", "vienna"]
TARGET_SLUG = "freiburg"


def make_df(n, seed=0):
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(size=n) for col in SEQ_COLS}
    for col in COV_COLS:
        data[col] = rng.normal(size=n)
    data["target"] = rng.normal(size=n)
    data["target_raw"] = np.abs(rng.normal(size=n)) * 5
    data["station_id"] = 1
    data["target_hour"] = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    data["lag_0"] = np.abs(data["lag_0"])
    return pd.DataFrame(data)


def write_city_files(samples_dir, city_slug, budgets, seed=0):
    for split in ["validation", "test"]:
        make_df(30, seed=seed).to_parquet(samples_dir / f"{city_slug}_{split}.parquet")
    make_df(60, seed=seed).to_parquet(samples_dir / f"{city_slug}_train_budget_full.parquet")
    for b in budgets:
        make_df(10, seed=seed).to_parquet(samples_dir / f"{city_slug}_train_budget_{b}.parquet")
    with open(samples_dir / f"{city_slug}_scalers.json", "w") as f:
        json.dump({"demand": {"mean": 0.0, "std": 1.0}}, f)


def make_config(processed_dir, budgets):
    return {
        "data": {"processed_dir": str(processed_dir)},
        "budgets": {"target_history_days": budgets},
        "cities": [
            {"name": "Bilbao", "role": "source"},
            {"name": "Vienna", "role": "source"},
            {"name": "Freiburg", "role": "target"},
        ],
        "sample": {"demand_lookback_hours": 24, "covariates": COV_COLS},
        "model": {
            "hidden_size": 4, "num_layers": 1, "dropout": 0.0, "bidirectional": False,
            "learning_rate": 0.01, "batch_size": 8, "max_epochs": 2,
            "early_stopping_patience": 2, "gradient_clip_norm": 1.0, "seed": 0,
        },
    }


@pytest.fixture()
def experiment_env(tmp_path):
    processed_dir = tmp_path / "processed"
    samples_dir = processed_dir / "samples"
    samples_dir.mkdir(parents=True)

    for slug in SOURCE_SLUGS + [TARGET_SLUG]:
        write_city_files(samples_dir, slug, BUDGETS)

    config = make_config(processed_dir, BUDGETS)
    config_path = tmp_path / "config.yaml"
    import yaml
    config_path.write_text(yaml.dump(config))
    return tmp_path, config_path


def test_output_csv_has_all_methods_and_budgets(experiment_env):
    tmp_path, config_path = experiment_env
    out = tmp_path / "results" / "all_metrics.csv"
    with patch("sys.argv", ["run_experiment.py", "--config", str(config_path), "--output", str(out)]):
        main()

    assert out.exists()
    df = pd.read_csv(out)
    expected_methods = {
        "historical_average", "persistence",
        "target_only_gru", "pooled_gru", "source_pretrained_finetuned_gru",
    }
    assert set(df["method"].unique()) == expected_methods
    assert set(str(b) for b in df["budget"].unique()) == {str(b) for b in BUDGETS}


def test_output_csv_has_no_nan_metrics(experiment_env):
    tmp_path, config_path = experiment_env
    out = tmp_path / "results" / "metrics.csv"
    with patch("sys.argv", ["run_experiment.py", "--config", str(config_path), "--output", str(out)]):
        main()

    df = pd.read_csv(out)
    assert df[["mae", "rmse"]].notna().all().all()


def test_no_overwrite_flag_aborts_if_output_exists(experiment_env):
    tmp_path, config_path = experiment_env
    out = tmp_path / "results" / "metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("existing")

    with pytest.raises(FileExistsError):
        with patch("sys.argv", ["run_experiment.py", "--config", str(config_path),
                                "--output", str(out), "--no-overwrite"]):
            main()


def test_checkpoints_are_saved(experiment_env):
    tmp_path, config_path = experiment_env
    out = tmp_path / "results" / "metrics.csv"
    with patch("sys.argv", ["run_experiment.py", "--config", str(config_path), "--output", str(out)]):
        main()

    checkpoints = list((tmp_path / "results" / "checkpoints").glob("*.pt"))
    assert len(checkpoints) > 0
