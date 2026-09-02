import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.gru import GRUForecaster
from src.training.train import (
    EarlyStopper,
    WindowedDataset,
    build_loss_fn,
    finetune_on_target,
    fit,
    pretrain_on_sources,
    predict_raw,
    set_seed,
    train_pooled,
)

SEQ_COLS = [f"lag_{k}" for k in range(23, -1, -1)]
COV_COLS = ["cov_a", "cov_b"]


def make_df(n, seed=0):
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(size=n) for col in SEQ_COLS}
    for col in COV_COLS:
        data[col] = rng.normal(size=n)
    scaled_target = rng.normal(size=n)
    data["target"] = scaled_target
    data["target_raw"] = scaled_target * 2.0 + 5.0  # arbitrary consistent raw units
    return pd.DataFrame(data)


def write_city_parquet(samples_dir, city_slug, name, n, seed):
    """name is e.g. 'validation', 'test', 'train_budget_full', 'train_budget_7'."""
    df = make_df(n, seed=seed)
    df.to_parquet(samples_dir / f"{city_slug}_{name}.parquet")
    return df


def write_scalers(samples_dir, city_slug, mean=0.0, std=1.0):
    with open(samples_dir / f"{city_slug}_scalers.json", "w") as f:
        json.dump({"demand": {"mean": mean, "std": std}}, f)


def make_regime_config(learning_rate=0.02):
    return {
        "sample": {"demand_lookback_hours": 24, "covariates": COV_COLS},
        "model": {
            "hidden_size": 4,
            "num_layers": 1,
            "dropout": 0.0,
            "learning_rate": learning_rate,
            "batch_size": 8,
            "max_epochs": 1,
            "early_stopping_patience": 1,
            "seed": 0,
        },
    }


def test_loss_function_is_l1_mae_not_smooth_l1_or_mse():
    loss_fn = build_loss_fn()
    assert isinstance(loss_fn, nn.L1Loss)
    assert not isinstance(loss_fn, nn.SmoothL1Loss)
    assert not isinstance(loss_fn, nn.MSELoss)

    pred = torch.tensor([0.0, 5.0, -3.0])
    target = torch.tensor([2.0, 1.0, 0.0])
    expected_mae = torch.tensor([2.0, 4.0, 3.0]).mean()
    assert torch.allclose(loss_fn(pred, target), expected_mae)


def test_fit_signature_has_no_test_loader_parameter():
    # Structural guard: early stopping physically cannot see test data if
    # fit() never receives it.
    params = set(inspect.signature(fit).parameters)
    assert "test_loader" not in params
    assert {"model", "train_loader", "val_loader", "config", "device"} <= params


def test_early_stopper_keeps_the_best_epoch_not_the_last():
    stopper = EarlyStopper(patience=2)
    losses = [5.0, 4.0, 4.5, 4.6]  # improves at epoch 1, then two non-improving epochs
    stopped_at = None
    for i, loss in enumerate(losses):
        state = {"epoch_marker": torch.tensor([float(i)])}
        if stopper.step(loss, state):
            stopped_at = i
            break
    assert stopped_at == 3
    assert stopper.best_loss == pytest.approx(4.0)
    # best_state must be the one captured at the improving epoch (index 1),
    # not the last (non-improving) epoch seen.
    assert stopper.best_state["epoch_marker"].item() == 1.0


def test_predict_raw_inverse_transforms_and_clips_negative_predictions():
    class ConstantModel(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, sequence, covariates):
            return torch.full((sequence.shape[0],), self.value)

    scalers = {"demand": {"mean": 10.0, "std": 4.0}}
    df = make_df(n=6)
    loader = DataLoader(WindowedDataset(df, SEQ_COLS, COV_COLS), batch_size=4)

    # scaled prediction of -5 -> raw = -5*4 + 10 = -10 -> must clip to 0
    model = ConstantModel(-5.0)
    y_true, y_pred = predict_raw(model, loader, scalers, torch.device("cpu"))
    assert (y_pred == 0.0).all()

    # scaled prediction of 0 -> raw = 10 (the city mean), never clipped
    model = ConstantModel(0.0)
    _, y_pred = predict_raw(model, loader, scalers, torch.device("cpu"))
    assert np.allclose(y_pred, 10.0)

    np.testing.assert_allclose(y_true, df["target_raw"].to_numpy(), rtol=1e-6)


def test_fit_runs_end_to_end_on_synthetic_data_and_uses_only_validation():
    set_seed(0)
    train_df = make_df(n=40, seed=1)
    val_df = make_df(n=16, seed=2)
    train_loader = DataLoader(WindowedDataset(train_df, SEQ_COLS, COV_COLS), batch_size=8, shuffle=True)
    val_loader = DataLoader(WindowedDataset(val_df, SEQ_COLS, COV_COLS), batch_size=8)

    model = GRUForecaster(n_covariates=len(COV_COLS), hidden_size=4, num_layers=1, dropout=0.0)
    config = {
        "model": {
            "learning_rate": 0.01,
            "early_stopping_patience": 2,
            "max_epochs": 4,
            "gradient_clip_norm": 1.0,
        }
    }
    fitted_model, best_val_loss, history = fit(model, train_loader, val_loader, config, torch.device("cpu"))
    assert isinstance(fitted_model, GRUForecaster)
    assert 1 <= len(history) <= 4
    assert best_val_loss == pytest.approx(min(history))


def test_fit_uses_learning_rate_override_not_configs_when_given(monkeypatch):
    captured_lrs = []
    real_adam = torch.optim.Adam

    def spy_adam(params, lr, *args, **kwargs):
        captured_lrs.append(lr)
        return real_adam(params, lr=lr, *args, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", spy_adam)

    model = GRUForecaster(n_covariates=len(COV_COLS), hidden_size=4, num_layers=1, dropout=0.0)
    train_loader = DataLoader(WindowedDataset(make_df(n=8, seed=1), SEQ_COLS, COV_COLS), batch_size=8)
    val_loader = DataLoader(WindowedDataset(make_df(n=8, seed=2), SEQ_COLS, COV_COLS), batch_size=8)
    config = {"model": {"learning_rate": 0.05, "early_stopping_patience": 1, "max_epochs": 1}}

    fit(model, train_loader, val_loader, config, torch.device("cpu"), learning_rate=0.005)
    assert captured_lrs == [pytest.approx(0.005)]  # override used, not config's 0.05


def test_train_pooled_combines_source_full_data_with_target_budget_data(tmp_path):
    # Sources contribute their FULL history regardless of the target's
    # budget; only the target city's contribution is budget-limited.
    write_city_parquet(tmp_path, "bilbao", "train_budget_full", n=20, seed=1)
    write_city_parquet(tmp_path, "vienna", "train_budget_full", n=15, seed=2)
    write_city_parquet(tmp_path, "freiburg", "train_budget_7", n=6, seed=3)
    write_city_parquet(tmp_path, "freiburg", "validation", n=8, seed=4)
    write_city_parquet(tmp_path, "freiburg", "test", n=9, seed=5)
    write_scalers(tmp_path, "freiburg")

    result = train_pooled("freiburg", ["bilbao", "vienna"], 7, make_regime_config(), tmp_path, torch.device("cpu"))
    assert result["row"]["n_train"] == 20 + 15 + 6
    assert result["row"]["n_test"] == 9
    assert result["row"]["method"] == "pooled_gru"
    assert result["row"]["budget"] == 7


def test_pretrain_on_sources_never_reads_target_city_files(tmp_path):
    # No freiburg_*.parquet exists in this directory at all -- if
    # pretrain_on_sources tried to touch target data, this would raise
    # FileNotFoundError instead of completing.
    write_city_parquet(tmp_path, "bilbao", "train_budget_full", n=20, seed=1)
    write_city_parquet(tmp_path, "vienna", "train_budget_full", n=15, seed=2)
    write_city_parquet(tmp_path, "bilbao", "validation", n=5, seed=3)
    write_city_parquet(tmp_path, "vienna", "validation", n=6, seed=4)

    model, val_loss = pretrain_on_sources(["bilbao", "vienna"], make_regime_config(), tmp_path, torch.device("cpu"))
    assert isinstance(model, GRUForecaster)
    assert np.isfinite(val_loss)


def test_finetune_uses_one_tenth_the_pretraining_learning_rate(tmp_path, monkeypatch):
    write_city_parquet(tmp_path, "freiburg", "train_budget_7", n=10, seed=1)
    write_city_parquet(tmp_path, "freiburg", "validation", n=5, seed=2)
    write_city_parquet(tmp_path, "freiburg", "test", n=5, seed=3)
    write_scalers(tmp_path, "freiburg")

    captured_lrs = []
    real_adam = torch.optim.Adam

    def spy_adam(params, lr, *args, **kwargs):
        captured_lrs.append(lr)
        return real_adam(params, lr=lr, *args, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", spy_adam)

    pretrained = GRUForecaster(n_covariates=len(COV_COLS), hidden_size=4, num_layers=1, dropout=0.0)
    config = make_regime_config(learning_rate=0.02)
    finetune_on_target(pretrained, "freiburg", 7, config, tmp_path, torch.device("cpu"))
    assert captured_lrs == [pytest.approx(0.002)]


def test_finetune_starts_from_a_copy_and_does_not_mutate_the_pretrained_model(tmp_path):
    # Each budget's fine-tuning must start from the SAME pretrained weights,
    # so reusing one pretrained model across all four budgets is only safe
    # if finetune_on_target never mutates it in place.
    write_city_parquet(tmp_path, "freiburg", "train_budget_1", n=10, seed=1)
    write_city_parquet(tmp_path, "freiburg", "validation", n=5, seed=2)
    write_city_parquet(tmp_path, "freiburg", "test", n=5, seed=3)
    write_scalers(tmp_path, "freiburg")

    pretrained = GRUForecaster(n_covariates=len(COV_COLS), hidden_size=4, num_layers=1, dropout=0.0)
    original_state = {k: v.clone() for k, v in pretrained.state_dict().items()}

    finetune_on_target(pretrained, "freiburg", 1, make_regime_config(), tmp_path, torch.device("cpu"))

    for key, value in pretrained.state_dict().items():
        assert torch.equal(value, original_state[key])
