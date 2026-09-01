"""Training loop for GRUForecaster (src/models/gru.py).

This is the target-only regime: a single city's GRU is trained from scratch
on that same city's own (budget-limited) training data. Pooled (source data
+ available target history) and source-pretrained-then-fine-tuned come
later, reusing the same GRUForecaster and the fit()/predict_raw() functions
here -- only how train_loader gets built differs between regimes.

Reads the samples and per-city scaler src/features/windowing.py already
wrote (<city>_train_budget_<n>.parquet, _validation.parquet, _test.parquet,
_scalers.json) rather than rebuilding anything.

Four rules carried over from a Colab run, each load-bearing:

1. Early stopping tracks validation loss only. fit() takes a train_loader
   and a val_loader -- no test_loader parameter exists, so there is no
   accidental path from the test split into a training decision. The test
   set is only ever touched once, in predict_raw(), after training is over.
2. The training loss is nn.L1Loss (MAE) -- see build_loss_fn. The project's
   primary metric is MAE, so the model is trained directly against it
   rather than a proxy like MSE or SmoothL1Loss's default beta=1.0 (which
   behaves close to MSE in practice).
3. Both the training loss and early stopping operate in *scaled* space --
   consistent with what the optimizer actually sees. Final reported metrics
   are computed in raw units: predict_raw() inverse-transforms predictions
   through the city's saved demand scaler (windowing.inverse_transform_target)
   before src/eval/metrics.regression_metrics ever sees them.
4. Negative raw predictions are clipped to 0 in predict_raw(), *after* the
   inverse transform -- never inside the model (see gru.py's no-softplus
   rule for why the model itself must stay unconstrained).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

# Running this file directly (`python src/training/train.py`, matching how
# build_panel.py/windowing.py/baselines.py are invoked) only puts
# src/training/ on sys.path, not the repo root -- the imports below need
# the root there too.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.metrics import regression_metrics
from src.features.windowing import inverse_transform_target, sequence_column_names
from src.models.gru import build_model_from_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loss_fn() -> nn.Module:
    return nn.L1Loss()


class WindowedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sequence_columns: list, covariate_order: list):
        self.sequence = torch.tensor(df[sequence_columns].to_numpy(dtype=np.float32)).unsqueeze(-1)
        self.covariates = torch.tensor(df[covariate_order].to_numpy(dtype=np.float32))
        self.target = torch.tensor(df["target"].to_numpy(dtype=np.float32))
        self.target_raw = torch.tensor(df["target_raw"].to_numpy(dtype=np.float32))

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.sequence[idx], self.covariates[idx], self.target[idx], self.target_raw[idx]


class EarlyStopper:
    """Tracks a validation loss handed to it -- has no notion of a test set,
    so it cannot leak one even by mistake. See predict_raw() for the only
    place the test split is used, strictly after fit() returns."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best_loss = float("inf")
        self.best_state = None
        self.epochs_without_improvement = 0

    def step(self, val_loss: float, state_dict) -> bool:
        """Record this epoch's validation loss; returns True if training should stop now."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_state = {k: v.detach().clone() for k, v in state_dict.items()}
            self.epochs_without_improvement = 0
            return False
        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for seq, cov, target, _ in loader:
        seq, cov, target = seq.to(device), cov.to(device), target.to(device)
        loss = loss_fn(model(seq, cov), target)
        total += loss.item() * len(target)
        n += len(target)
    return total / n


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: dict, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])
    loss_fn = build_loss_fn()
    stopper = EarlyStopper(config["model"]["early_stopping_patience"])
    grad_clip = config["model"].get("gradient_clip_norm")
    history = []

    for _ in range(config["model"]["max_epochs"]):
        model.train()
        for seq, cov, target, _ in train_loader:
            seq, cov, target = seq.to(device), cov.to(device), target.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(seq, cov), target)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        history.append(val_loss)
        if stopper.step(val_loss, model.state_dict()):
            break

    model.load_state_dict(stopper.best_state)
    return model, stopper.best_loss, history


@torch.no_grad()
def predict_raw(model: nn.Module, loader: DataLoader, scalers: dict, device) -> tuple:
    model.eval()
    scaled_preds, raw_targets = [], []
    for seq, cov, _, target_raw in loader:
        seq, cov = seq.to(device), cov.to(device)
        scaled_preds.append(model(seq, cov).cpu().numpy())
        raw_targets.append(target_raw.numpy())
    scaled_preds = np.concatenate(scaled_preds)
    raw_targets = np.concatenate(raw_targets)

    raw_preds = inverse_transform_target(scaled_preds, scalers)
    # Negativity is handled here, after inverse-transforming to raw units --
    # never inside the model. See src/models/gru.py's module docstring.
    raw_preds = np.clip(raw_preds, a_min=0, a_max=None)
    return raw_targets, raw_preds


def train_target_only(city_slug: str, budget, config: dict, samples_dir: Path, device) -> dict:
    seq_cols = sequence_column_names(config["sample"]["demand_lookback_hours"])
    cov_cols = config["sample"]["covariates"]

    with open(samples_dir / f"{city_slug}_scalers.json") as f:
        scalers = json.load(f)

    train_df = pd.read_parquet(samples_dir / f"{city_slug}_train_budget_{budget}.parquet")
    val_df = pd.read_parquet(samples_dir / f"{city_slug}_validation.parquet")
    test_df = pd.read_parquet(samples_dir / f"{city_slug}_test.parquet")

    batch_size = config["model"]["batch_size"]
    train_loader = DataLoader(WindowedDataset(train_df, seq_cols, cov_cols), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WindowedDataset(val_df, seq_cols, cov_cols), batch_size=batch_size)
    test_loader = DataLoader(WindowedDataset(test_df, seq_cols, cov_cols), batch_size=batch_size)

    set_seed(config["model"]["seed"])
    model = build_model_from_config(config, n_covariates=len(cov_cols)).to(device)
    model, best_val_loss, _ = fit(model, train_loader, val_loader, config, device)

    y_true, y_pred = predict_raw(model, test_loader, scalers, device)
    metrics = regression_metrics(y_true, y_pred)

    return {
        "model": model,
        "row": {
            "city": city_slug,
            "budget": budget,
            "method": "target_only_gru",
            "n_train": len(train_df),
            "n_test": len(test_df),
            "best_val_loss_scaled": best_val_loss,
            **metrics,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--city", default=None, help="City name to train on (default: the config's target-role city)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    target_city = args.city or next(c["name"] for c in config["cities"] if c["role"] == "target")
    city_slug = target_city.lower()
    samples_dir = Path(config["data"]["processed_dir"]) / "samples"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = Path("results")
    checkpoints_dir = results_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for budget in config["budgets"]["target_history_days"]:
        result = train_target_only(city_slug, budget, config, samples_dir, device)
        rows.append(result["row"])
        torch.save(result["model"].state_dict(), checkpoints_dir / f"{city_slug}_target_only_budget_{budget}.pt")

    results = pd.DataFrame(rows)
    results.to_csv(results_dir / "target_only_gru_metrics.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
