"""Training loop for GRUForecaster (src/models/gru.py).

Three regimes, all sharing the same fit()/predict_raw() machinery -- they
differ only in what data feeds train_loader/val_loader and, for
fine-tuning, the learning rate:

- target_only: a single city's GRU trained from scratch on that city's own
  budget-limited training data.
- pooled: one GRU trained on the source cities' (Bilbao, Vienna, Glasgow)
  full training data concatenated with the target city's budget-limited
  training data.
- pretrain_finetune: first pretrain_on_sources() trains a GRU on the source
  cities' combined data only; finetune_on_target() then continues training
  that same model on the target city's budget-limited data, at 1/10th the
  learning rate.

Reads the samples and per-city scalers src/features/windowing.py already
wrote (<city>_train_budget_<n>.parquet, _validation.parquet, _test.parquet,
_scalers.json) rather than rebuilding anything.

Rules carried over from a Colab run, each load-bearing:

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
5. Validation is always the TARGET city (Freiburg) -- that's the question
   pooled and fine-tuned are being compared on -- with one exception:
   pretrain_on_sources()'s own validation is the combined source cities'
   data, since there is no target-city signal yet at that point.
6. Scaling is per city. windowing.py already scales every city's samples
   with that city's own demand mean/std (Bilbao's raw mean is 5.85,
   Vienna's is 0.33 -- pooling raw values across cities would hand the
   model a mix of wildly different scales and invite negative transfer).
   Concatenating cities here always concatenates their already
   per-city-scaled parquet output, never raw values re-scaled by a single
   pooled scaler; evaluation always inverse-transforms through the target
   city's own scaler, since evaluation is always on the target city.
7. Pretraining is budget-independent -- it never touches the target city's
   data, so it does not need to be repeated per budget. main() calls
   pretrain_on_sources() exactly once and reuses those weights (via a fresh
   deepcopy per budget, so one budget's fine-tuning never leaks into
   another's starting point) across all four budgets. Retraining it per
   budget would be a 4x slowdown for identical work.
"""
import argparse
import copy
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


def make_loader(df: pd.DataFrame, sequence_columns: list, covariate_order: list, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(WindowedDataset(df, sequence_columns, covariate_order), batch_size=batch_size, shuffle=shuffle)


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


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: dict, device, learning_rate: float = None):
    # learning_rate overrides config["model"]["learning_rate"] -- used for
    # fine-tuning at 1/10th the pretraining rate (see finetune_on_target).
    lr = learning_rate if learning_rate is not None else config["model"]["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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


def _target_result_row(model, test_loader, target_scalers, device, method, target_city_slug, budget, n_train, best_val_loss) -> dict:
    y_true, y_pred = predict_raw(model, test_loader, target_scalers, device)
    metrics = regression_metrics(y_true, y_pred)
    return {
        "city": target_city_slug,
        "budget": budget,
        "method": method,
        "n_train": n_train,
        "n_test": len(y_true),
        "best_val_loss_scaled": best_val_loss,
        **metrics,
    }


def train_target_only(target_city_slug: str, budget, config: dict, samples_dir: Path, device) -> dict:
    seq_cols = sequence_column_names(config["sample"]["demand_lookback_hours"])
    cov_cols = config["sample"]["covariates"]
    batch_size = config["model"]["batch_size"]

    with open(samples_dir / f"{target_city_slug}_scalers.json") as f:
        target_scalers = json.load(f)

    train_df = pd.read_parquet(samples_dir / f"{target_city_slug}_train_budget_{budget}.parquet")
    val_df = pd.read_parquet(samples_dir / f"{target_city_slug}_validation.parquet")
    test_df = pd.read_parquet(samples_dir / f"{target_city_slug}_test.parquet")

    train_loader = make_loader(train_df, seq_cols, cov_cols, batch_size, shuffle=True)
    val_loader = make_loader(val_df, seq_cols, cov_cols, batch_size)
    test_loader = make_loader(test_df, seq_cols, cov_cols, batch_size)

    set_seed(config["model"]["seed"])
    model = build_model_from_config(config, n_covariates=len(cov_cols)).to(device)
    model, best_val_loss, _ = fit(model, train_loader, val_loader, config, device)

    row = _target_result_row(model, test_loader, target_scalers, device, "target_only_gru", target_city_slug, budget, len(train_df), best_val_loss)
    return {"model": model, "row": row}


def train_pooled(target_city_slug: str, source_city_slugs: list, budget, config: dict, samples_dir: Path, device) -> dict:
    seq_cols = sequence_column_names(config["sample"]["demand_lookback_hours"])
    cov_cols = config["sample"]["covariates"]
    batch_size = config["model"]["batch_size"]

    with open(samples_dir / f"{target_city_slug}_scalers.json") as f:
        target_scalers = json.load(f)

    # Sources always contribute their full history (no budget restriction --
    # only the target city's history is scarce by construction); each
    # frame is already scaled by that city's own scaler, so concatenating
    # them does not mix raw scales across cities (see rule 6 above).
    source_train_frames = [pd.read_parquet(samples_dir / f"{s}_train_budget_full.parquet") for s in source_city_slugs]
    target_train = pd.read_parquet(samples_dir / f"{target_city_slug}_train_budget_{budget}.parquet")
    train_df = pd.concat([*source_train_frames, target_train], ignore_index=True)

    val_df = pd.read_parquet(samples_dir / f"{target_city_slug}_validation.parquet")
    test_df = pd.read_parquet(samples_dir / f"{target_city_slug}_test.parquet")

    train_loader = make_loader(train_df, seq_cols, cov_cols, batch_size, shuffle=True)
    val_loader = make_loader(val_df, seq_cols, cov_cols, batch_size)
    test_loader = make_loader(test_df, seq_cols, cov_cols, batch_size)

    set_seed(config["model"]["seed"])
    model = build_model_from_config(config, n_covariates=len(cov_cols)).to(device)
    model, best_val_loss, _ = fit(model, train_loader, val_loader, config, device)

    row = _target_result_row(model, test_loader, target_scalers, device, "pooled_gru", target_city_slug, budget, len(train_df), best_val_loss)
    return {"model": model, "row": row}


def pretrain_on_sources(source_city_slugs: list, config: dict, samples_dir: Path, device):
    """Train once on the source cities' combined data only. Budget-
    independent by construction: the target city never appears here, so
    the caller (main()) runs this exactly once per program run and reuses
    the resulting weights for every budget's fine-tuning pass -- see
    finetune_on_target and rule 7 in the module docstring."""
    seq_cols = sequence_column_names(config["sample"]["demand_lookback_hours"])
    cov_cols = config["sample"]["covariates"]
    batch_size = config["model"]["batch_size"]

    source_train = pd.concat(
        [pd.read_parquet(samples_dir / f"{s}_train_budget_full.parquet") for s in source_city_slugs], ignore_index=True
    )
    # Exception to "validation is always the target city" (rule 5): there
    # is no target-city signal yet at the pretraining stage, so validation
    # here is the combined source cities' own validation data.
    source_val = pd.concat([pd.read_parquet(samples_dir / f"{s}_validation.parquet") for s in source_city_slugs], ignore_index=True)

    train_loader = make_loader(source_train, seq_cols, cov_cols, batch_size, shuffle=True)
    val_loader = make_loader(source_val, seq_cols, cov_cols, batch_size)

    set_seed(config["model"]["seed"])
    model = build_model_from_config(config, n_covariates=len(cov_cols)).to(device)
    model, best_val_loss, _ = fit(model, train_loader, val_loader, config, device)
    return model, best_val_loss


def finetune_on_target(pretrained_model: nn.Module, target_city_slug: str, budget, config: dict, samples_dir: Path, device) -> dict:
    seq_cols = sequence_column_names(config["sample"]["demand_lookback_hours"])
    cov_cols = config["sample"]["covariates"]
    batch_size = config["model"]["batch_size"]

    with open(samples_dir / f"{target_city_slug}_scalers.json") as f:
        target_scalers = json.load(f)

    train_df = pd.read_parquet(samples_dir / f"{target_city_slug}_train_budget_{budget}.parquet")
    val_df = pd.read_parquet(samples_dir / f"{target_city_slug}_validation.parquet")
    test_df = pd.read_parquet(samples_dir / f"{target_city_slug}_test.parquet")

    train_loader = make_loader(train_df, seq_cols, cov_cols, batch_size, shuffle=True)
    val_loader = make_loader(val_df, seq_cols, cov_cols, batch_size)
    test_loader = make_loader(test_df, seq_cols, cov_cols, batch_size)

    # Start from a fresh copy of the pretrained weights every time -- one
    # budget's fine-tuning must never carry over into another's starting
    # point (see rule 7).
    model = copy.deepcopy(pretrained_model)
    finetune_lr = config["model"]["learning_rate"] / 10

    set_seed(config["model"]["seed"])
    model, best_val_loss, _ = fit(model, train_loader, val_loader, config, device, learning_rate=finetune_lr)

    row = _target_result_row(
        model, test_loader, target_scalers, device, "source_pretrained_finetuned_gru", target_city_slug, budget, len(train_df), best_val_loss
    )
    return {"model": model, "row": row}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--city", default=None, help="Target city name (default: the config's target-role city)")
    parser.add_argument(
        "--regime",
        default="all",
        choices=["target_only", "pooled", "pretrain_finetune", "all"],
        help="Which regime(s) to run",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    target_city_slug = (args.city or next(c["name"] for c in config["cities"] if c["role"] == "target")).lower()
    source_city_slugs = [c["name"].lower() for c in config["cities"] if c["role"] == "source"]
    samples_dir = Path(config["data"]["processed_dir"]) / "samples"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = Path("results")
    checkpoints_dir = results_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    regimes = ["target_only", "pooled", "pretrain_finetune"] if args.regime == "all" else [args.regime]
    budgets = config["budgets"]["target_history_days"]
    rows = []

    if "target_only" in regimes:
        for budget in budgets:
            result = train_target_only(target_city_slug, budget, config, samples_dir, device)
            rows.append(result["row"])
            torch.save(result["model"].state_dict(), checkpoints_dir / f"{target_city_slug}_target_only_budget_{budget}.pt")

    if "pooled" in regimes:
        for budget in budgets:
            result = train_pooled(target_city_slug, source_city_slugs, budget, config, samples_dir, device)
            rows.append(result["row"])
            torch.save(result["model"].state_dict(), checkpoints_dir / f"{target_city_slug}_pooled_budget_{budget}.pt")

    if "pretrain_finetune" in regimes:
        # Trained once, reused across every budget below -- see rule 7.
        pretrained_model, pretrain_val_loss = pretrain_on_sources(source_city_slugs, config, samples_dir, device)
        sources_key = "_".join(source_city_slugs)
        torch.save(pretrained_model.state_dict(), checkpoints_dir / f"{sources_key}_pretrained.pt")
        print(f"Pretrained on {source_city_slugs}, best source-validation MAE (scaled): {pretrain_val_loss:.4f}")

        for budget in budgets:
            result = finetune_on_target(pretrained_model, target_city_slug, budget, config, samples_dir, device)
            rows.append(result["row"])
            torch.save(result["model"].state_dict(), checkpoints_dir / f"{target_city_slug}_pretrain_finetune_budget_{budget}.pt")

    results = pd.DataFrame(rows)
    results.to_csv(results_dir / "gru_metrics.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
