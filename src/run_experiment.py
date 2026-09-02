"""Run all 5 methods from the proposal across all target-history budgets.

Methods evaluated:
  1. historical_average    -- per-(station, hour-of-week) mean from training data
  2. persistence           -- predict last observed value
  3. target_only_gru       -- GRU trained from scratch on Freiburg budget only
  4. pooled_gru            -- GRU trained on source cities + Freiburg budget
  5. source_pretrained_finetuned_gru -- pretrained on sources, fine-tuned on Freiburg

Output: results/all_metrics.csv with one row per (method, budget).

Usage:
    python src/run_experiment.py
    python src/run_experiment.py --config configs/colab.yaml
    python src/run_experiment.py --output results/run2.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.baselines import evaluate_city as evaluate_baselines
from src.training.train import (
    finetune_on_target,
    pretrain_on_sources,
    train_pooled,
    train_target_only,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="results/all_metrics.csv", help="Path for the output CSV")
    parser.add_argument("--no-overwrite", action="store_true", help="Abort if the output file already exists")
    args = parser.parse_args()

    out_path = Path(args.output)
    if args.no_overwrite and out_path.exists():
        raise FileExistsError(f"{out_path} already exists. Use a different --output path or remove --no-overwrite.")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    samples_dir = Path(config["data"]["processed_dir"]) / "samples"
    budgets = config["budgets"]["target_history_days"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = out_path.parent
    checkpoints_dir = results_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    target_city = next(c for c in config["cities"] if c["role"] == "target")
    source_cities = [c for c in config["cities"] if c["role"] == "source"]
    target_slug = target_city["name"].lower()
    source_slugs = [c["name"].lower() for c in source_cities]

    print(f"Target city: {target_city['name']}")
    print(f"Source cities: {[c['name'] for c in source_cities]}")
    print(f"Budgets: {budgets}")
    print(f"Device: {device}\n")

    all_rows = []

    # 1 & 2: Baselines (historical average + persistence)
    print("Running baselines...")
    baseline_rows = evaluate_baselines(target_slug, samples_dir, budgets)
    all_rows.extend(baseline_rows)
    print(f"  Done. {len(baseline_rows)} rows.\n")

    # 3: Target-only GRU
    print("Running target_only_gru...")
    for budget in budgets:
        print(f"  budget={budget}")
        result = train_target_only(target_slug, budget, config, samples_dir, device)
        all_rows.append(result["row"])
        torch.save(result["model"].state_dict(), checkpoints_dir / f"{target_slug}_target_only_budget_{budget}.pt")
    print()

    # 4: Pooled GRU
    print("Running pooled_gru...")
    for budget in budgets:
        print(f"  budget={budget}")
        result = train_pooled(target_slug, source_slugs, budget, config, samples_dir, device)
        all_rows.append(result["row"])
        torch.save(result["model"].state_dict(), checkpoints_dir / f"{target_slug}_pooled_budget_{budget}.pt")
    print()

    # 5: Source-pretrained + fine-tuned GRU
    # Pretrain once on source cities, then fine-tune separately per budget.
    print("Running source_pretrained_finetuned_gru...")
    print("  Pretraining on source cities (runs once)...")
    pretrained_model, _ = pretrain_on_sources(source_slugs, config, samples_dir, device)
    sources_key = "_".join(source_slugs)
    torch.save(pretrained_model.state_dict(), checkpoints_dir / f"{sources_key}_pretrained.pt")
    for budget in budgets:
        print(f"  Fine-tuning budget={budget}")
        result = finetune_on_target(pretrained_model, target_slug, budget, config, samples_dir, device)
        all_rows.append(result["row"])
        torch.save(result["model"].state_dict(), checkpoints_dir / f"{target_slug}_finetuned_budget_{budget}.pt")
    print()

    results = pd.DataFrame(all_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")
    print(results[["method", "budget", "mae", "rmse", "wape"]].to_string(index=False))


if __name__ == "__main__":
    main()
