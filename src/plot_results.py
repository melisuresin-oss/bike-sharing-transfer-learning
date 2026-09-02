"""Plot the transfer-gain curve from results/all_metrics.csv.

Produces results/transfer_gain_curve.png: MAE vs target-history budget
for all 5 methods, with the x-axis on a log scale to spread out the
short-budget region (1d, 7d, 30d, full).

Usage:
    python src/plot_results.py
    python src/plot_results.py --results results/all_metrics.csv
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BUDGET_ORDER = [1, 7, 30, "full"]
BUDGET_LABELS = {1: "1 day", 7: "7 days", 30: "30 days", "full": "Full"}
BUDGET_X = {1: 1, 7: 7, 30: 30, "full": 170}

METHOD_STYLE = {
    "historical_average":              {"label": "Historical Average",       "color": "#888888", "linestyle": "--",  "marker": "s"},
    "persistence":                     {"label": "Persistence",              "color": "#aaaaaa", "linestyle": ":",   "marker": "D"},
    "target_only_gru":                 {"label": "Target-only GRU",          "color": "#e07b39", "linestyle": "-",   "marker": "o"},
    "pooled_gru":                      {"label": "Pooled GRU",               "color": "#4e9af1", "linestyle": "-",   "marker": "^"},
    "source_pretrained_finetuned_gru": {"label": "Pretrained + Fine-tuned",  "color": "#2db37a", "linestyle": "-",   "marker": "*"},
}

METHOD_ORDER = list(METHOD_STYLE.keys())


def parse_budget(b):
    try:
        return int(b)
    except (ValueError, TypeError):
        return str(b).lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/all_metrics.csv")
    parser.add_argument("--out", default="results/transfer_gain_curve.png")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    df["budget"] = df["budget"].apply(parse_budget)

    present_methods = set(df["method"].unique())
    missing = [m for m in METHOD_ORDER if m not in present_methods]
    if missing:
        print(f"Warning: results CSV is missing methods {missing} — they will be skipped.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    metrics = ["mae", "rmse", "wape"]
    metric_labels = {"mae": "MAE (departures)", "rmse": "RMSE (departures)", "wape": "WAPE"}

    for ax, metric in zip(axes, metrics):
        for method in METHOD_ORDER:
            style = METHOD_STYLE[method]
            subset = df[df["method"] == method].copy()
            if subset.empty:
                continue
            subset["x"] = subset["budget"].map(BUDGET_X)
            subset = subset.sort_values("x")
            ax.plot(
                subset["x"],
                subset[metric],
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2,
                markersize=7,
            )

        ax.set_xscale("log")
        ax.set_xticks(list(BUDGET_X.values()))
        ax.set_xticklabels(list(BUDGET_LABELS.values()))
        ax.set_xlabel("Target-city history budget")
        ax.set_ylabel(metric_labels[metric])
        ax.set_title(metric.upper())
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(METHOD_ORDER), bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle("Cross-City Transfer Learning — Freiburg Demand Forecasting", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
