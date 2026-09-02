import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plot_results import main, parse_budget

ALL_METHODS = [
    "historical_average", "persistence",
    "target_only_gru", "pooled_gru", "source_pretrained_finetuned_gru",
]
BUDGETS = [1, 7, 30, "full"]


def make_results_csv(path, methods=None, budgets=None):
    methods = methods or ALL_METHODS
    budgets = budgets or BUDGETS
    rows = []
    for method in methods:
        for budget in budgets:
            rows.append({"method": method, "budget": budget, "mae": 1.0, "rmse": 1.5, "wape": 0.3,
                         "n_train": 100, "n_test": 50, "best_val_loss_scaled": 0.5})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def test_parse_budget_handles_int_and_full_string():
    assert parse_budget(1) == 1
    assert parse_budget("7") == 7
    assert parse_budget("full") == "full"
    assert parse_budget("FULL") == "full"


def test_plot_saves_png_file(tmp_path):
    csv = make_results_csv(tmp_path / "metrics.csv")
    out = tmp_path / "curve.png"
    with patch("sys.argv", ["plot_results.py", "--results", str(csv), "--out", str(out)]):
        main()
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_does_not_crash_with_missing_methods(tmp_path, capsys):
    # Only two methods present — the other three should be silently skipped.
    csv = make_results_csv(tmp_path / "metrics.csv", methods=["historical_average", "persistence"])
    out = tmp_path / "curve.png"
    with patch("sys.argv", ["plot_results.py", "--results", str(csv), "--out", str(out)]):
        main()
    assert out.exists()
    captured = capsys.readouterr()
    assert "Warning" in captured.out


def test_plot_creates_output_directory_if_missing(tmp_path):
    csv = make_results_csv(tmp_path / "metrics.csv")
    out = tmp_path / "subdir" / "nested" / "curve.png"
    with patch("sys.argv", ["plot_results.py", "--results", str(csv), "--out", str(out)]):
        main()
    assert out.exists()
