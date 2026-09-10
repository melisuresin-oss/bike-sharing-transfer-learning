"""Focused synthetic tests for the PB1 R6 scoring-only correction."""
from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research.final_v2_2_r7_r1_pb1 import scorer as frozen

from . import core, scorer


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_MANIFEST = (
    ROOT / "research/results/final_v2_2_execution_seal/evaluation_manifest.json")


def _macro(values):
    return {(method, "0"): {"point": {"mae": value}}
            for method, value in values.items()}


def _synthetic_outputs(candidate_available: dict[str, bool]):
    plans = json.loads(EVALUATION_MANIFEST.read_text(encoding="utf-8"))["passes"]
    evaluations = []
    n = 24
    station = np.ones(n, dtype=np.int64)
    hour = np.arange(n, dtype=np.int64)
    labels = np.ones(n, dtype=np.float64)
    valid = np.ones(n, dtype=bool)
    for plan in plans:
        available = np.ones(n, dtype=bool)
        if plan["method"] in core.REGISTERED_ZERO_REFERENCES:
            available[:] = candidate_available[plan["method"]]
        evaluations.append(frozen.EvaluationData(
            plan["evaluation_id"], plan["target_city_id"], plan["method"],
            plan["budget"], plan["seed"], station, hour, labels, valid,
            np.ones(n, dtype=np.float64), available))
    weights = np.ones((2000, 10), dtype=np.float64)
    with patch.object(frozen, "_load_labels", return_value={}), \
            patch.object(frozen, "_load_evaluations",
                         return_value=(plans, evaluations)), \
            patch.object(frozen, "_draw_weights", return_value=weights):
        return scorer.compute_outputs(Path("unused"), Path("unused"))


class PB1R6ScoringFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.two_available = _synthetic_outputs({
            "HA_SOURCE": True,
            "persistence": False,
            "seasonal_naive": True,
        })
        cls.none_available = _synthetic_outputs({
            "HA_SOURCE": False,
            "persistence": False,
            "seasonal_naive": False,
        })

    def test_001_two_estimable_one_unavailable_succeeds(self):
        metrics = self.two_available[0]
        self.assertEqual("HA_SOURCE", metrics["selected_parameter_zero_reference"])
        self.assertEqual([
            {"method": "HA_SOURCE", "equal_city_macro_mae_estimable": True},
            {"method": "persistence", "equal_city_macro_mae_estimable": False},
            {"method": "seasonal_naive", "equal_city_macro_mae_estimable": True},
        ], metrics["registered_parameter_zero_candidates"])

    def test_002_lowest_then_lexicographic_exact_tie(self):
        _, selected = scorer._zero_reference_selection(_macro({
            "HA_SOURCE": 2.0, "persistence": None, "seasonal_naive": 1.0}))
        self.assertEqual("seasonal_naive", selected)
        _, tied = scorer._zero_reference_selection(_macro({
            "HA_SOURCE": 1.0, "persistence": None, "seasonal_naive": 1.0}))
        self.assertEqual("HA_SOURCE", tied)

    def test_003_one_estimable_candidate_selected(self):
        statuses, selected = scorer._zero_reference_selection(_macro({
            "HA_SOURCE": None, "persistence": 3.0, "seasonal_naive": None}))
        self.assertEqual("persistence", selected)
        self.assertEqual(1, sum(item["equal_city_macro_mae_estimable"]
                                for item in statuses))

    def test_004_all_unavailable_keeps_other_results(self):
        metrics, _, comparisons, _ = self.none_available
        self.assertIsNone(metrics["selected_parameter_zero_reference"])
        self.assertEqual(468, metrics["evaluation_pass_count"])
        source_rows = [row for row in metrics["aggregated_city_cells"]
                       if row["method"] in {
                           "ordinary_source_graphgru",
                           "grl_l50_constant_source_graphgru"} and
                       row["budget"] == "0"]
        self.assertEqual(8, len(source_rows))
        self.assertTrue(all(row["point"]["mae"] is not None for row in source_rows))
        nonzero = [row for row in comparisons["city_comparisons"]
                   if row["budget"] != "0"]
        self.assertEqual(96, len(nonzero))
        self.assertTrue(all(row["point_gain"]["mae"] is not None
                            for row in nonzero))
        incremental = [row for row in comparisons["city_comparisons"]
                       if row["family"] == "grl_zero_incremental"]
        self.assertEqual(4, len(incremental))
        self.assertTrue(all(row["point_gain"]["mae"] is not None
                            for row in incremental))
        unavailable = [row for row in comparisons["city_comparisons"]
                       if row["family"] in {
                           "ordinary_zero_transfer", "grl_zero_transfer"}]
        self.assertEqual(8, len(unavailable))
        self.assertTrue(all(row["point_gain"]["mae"] is None
                            for row in unavailable))
        self.assertIs(False, metrics["unknown_or_unavailable_filled_with_zero"])

    def test_005_unavailable_prediction_is_not_zero_imputed(self):
        label = np.array([5.0, 0.0])
        prediction = np.array([0.0, 0.0])
        station = np.array([1, 1], dtype=np.int64)
        hour = np.array([0, 1], dtype=np.int64)
        stats = frozen._sufficient(
            label, prediction, np.array([False, True]), station, hour)
        self.assertEqual(1, int(stats["valid"].sum()))
        self.assertFalse(bool(stats["valid"][0]))
        self.assertTrue(bool(stats["valid"][1]))

    def test_006_468_660_structure_unchanged(self):
        metrics, bootstrap, comparisons, _ = self.two_available
        self.assertEqual((468, 660, 180, 45), (
            metrics["evaluation_pass_count"],
            metrics["registered_reporting_cell_count"],
            metrics["aggregated_city_cell_count"],
            metrics["equal_city_macro_cell_count"]))
        self.assertEqual(180, len(bootstrap["city_cells"]))
        self.assertEqual(45, len(bootstrap["equal_city_macro_cells"]))
        self.assertEqual(108, len(comparisons["city_comparisons"]))
        self.assertEqual(27, len(comparisons["equal_city_macro_comparisons"]))

    def test_007_nonzero_comparison_specs_are_byte_for_byte_unchanged(self):
        expected = []
        for budget in ("1", "7", "30", "full"):
            expected.extend([
                ("graph_contribution", budget, "target_only_graph", "target_only_vanilla"),
                ("ordinary_transfer", budget, "ordinary_source_adapted", "target_only_graph"),
                ("adaptation", budget, "ordinary_source_adapted", "ordinary_source_graphgru"),
                ("pooled_vs_sequential", budget, "pooled_graph", "ordinary_source_adapted"),
                ("grl_incremental", budget, "grl_l50_constant_adapted", "ordinary_source_adapted"),
                ("grl_source_invariance", budget, "grl_l50_constant_source_graphgru", "ordinary_source_graphgru"),
            ])
        self.assertEqual(expected, scorer._comparison_specs("HA_SOURCE")[:24])

    def test_008_no_training_recovery_materialization_or_raw_reader_call(self):
        forbidden = {
            "recover", "materialize", "_materialize_arrays",
            "verify_raw_sources_after_authorization", "predict_all", "train", "fit",
        }
        package = Path(__file__).resolve().parent
        calls = []
        for path in package.glob("*.py"):
            if path.name == Path(__file__).name:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        calls.append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        calls.append(node.func.attr)
        self.assertEqual(set(), forbidden & set(calls))
        cli = (package / "cli.py").read_text(encoding="utf-8")
        self.assertNotIn('"recover"', cli)
        self.assertNotIn('"materialize"', cli)


if __name__ == "__main__":
    unittest.main()

