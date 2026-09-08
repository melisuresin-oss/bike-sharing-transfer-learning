from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for LOCAL_DEPS in (
    ROOT / "tmp" / "baseline_build" / "pydeps",
    ROOT / "tmp" / "protocol_build" / "pydeps",
):
    if LOCAL_DEPS.exists():
        sys.path.insert(0, str(LOCAL_DEPS))
import duckdb  # type: ignore  # noqa: E402


def qpath(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def load_builder_module():
    path = ROOT / "research" / "scripts" / "build_protocol_dataset.py"
    spec = importlib.util.spec_from_file_location("build_protocol_dataset", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load builder module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProtocolDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output = Path(
            os.environ.get(
                "PROTOCOL_DATASET_DIR",
                ROOT / "processed" / "protocol_v2_1",
            )
        ).resolve()
        if not cls.output.exists():
            raise unittest.SkipTest(f"Protocol output does not exist: {cls.output}")
        cls.con = duckdb.connect()
        cls.con.execute("SET TimeZone='UTC'")
        cls.manifest = json.loads(
            (cls.output / "PROTOCOL_DATASET_MANIFEST.json").read_text(
                encoding="utf-8"
            )
        )
        cls.builder = load_builder_module()
        cls.dev = cls.output / "development" / "development_panel.parquet"
        cls.adapt = (
            cls.output / "final_adaptation" / "final_adaptation_panel.parquet"
        )
        cls.features = (
            cls.output / "final_features" / "final_evaluation_features.parquet"
        )
        cls.keys = cls.output / "final_features" / "final_prediction_keys.parquet"
        cls.labels = (
            cls.output
            / "final_labels"
            / "SEALED_final_evaluation_labels.parquet"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.con.close()

    def scalar(self, query: str):
        return self.con.sql(query).fetchone()[0]

    def test_required_artifacts_exist(self) -> None:
        for path in (
            self.dev,
            self.adapt,
            self.features,
            self.keys,
            self.labels,
            self.output / "MODEL_READY_PANEL_AUDIT.md",
            self.output / "FEATURE_DATA_AUDIT.md",
            self.output / "artifact_hashes.json",
        ):
            self.assertTrue(path.is_file(), path)

    def test_unique_keys_in_every_panel_artifact(self) -> None:
        for path in (self.dev, self.adapt, self.features, self.keys, self.labels):
            duplicates = self.scalar(
                f"SELECT count(*) - count(DISTINCT (city_id,station_id,timestamp_utc)) "
                f"FROM read_parquet('{qpath(path)}')"
            )
            self.assertEqual(duplicates, 0, path.name)

    def test_frozen_station_counts(self) -> None:
        self.assertEqual(
            {int(k): int(v) for k, v in self.manifest["station_counts"].items()},
            self.builder.EXPECTED_STATIONS,
        )
        self.assertEqual(sum(self.manifest["station_counts"].values()), 799)

    def test_primary_target_null_iff_mask(self) -> None:
        for path in (self.dev, self.adapt, self.labels):
            bad = self.scalar(
                f"SELECT count(*) FROM read_parquet('{qpath(path)}') "
                "WHERE (target_12h IS NOT NULL) IS DISTINCT FROM coverage_observed_12h"
            )
            self.assertEqual(bad, 0, path.name)

    def test_coverage_masks_are_never_null(self) -> None:
        for path in (self.dev, self.adapt, self.labels):
            bad = self.scalar(
                f"SELECT count(*) FROM read_parquet('{qpath(path)}') WHERE "
                "coverage_observed_12h IS NULL OR coverage_observed_24h IS NULL "
                "OR coverage_observed_12h_no_maintenance IS NULL"
            )
            self.assertEqual(bad, 0, path.name)

    def test_unknown_positive_never_enters_primary_target(self) -> None:
        for path in (self.dev, self.adapt, self.labels):
            bad = self.scalar(
                f"SELECT count(*) FROM read_parquet('{qpath(path)}') "
                "WHERE raw_departure_count>0 AND NOT coverage_observed_12h "
                "AND target_12h IS NOT NULL"
            )
            self.assertEqual(bad, 0, path.name)

    def test_bracket_only_rule_has_no_same_hour_event_input(self) -> None:
        self.assertTrue(self.builder.primary_mask_rule(True, True, 5, 10))
        self.assertFalse(self.manifest["coverage"]["same_hour_event_required"])
        self.assertNotIn(
            "same_hour",
            self.builder.primary_mask_rule.__code__.co_varnames,
        )

    def test_primary_and_24h_masks_reproduce_from_diagnostics(self) -> None:
        for path in (self.dev, self.adapt):
            primary_bad = self.scalar(
                f"SELECT count(*) FROM read_parquet('{qpath(path)}') WHERE "
                "coverage_observed_12h IS DISTINCT FROM (station_in_status_lifetime "
                "AND station_status_bracketed_12h AND city_observable_12h)"
            )
            sensitivity_bad = self.scalar(
                f"SELECT count(*) FROM read_parquet('{qpath(path)}') WHERE "
                "coverage_observed_24h IS DISTINCT FROM (station_in_status_lifetime "
                "AND station_status_bracketed_24h AND city_observable_24h)"
            )
            self.assertEqual(primary_bad, 0, path.name)
            self.assertEqual(sensitivity_bad, 0, path.name)

    def test_no_maintenance_cohort_is_symmetric(self) -> None:
        for path in (self.dev, self.adapt, self.labels):
            bad = self.scalar(
                f"SELECT count(*) FROM read_parquet('{qpath(path)}') WHERE "
                "((target_12h_no_maintenance IS NOT NULL) IS DISTINCT FROM "
                "coverage_observed_12h_no_maintenance)"
            )
            self.assertEqual(bad, 0, path.name)

    def test_registered_temporal_boundaries_and_no_overlap(self) -> None:
        self.assertEqual(
            self.manifest["temporal_boundaries"],
            {
                "H0": "2022-08-29T05:00:00Z",
                "HD": "2023-02-15T22:00:00Z",
                "HF": "2023-04-16T22:00:00Z",
                "HT": "2023-05-11T15:00:00Z",
                "HE": "2023-07-15T20:00:00Z",
            },
        )
        dev_final_rows = self.scalar(
            f"SELECT count(*) FROM read_parquet('{qpath(self.dev)}') "
            "WHERE final_target OR timestamp_utc >= TIMESTAMPTZ '2023-04-16 22:00:00+00'"
        )
        self.assertEqual(dev_final_rows, 0)

    def test_budget_windows_are_nested(self) -> None:
        for path in (self.dev, self.adapt):
            bad = self.scalar(
                f"SELECT count(*) FROM read_parquet('{qpath(path)}') WHERE "
                "budget_1_day AND NOT budget_7_days "
                "OR budget_7_days AND NOT budget_30_days "
                "OR budget_30_days AND NOT budget_full"
            )
            self.assertEqual(bad, 0, path.name)

    def test_dynamic_features_are_causal_and_masked(self) -> None:
        feature_schema = self.manifest["feature_schema"]
        dynamic = [
            item for item in feature_schema if item["tensor"] in ("X_hist", "X_week")
        ]
        self.assertTrue(dynamic)
        self.assertTrue(all(item["offset_hours"] < 0 for item in dynamic))
        for path in (self.dev, self.adapt, self.features):
            pieces = [
                f"count_if(NOT hist_lag_{lag:03d}_observed "
                f"AND hist_lag_{lag:03d}_log1p != 0)"
                for lag in range(1, 25)
            ] + [
                "count_if(NOT week_lag_168_observed AND week_lag_168_log1p != 0)"
            ]
            bad = self.scalar(
                "SELECT " + "+".join(pieces) + f" FROM read_parquet('{qpath(path)}')"
            )
            self.assertEqual(bad, 0, path.name)

    def test_final_feature_schema_is_label_free(self) -> None:
        columns = {
            row[0]
            for row in self.con.sql(
                f"DESCRIBE SELECT * FROM read_parquet('{qpath(self.features)}')"
            ).fetchall()
        }
        forbidden = {
            "raw_departure_count",
            "target_12h",
            "target_24h",
            "target_12h_no_maintenance",
            "coverage_state_12h",
            "next_status_utc",
            "previous_maintenance",
        }
        self.assertFalse(columns & forbidden)
        registered = {item["column"] for item in self.manifest["feature_schema"]}
        self.assertTrue(registered <= columns)

    def test_prediction_keys_equal_primary_label_keys(self) -> None:
        mismatch = self.scalar(
            f"SELECT count(*) FROM ((SELECT * FROM read_parquet('{qpath(self.keys)}') "
            f"EXCEPT SELECT city_id,station_id,timestamp_utc FROM read_parquet('{qpath(self.labels)}') "
            "WHERE coverage_observed_12h) UNION ALL "
            f"(SELECT city_id,station_id,timestamp_utc FROM read_parquet('{qpath(self.labels)}') "
            f"WHERE coverage_observed_12h EXCEPT SELECT * FROM read_parquet('{qpath(self.keys)}')))"
        )
        self.assertEqual(mismatch, 0)

    def test_manifest_validations_have_no_failure(self) -> None:
        self.assertEqual(self.manifest["validation_summary"]["fail"], 0)
        self.assertGreater(self.manifest["validation_summary"]["pass"], 0)

    def test_independent_rebuild_is_reproducible(self) -> None:
        path = self.output / "reproducibility_check.json"
        self.assertTrue(path.is_file(), "Run compare_protocol_builds.py first")
        result = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["deterministic_artifact_hashes_identical"])
        self.assertTrue(result["protocol_manifests_identical_after_removing_required_build_timestamp"])


if __name__ == "__main__":
    unittest.main()
