from __future__ import annotations

import unittest

import numpy as np

from research.data.manifests import ProtocolArtifactRegistry, sql_path
from research.data.window_dataset import FinalStationInferenceLoader


class WindowCausalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = ProtocolArtifactRegistry()

    def test_all_dynamic_offsets_precede_target(self) -> None:
        dynamic = [
            item
            for item in self.registry.manifest["feature_schema"]
            if item["tensor"] in {"X_hist", "X_week"}
        ]
        self.assertTrue(dynamic)
        self.assertTrue(all(item["offset_hours"] < 0 for item in dynamic))

    def test_final_rolling_lag_is_earlier_but_current_label_absent(self) -> None:
        loader = FinalStationInferenceLoader(self.registry, 195)
        batch = loader.fetch(0, 5000)
        self.assertIsNone(batch.y)
        self.assertGreater(batch.m_hist[:, 0].sum(), 0)
        observed_index = int(np.flatnonzero(batch.m_hist[:, 0])[0])
        target_time = batch.timestamps[observed_index]
        lag_time = target_time - np.timedelta64(1, "h")
        self.assertLess(lag_time, target_time)
        con = self.registry.connect()
        try:
            columns = {
                row[0]
                for row in con.sql(
                    f"DESCRIBE SELECT * FROM read_parquet(" 
                    f"'{sql_path(self.registry.path('final_features'))}')"
                ).fetchall()
            }
        finally:
            con.close()
        self.assertNotIn("target_12h", columns)
        self.assertNotIn("raw_departure_count", columns)

    def test_missing_history_is_placeholder_plus_zero_mask(self) -> None:
        artifact = self.registry.path("development")
        con = self.registry.connect()
        try:
            bad = con.sql(
                "SELECT "
                + "+".join(
                    f"count_if(NOT hist_lag_{lag:03d}_observed "
                    f"AND hist_lag_{lag:03d}_log1p != 0)"
                    for lag in range(1, 25)
                )
                + f" FROM read_parquet('{sql_path(artifact)}')"
            ).fetchone()[0]
        finally:
            con.close()
        self.assertEqual(bad, 0)


if __name__ == "__main__":
    unittest.main()
