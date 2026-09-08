from __future__ import annotations

import unittest
from pathlib import Path

from research.data.manifests import ProtocolArtifactRegistry, sql_path
from research.data.window_dataset import FinalStationInferenceLoader


ROOT = Path(__file__).resolve().parents[2]


class FinalLabelSealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = ProtocolArtifactRegistry()

    def test_registry_has_no_final_label_artifact(self) -> None:
        with self.assertRaises(KeyError):
            self.registry.path("final_labels")

    def test_model_development_modules_do_not_name_sealed_directory(self) -> None:
        for directory in ("data", "baselines", "evaluation"):
            for path in (ROOT / "research" / directory).glob("*.py"):
                self.assertNotIn("final_labels", path.read_text(encoding="utf-8"), path)

    def test_final_inference_schema_is_label_free(self) -> None:
        loader = FinalStationInferenceLoader(self.registry, 237)
        self.assertIsNone(loader.fetch(0, 2).y)
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
        self.assertFalse(
            columns.intersection(
                {"target_12h", "target_24h", "raw_departure_count", "coverage_state_12h"}
            )
        )


if __name__ == "__main__":
    unittest.main()
