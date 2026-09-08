from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ROOT = ROOT / "processed" / "protocol_v2_1"


def import_duckdb():
    try:
        import duckdb  # type: ignore

        return duckdb
    except ModuleNotFoundError:
        local_candidates = (
            ROOT / "tmp" / "neural_build" / "pydeps",
            ROOT / "tmp" / "baseline_build" / "pydeps",
            ROOT / "tmp" / "protocol_build" / "pydeps",
        )
        for local in local_candidates:
            if local.exists():
                sys.path.insert(0, str(local))
                import duckdb  # type: ignore

                return duckdb
        raise RuntimeError(
            "DuckDB is required. Install research/requirements.txt before use."
        )


duckdb = import_duckdb()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ProtocolArtifactRegistry:
    """Hash-verified allowlist for model-development inputs.

    The evaluation-label directory is intentionally absent from this class.
    Callers cannot supply an arbitrary Parquet path.
    """

    _ARTIFACTS = {
        "development": "development/development_panel.parquet",
        "final_adaptation": "final_adaptation/final_adaptation_panel.parquet",
        "final_features": "final_features/final_evaluation_features.parquet",
        "final_prediction_keys": "final_features/final_prediction_keys.parquet",
        "station_manifest_index": "station_manifests/station_manifest.json",
        "budget_manifest": "manifests/budget_manifest.json",
        "feature_manifest": "manifests/feature_manifest.json",
        "split_manifest": "manifests/split_manifest.json",
    }

    def __init__(self, protocol_root: Path = PROTOCOL_ROOT, verify_hashes: bool = True):
        self.protocol_root = protocol_root.resolve()
        self.manifest_path = self.protocol_root / "PROTOCOL_DATASET_MANIFEST.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.manifest: dict[str, Any] = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        if self.manifest.get("protocol_version") != "2.1":
            raise RuntimeError("Only protocol V2.1 is supported.")
        self._station_index = json.loads(
            self.path("station_manifest_index", verify=False).read_text(encoding="utf-8")
        )
        self._station_entries = {
            int(item["city_id"]): item for item in self._station_index["cities"]
        }
        if verify_hashes:
            self.verify_immutable_inputs()

    def path(self, artifact: str, verify: bool = True) -> Path:
        if artifact not in self._ARTIFACTS:
            raise KeyError(f"Artifact is not approved for model-development access: {artifact}")
        path = (self.protocol_root / self._ARTIFACTS[artifact]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if verify:
            expected = self.manifest.get("artifact_hashes", {}).get(
                self._ARTIFACTS[artifact], {}
            ).get("sha256")
            if expected and sha256_file(path) != expected:
                raise RuntimeError(f"Immutable protocol artifact hash mismatch: {path}")
        return path

    def station_manifest(self, city_id: int, verify: bool = True) -> Path:
        try:
            entry = self._station_entries[int(city_id)]
        except KeyError as exc:
            raise KeyError(f"City {city_id} is not in the frozen station manifest") from exc
        path = (self.protocol_root / entry["artifact"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if verify:
            expected = self.manifest["artifact_hashes"][entry["artifact"]]["sha256"]
            if sha256_file(path) != expected:
                raise RuntimeError(f"Station manifest hash mismatch: {path}")
        return path

    def station_manifest_entry(self, city_id: int) -> dict[str, Any]:
        """Return immutable graph/roster metadata without exposing arbitrary paths."""
        try:
            return dict(self._station_entries[int(city_id)])
        except KeyError as exc:
            raise KeyError(f"City {city_id} is not in the frozen station manifest") from exc

    def verify_immutable_inputs(self) -> None:
        for artifact in self._ARTIFACTS:
            self.path(artifact)
        for city_id in self._station_entries:
            self.station_manifest(city_id)

    @property
    def city_roles(self) -> dict[int, dict[str, Any]]:
        return {
            int(row["city_id"]): row for row in self.manifest["city_roles"]
        }

    @property
    def feature_columns(self) -> list[str]:
        return [item["column"] for item in self.manifest["feature_schema"]]

    @property
    def boundaries(self) -> dict[str, str]:
        return dict(self.manifest["temporal_boundaries"])

    @property
    def expected_budget_counts(self) -> dict[tuple[int, str], int]:
        return {
            (int(row["city_id"]), row["budget"]): int(row["eligible_target_labels"])
            for row in self.manifest["actual_budget_label_counts"]
        }

    def panel_for_city(self, city_id: int) -> Path:
        role = self.city_roles[int(city_id)]
        if role["final_target"]:
            return self.path("final_adaptation")
        if role["development_source"]:
            return self.path("development")
        raise KeyError(f"City {city_id} has no approved label-bearing panel")

    def connect(self):
        con = duckdb.connect()
        # Parallel floating-point reductions can differ in their final bits because
        # partial aggregates are merged in scheduler-dependent order.  Baseline
        # artifacts are protocol evidence, so use one execution thread to make
        # aggregate values and Parquet bytes independently reproducible.
        con.execute("SET threads=1")
        con.execute("SET TimeZone='UTC'")
        con.execute("SET preserve_insertion_order=false")
        return con


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")
