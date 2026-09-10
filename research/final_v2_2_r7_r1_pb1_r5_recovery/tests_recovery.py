"""Synthetic interrupted-R4 recovery regression."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research.final_v2_2_r7_r1_pb1 import authorization, staging
from research.final_v2_2_r7_r1_pb1_r4 import materializer as r4_materializer

from . import core, recovery


class PB1R5RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _json(path: Path, value: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_001_resume_uses_existing_snapshots_once_and_never_reopens_sources(self):
        stage = self.root / "stage"
        raw = self.root / "raw"
        stage.mkdir()
        raw.mkdir()
        output = stage / core.PB1_OUT
        output.mkdir(parents=True)
        authority_files = []
        snapshot_payloads = []
        for index in range(11):
            role = "station_status" if index < 5 else "trips"
            payload = f"verified-snapshot-{index:02d}".encode("ascii")
            snapshot_payloads.append(payload)
            authority_files.append({
                "path": f"synthetic/{role}/shard_{index:02d}.parquet",
                "role": role,
                "bytes": len(payload),
                "sha256": core.sha256_bytes(payload),
            })
        target_authority = self._json(stage / "target_authority.json",
                                      {"files": authority_files})
        staging_manifest = self._json(output / "staging_manifest.json",
                                      {"phase_a_archive": {"sha256": "4" * 64}})

        r4_package = self._json(stage / core.R4_PACKAGE_RELATIVE, {
            "schema_version": core.REVISION + ".package.1",
            "status": "PB1_PACKAGE_FROZEN_NOT_EXECUTED",
            "revision": core.R4_REVISION,
            "package_member_manifest": {"sha256": "1" * 64},
            "executable_code_manifest": {"sha256": "2" * 64},
            "implementation_contract": {"sha256": "3" * 64},
        })
        dependency_entries = {}
        for name in ("member", "code", "implementation"):
            path = stage / f"recovery/{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode("ascii"))
            dependency_entries[name] = core.entry(path, relative_to=stage)
        recovery_package = self._json(stage / "recovery/package.json", {
            "schema_version": core.RECOVERY_REVISION + ".package.1",
            "status": "PB1_R5_RECOVERY_PACKAGE_FROZEN_NOT_EXECUTED",
            "original_r4_package_sha256": core.sha256_file(r4_package),
            "package_member_manifest": dependency_entries["member"],
            "executable_code_manifest": dependency_entries["code"],
            "implementation_contract": dependency_entries["implementation"],
        })
        auth_path = output / "authorization.json"
        phase_a_binding = {"synthetic_phase_a": True}
        validation_patches = (
            patch.object(staging, "validate", return_value={"status": "PASS", "phase_a_files": 4167}),
            patch.object(authorization.authority, "validate", return_value={"status": "PASS"}),
            patch.object(authorization, "_phase_a_bindings", return_value=phase_a_binding),
        )
        with validation_patches[0], validation_patches[1], validation_patches[2]:
            authorization.create(stage, raw, target_authority, r4_package,
                                 core.sha256_file(r4_package), auth_path)
        auth_value = core.read_json(auth_path)
        first_access = self._json(output / "first_label_access.json", {
            "schema_version": core.ACCESS_SCHEMA,
            "status": "FIRST_HELD_OUT_BYTE_ACCESS_ABOUT_TO_BEGIN",
            "execution_id": auth_value["execution_id"],
            "authorization": core.entry(auth_path, relative_to=stage),
            "target_authority": core.entry(target_authority, relative_to=stage),
            "raw_data_root_canonical_path": str(raw),
            "event_utc": core.utc(),
            "next_operation": "same-handle verified stage-local snapshot creation",
            "raw_bytes_read_before_this_event": False,
        })
        snapshot_root = output / "verified_raw_snapshots"
        snapshot_root.mkdir()
        for index, (registered, payload) in enumerate(zip(authority_files, snapshot_payloads)):
            relative = r4_materializer._snapshot_relative(index, registered)
            path = stage / relative
            path.write_bytes(payload)

        evidence = {
            "authorization": core.sha256_file(auth_path),
            "first_access": core.sha256_file(first_access),
            "snapshots": {path.name: core.sha256_file(path) for path in snapshot_root.iterdir()},
        }
        self.assertEqual([], list(raw.rglob("*")))

        def synthetic_parser(parse_stage, trips, status_files):
            self.assertEqual(6, len(trips))
            self.assertEqual(5, len(status_files))
            for path in trips + status_files:
                path.resolve().relative_to(snapshot_root.resolve())
                self.assertTrue(path.read_bytes().startswith(b"verified-snapshot-"))
            self.assertEqual([], list(raw.rglob("*")))
            return {
                "city_id": np.asarray(core.TARGETS, dtype=np.int64),
                "station_id": np.arange(4, dtype=np.int64),
                "forecast_origin_us": np.full(4, core.HT, dtype=np.int64),
                "opaque_label_join_key": np.arange(4, dtype=np.uint64),
                "count": np.zeros(4, dtype=np.int64),
                "label_valid": np.ones(4, dtype=bool),
            }

        constant_patches = (
            patch.object(core, "EXPECTED_AUTHORIZATION_SHA256", core.sha256_file(auth_path)),
            patch.object(core, "EXPECTED_EXECUTION_ID", auth_value["execution_id"]),
            patch.object(core, "EXPECTED_STAGING_MANIFEST_SHA256", core.sha256_file(staging_manifest)),
            patch.object(core, "EXPECTED_FIRST_ACCESS_SHA256", core.sha256_file(first_access)),
            patch.object(core, "EXPECTED_TARGET_AUTHORITY_SHA256", core.sha256_file(target_authority)),
            patch.object(core, "EXPECTED_R4_PACKAGE_SHA256", core.sha256_file(r4_package)),
        )
        with validation_patches[0], validation_patches[1], validation_patches[2], \
                constant_patches[0], constant_patches[1], constant_patches[2], \
                constant_patches[3], constant_patches[4], constant_patches[5], \
                patch.object(r4_materializer, "verify_raw_sources_after_authorization",
                             side_effect=AssertionError("raw verifier must never run")), \
                patch.object(r4_materializer, "_materialize_arrays",
                             side_effect=synthetic_parser):
            result = recovery.recover(
                auth_path, stage, raw, target_authority, r4_package,
                core.sha256_file(r4_package), recovery_package,
                core.sha256_file(recovery_package))
            with self.assertRaises(FileExistsError):
                recovery.recover(
                    auth_path, stage, raw, target_authority, r4_package,
                    core.sha256_file(r4_package), recovery_package,
                    core.sha256_file(recovery_package))

        self.assertEqual("PASS_PB1_R5_OPERATIONAL_RECOVERY", result["status"])
        self.assertTrue((output / "materialized_targets.npz").is_file())
        self.assertTrue((output / "target_materialization_provenance.json").is_file())
        self.assertTrue((output / "recovery_event.json").is_file())
        self.assertTrue((output / "recovery_provenance.json").is_file())
        self.assertEqual(evidence["authorization"], core.sha256_file(auth_path))
        self.assertEqual(evidence["first_access"], core.sha256_file(first_access))
        self.assertEqual(evidence["snapshots"],
                         {path.name: core.sha256_file(path) for path in snapshot_root.iterdir()})
        provenance = core.read_json(output / "target_materialization_provenance.json")
        self.assertIs(False, provenance["original_raw_sources_reopened_during_recovery"])
        self.assertEqual(11, len(provenance["verified_snapshots"]))
        self.assertEqual([], list(raw.rglob("*")))


if __name__ == "__main__":
    unittest.main()
