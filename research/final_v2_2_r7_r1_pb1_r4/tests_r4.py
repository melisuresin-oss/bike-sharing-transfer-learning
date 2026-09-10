"""Synthetic regression for the PB1 R4 verified-snapshot boundary."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research.final_v2_2_r7_r1_pb1 import authorization, staging

from . import core, materializer


class PB1R4TOCTOUTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _write(path: Path, payload: bytes) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def test_001_post_verification_source_replacement_cannot_affect_materialization(self):
        stage = self.root / "stage"
        raw = self.root / "raw"
        stage.mkdir()
        raw.mkdir()
        output_root = stage / core.PB1_OUT
        output_root.mkdir(parents=True)

        authority_files = []
        original_payloads = {}
        for index in range(11):
            role = "trips" if index < 6 else "station_status"
            relative = f"synthetic/{role}/shard_{index:02d}.parquet"
            payload = f"registered-original-{index:02d}".encode("ascii")
            source = self._write(raw / relative, payload)
            original_payloads[relative] = payload
            authority_files.append({
                "path": relative,
                "role": role,
                "bytes": len(payload),
                "sha256": core.sha256_file(source),
            })

        target_authority = stage / "target_authority.json"
        target_authority.write_text(json.dumps({"files": authority_files}), encoding="utf-8")
        package = stage / "package_manifest.json"
        package.write_text(json.dumps({
            "schema_version": core.REVISION + ".package.1",
            "status": "PB1_PACKAGE_FROZEN_NOT_EXECUTED",
            "package_member_manifest": {"sha256": "1" * 64},
            "executable_code_manifest": {"sha256": "2" * 64},
            "implementation_contract": {"sha256": "3" * 64},
        }), encoding="utf-8")
        (output_root / "staging_manifest.json").write_text(
            json.dumps({"phase_a_archive": {"sha256": "4" * 64}}), encoding="utf-8")
        authorization_path = output_root / "authorization.json"
        phase_a_binding = {"synthetic_phase_a": True}

        validation_patches = (
            patch.object(staging, "validate", return_value={"status": "PASS"}),
            patch.object(authorization.authority, "validate", return_value={"status": "PASS"}),
            patch.object(authorization, "_phase_a_bindings", return_value=phase_a_binding),
        )
        with validation_patches[0], validation_patches[1], validation_patches[2]:
            authorization.create(stage, raw, target_authority, package,
                                 core.sha256_file(package), authorization_path)

        real_verify = materializer.verify_raw_sources_after_authorization
        victim = raw / authority_files[0]["path"]
        replacement = b"post-verification-replacement"

        def verify_then_replace(*args, **kwargs):
            result = real_verify(*args, **kwargs)
            victim.write_bytes(replacement)
            return result

        def parse_only_snapshots(parse_stage, trips, status_files):
            snapshot_root = (parse_stage / core.PB1_OUT / "verified_raw_snapshots").resolve()
            inputs = trips + status_files
            self.assertEqual(11, len(inputs))
            for path in inputs:
                path.resolve().relative_to(snapshot_root)
                with self.assertRaises(ValueError):
                    path.resolve().relative_to(raw.resolve())
            self.assertEqual(original_payloads[authority_files[0]["path"]], inputs[0].read_bytes())
            self.assertNotEqual(replacement, inputs[0].read_bytes())
            return {
                "city_id": np.asarray(core.TARGETS, dtype=np.int64),
                "station_id": np.arange(4, dtype=np.int64),
                "forecast_origin_us": np.full(4, core.HT, dtype=np.int64),
                "opaque_label_join_key": np.arange(4, dtype=np.uint64),
                "count": np.zeros(4, dtype=np.int64),
                "label_valid": np.ones(4, dtype=bool),
            }

        with validation_patches[0], validation_patches[1], validation_patches[2], \
                patch.object(materializer, "verify_raw_sources_after_authorization",
                             side_effect=verify_then_replace), \
                patch.object(materializer, "_materialize_arrays", side_effect=parse_only_snapshots):
            result = materializer.materialize(
                authorization_path, stage, raw, target_authority, package,
                core.sha256_file(package))

        self.assertEqual("PASS_PB1_R4_TARGET_MATERIALIZATION", result["status"])
        self.assertEqual(replacement, victim.read_bytes())
        provenance = core.read_json(output_root / "target_materialization_provenance.json")
        self.assertIs(False, provenance["original_raw_paths_parsed"])
        self.assertEqual(11, len(provenance["verified_snapshots"]))
        self.assertEqual(core.canonical_hash(provenance["verified_snapshots"]),
                         provenance["verified_snapshot_root_sha256"])
        for expected, record in zip(authority_files, provenance["verified_snapshots"]):
            self.assertEqual(expected["sha256"], record["snapshot"]["sha256"])
            self.assertEqual(expected["bytes"], record["snapshot"]["bytes"])
            self.assertEqual(record["source_identity_before"], record["source_identity_after"])
            self.assertEqual("PASS_SAME_HANDLE_STREAMED_VERIFIED_SNAPSHOT",
                             record["verification_status"])


if __name__ == "__main__":
    unittest.main()
