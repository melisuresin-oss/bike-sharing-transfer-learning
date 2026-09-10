"""Independent PB1 R4 validator for verified snapshot provenance and results."""
from __future__ import annotations

import zipfile
from datetime import datetime
from pathlib import Path

from research.final_v2_2_r7_r1_pb1 import authorization
from research.final_v2_2_r7_r1_pb1 import staging
from research.final_v2_2_r7_r1_pb1 import validator as r3_validator

from . import core, scorer


def _verify_snapshot_records(stage_root: Path, authority_value: dict, provenance: dict) -> list[Path]:
    records = provenance.get("verified_snapshots")
    expected = authority_value.get("files")
    if not isinstance(records, list) or not isinstance(expected, list) or len(records) != 11 or len(expected) != 11:
        raise PermissionError("R4 verified-snapshot universe mismatch")
    if provenance.get("verified_snapshot_root_sha256") != core.canonical_hash(records):
        raise PermissionError("verified-snapshot provenance root mismatch")
    snapshot_root = core.no_symlink_path(stage_root / core.PB1_OUT / "verified_raw_snapshots",
                                         require_file=False)
    paths = []
    for registered, record in zip(expected, records):
        if (record.get("source_relative_path") != registered["path"] or
                record.get("registered_source_bytes") != registered["bytes"] or
                record.get("registered_source_sha256") != registered["sha256"] or
                record.get("role") != registered["role"] or
                record.get("verification_status") != "PASS_SAME_HANDLE_STREAMED_VERIFIED_SNAPSHOT"):
            raise PermissionError("snapshot/source authority mismatch")
        before, after = record.get("source_identity_before"), record.get("source_identity_after")
        if before != after or not isinstance(before, dict) or before.get("st_size") != registered["bytes"]:
            raise PermissionError("source identity changed during same-handle snapshot")
        snapshot = record.get("snapshot")
        path = core.verify_entry(stage_root, snapshot)
        path.relative_to(snapshot_root)
        if snapshot["bytes"] != registered["bytes"] or snapshot["sha256"] != registered["sha256"]:
            raise PermissionError("snapshot bytes differ from registered source authority")
        paths.append(path)
    relative_paths = [path.relative_to(stage_root).as_posix() for path in paths]
    if provenance.get("parsed_snapshot_paths") != relative_paths or provenance.get("original_raw_paths_parsed") is not False:
        raise PermissionError("materialization input provenance is not snapshot-only")
    if len(set(relative_paths)) != 11:
        raise PermissionError("duplicate verified snapshot")
    return paths


def _verify_materialization(capability: authorization.PhaseBCapability, stage_root: Path,
                            target_authority_path: Path):
    output_root = stage_root / core.PB1_OUT
    event_path = core.no_symlink_path(output_root / "first_label_access.json")
    provenance_path = core.no_symlink_path(output_root / "target_materialization_provenance.json")
    event = core.read_json(event_path); provenance = core.read_json(provenance_path)
    if event.get("schema_version") != core.ACCESS_SCHEMA or event.get("status") != "FIRST_HELD_OUT_BYTE_ACCESS_ABOUT_TO_BEGIN":
        raise PermissionError("first-access event mismatch")
    if event.get("authorization", {}).get("sha256") != capability.authorization_sha256 or event.get("raw_bytes_read_before_this_event") is not False:
        raise PermissionError("first-access authorization/order mismatch")
    if datetime.fromisoformat(event["event_utc"]) < datetime.fromisoformat(capability.authorization_created_utc):
        raise PermissionError("first access predates authorization")
    if (provenance.get("schema_version") != core.MATERIALIZATION_SCHEMA or
            provenance.get("status") != "PB1_R4_TARGETS_MATERIALIZED_FROM_VERIFIED_SNAPSHOTS"):
        raise PermissionError("R4 materialization provenance state mismatch")
    if provenance.get("authorization_sha256") != capability.authorization_sha256 or provenance.get("target_authority_sha256") != capability.target_authority_sha256:
        raise PermissionError("materialization authority mismatch")
    authority_value = core.read_json(core.no_symlink_path(target_authority_path))
    snapshots = _verify_snapshot_records(stage_root, authority_value, provenance)
    target_path = core.no_symlink_path(Path(provenance["materialized_targets"]["path"]))
    target = provenance["materialized_targets"]
    if core.sha256_file(target_path) != target["sha256"] or target_path.stat().st_size != target["bytes"]:
        raise PermissionError("materialized target changed")
    return target_path, provenance, snapshots


def validate_results(authorization_path: Path, stage_root: Path, raw_data_root: Path,
                     target_authority_path: Path, pb1_package_manifest: Path,
                     pb1_package_sha: str) -> dict:
    stage_root = core.no_symlink_path(stage_root, require_file=False)
    capability = authorization.validate(authorization_path, stage_root, raw_data_root,
                                        target_authority_path, pb1_package_manifest, pb1_package_sha)
    staged = staging.validate(stage_root)
    phase_a = r3_validator._verify_phase_a_commitment(stage_root)
    target_path, provenance, snapshots = _verify_materialization(capability, stage_root, target_authority_path)
    output_root = stage_root / core.PB1_OUT
    expected_metrics, expected_bootstrap, expected_comparisons, expected_replicates = scorer.compute_outputs(stage_root, target_path)
    expected = {
        "metrics.json": core.pretty_bytes(expected_metrics),
        "bootstrap_summary.json": core.pretty_bytes(expected_bootstrap),
        "paired_comparisons.json": core.pretty_bytes(expected_comparisons),
        "bootstrap_replicates.npz": core.deterministic_npz(expected_replicates),
    }
    for name, payload in expected.items():
        path = core.no_symlink_path(output_root / name)
        if core.sha256_file(path) != core.sha256_bytes(payload) or path.stat().st_size != len(payload):
            raise PermissionError("scoring output does not reproduce: " + name)
    metrics = core.read_json(output_root / "metrics.json")
    bootstrap = core.read_json(output_root / "bootstrap_summary.json")
    comparisons = core.read_json(output_root / "paired_comparisons.json")
    if metrics.get("evaluation_pass_count") != 468 or metrics.get("registered_reporting_cell_count") != 660:
        raise PermissionError("reporting count mismatch")
    if metrics.get("aggregated_city_cell_count") != 180 or len(metrics.get("equal_city_macro_cells", [])) != 45:
        raise PermissionError("city/macro reporting structure mismatch")
    if bootstrap.get("replicates") != 2000 or bootstrap.get("block_lengths") != [168] * 9 + [53]:
        raise PermissionError("bootstrap contract mismatch")
    if len(bootstrap.get("city_cells", [])) != 180 or len(bootstrap.get("equal_city_macro_cells", [])) != 45:
        raise PermissionError("bootstrap city/macro reporting structure mismatch")
    if bootstrap.get("undefined_replicates_redrawn") is not False or bootstrap.get("training_seed_randomness_inside_bootstrap") is not False:
        raise PermissionError("bootstrap firewall mismatch")
    if comparisons.get("gain_definition") != "error(reference)-error(method)" or comparisons.get("positive_favors_method") is not True:
        raise PermissionError("paired gain orientation mismatch")
    forbidden = [path for path in output_root.rglob("*") if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt"}]
    if forbidden:
        raise PermissionError("PB1 created an unauthorized checkpoint")
    return {
        "schema_version": core.VALIDATION_SCHEMA,
        "status": "PASS_PB1_R4_FINAL_RESULTS_PRE_FREEZE",
        "execution_id": capability.execution_id,
        "phase_a_files_verified": staged["phase_a_files"],
        "completions_verified": phase_a["completions"], "predictions_verified": phase_a["predictions"],
        "verified_snapshots": len(snapshots), "snapshot_only_parsing": True,
        "evaluation_passes": 468, "registered_reporting_cells": 660,
        "aggregated_city_cells": 180, "equal_city_macro_cells": 45,
        "bootstrap_replicates": 2000, "target_cities": list(core.TARGETS),
        "learned_seeds": list(core.FINAL_SEEDS), "authorization_predates_first_access": True,
        "phase_a_evidence_modified": False, "predictions_modified_after_commitment": False,
        "models_retrained_after_commitment": False, "new_learned_checkpoints": 0,
        "metrics_reproduced": True, "bootstrap_reproduced": True, "comparisons_reproduced": True,
    }


def validate_frozen(stage_root: Path) -> dict:
    stage_root = core.no_symlink_path(stage_root, require_file=False)
    output_root = stage_root / core.PB1_OUT
    manifest_path = core.no_symlink_path(output_root / "final_result_manifest.json")
    manifest = core.read_json(manifest_path)
    if manifest.get("schema_version") != core.FINAL_MANIFEST_SCHEMA or manifest.get("status") != "FINAL_V2_2_RESULTS_FROZEN":
        raise PermissionError("final-result manifest state mismatch")
    for item in manifest["bound_artifacts"].values():
        core.verify_entry(stage_root, item)
    provenance = core.read_json(stage_root / manifest["bound_artifacts"]["target_materialization_provenance"]["path"])
    authority_value = core.read_json(stage_root / manifest["bound_artifacts"]["target_authority"]["path"])
    snapshots = _verify_snapshot_records(stage_root, authority_value, provenance)
    if manifest.get("verified_snapshot_root_sha256") != provenance["verified_snapshot_root_sha256"]:
        raise PermissionError("final manifest does not bind verified snapshots")
    archive_entry = manifest["final_result_archive"]
    archive_path = core.verify_entry(stage_root, archive_entry)
    payload_manifest = core.read_json(stage_root / manifest["bound_artifacts"]["frozen_payload_manifest"]["path"])
    if (payload_manifest.get("schema_version") != core.PAYLOAD_MANIFEST_SCHEMA or
            payload_manifest.get("verified_snapshot_root_sha256") != provenance["verified_snapshot_root_sha256"]):
        raise PermissionError("frozen payload/snapshot binding mismatch")
    expected = {item["path"]: item for item in payload_manifest["files"]}
    with zipfile.ZipFile(archive_path) as archive:
        entries = [item for item in archive.infolist() if not item.is_dir()]
        if len(entries) != len(expected) or {item.filename for item in entries} != set(expected):
            raise PermissionError("final archive member-set mismatch")
        for item in entries:
            payload = archive.read(item); registered = expected[item.filename]
            if len(payload) != registered["bytes"] or core.sha256_bytes(payload) != registered["sha256"]:
                raise PermissionError("final archive member tampering: " + item.filename)
    return {"status": "PASS_PB1_R4_FINAL_RESULT_FROZEN", "archive": archive_entry,
            "manifest": core.entry(manifest_path, relative_to=stage_root),
            "verified_snapshots": len(snapshots), "phase_a_evidence_modified": False,
            "models_retrained_after_commitment": False}
