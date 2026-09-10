"""Strict validator for the append-only PB1 R5 recovery path."""
from __future__ import annotations

import zipfile
from datetime import datetime
from pathlib import Path

from research.final_v2_2_r7_r1_pb1 import staging
from research.final_v2_2_r7_r1_pb1 import validator as r3_validator
from research.final_v2_2_r7_r1_pb1_r4 import materializer as r4_materializer

from . import core, recovery, scorer


def _verify_snapshot_records(stage_root: Path, authority_value: dict,
                             provenance: dict) -> list[Path]:
    records = provenance.get("verified_snapshots")
    expected = authority_value.get("files")
    if (not isinstance(records, list) or not isinstance(expected, list) or
            len(records) != 11 or len(expected) != 11):
        raise PermissionError("recovery snapshot universe mismatch")
    if provenance.get("verified_snapshot_root_sha256") != core.canonical_hash(records):
        raise PermissionError("recovery snapshot-record root mismatch")
    snapshot_root = core.no_symlink_path(
        stage_root / core.PB1_OUT / "verified_raw_snapshots", require_file=False)
    children = list(snapshot_root.iterdir())
    if len(children) != 11 or any(path.is_symlink() or not path.is_file() for path in children):
        raise PermissionError("recovery snapshot directory mismatch")
    paths = []
    trips, status = [], []
    for index, (registered, record) in enumerate(zip(expected, records)):
        expected_relative = r4_materializer._snapshot_relative(index, registered).as_posix()
        if (record.get("source_relative_path") != registered["path"] or
                record.get("registered_source_bytes") != registered["bytes"] or
                record.get("registered_source_sha256") != registered["sha256"] or
                record.get("role") != registered["role"] or
                record.get("deterministic_snapshot_path") != expected_relative or
                record.get("verification_status") !=
                "PASS_EXISTING_R4_SNAPSHOT_REVERIFIED_FOR_RECOVERY" or
                record.get("original_raw_source_opened") is not False):
            raise PermissionError("recovery snapshot/source authority mismatch")
        snapshot = record.get("snapshot")
        path = core.verify_entry(stage_root, snapshot)
        path.relative_to(snapshot_root)
        if snapshot["bytes"] != registered["bytes"] or snapshot["sha256"] != registered["sha256"]:
            raise PermissionError("recovery snapshot bytes differ from authority")
        if record.get("snapshot_identity_at_recovery") != recovery._identity(path.stat()):
            raise PermissionError("snapshot identity changed after recovery parsing")
        paths.append(path)
        (trips if registered["role"] == "trips" else status).append(path)
    relative_paths = [path.relative_to(stage_root).as_posix() for path in trips + status]
    if (provenance.get("parsed_snapshot_paths") != relative_paths or
            provenance.get("original_raw_paths_parsed") is not False or
            provenance.get("original_raw_sources_reopened_during_recovery") is not False):
        raise PermissionError("recovery parser input provenance mismatch")
    if len(set(path.relative_to(stage_root).as_posix() for path in paths)) != 11:
        raise PermissionError("duplicate recovery snapshot")
    if {path.name for path in children} != {path.name for path in paths}:
        raise PermissionError("unregistered recovery snapshot")
    return paths


def _verify_materialization(stage_root: Path, context: dict):
    output_root = stage_root / core.PB1_OUT
    event_path = core.no_symlink_path(output_root / "recovery_event.json")
    materialization_path = core.no_symlink_path(
        output_root / "target_materialization_provenance.json")
    recovery_path = core.no_symlink_path(output_root / "recovery_provenance.json")
    event = core.read_json(event_path)
    provenance = core.read_json(materialization_path)
    recovery_value = core.read_json(recovery_path)
    capability = context["capability"]
    if (event.get("schema_version") != core.RECOVERY_EVENT_SCHEMA or
            event.get("status") != "PB1_R5_RECOVERY_ABOUT_TO_PARSE_VERIFIED_R4_SNAPSHOTS" or
            event.get("reason") != core.RECOVERY_REASON or
            event.get("execution_id") != capability.execution_id or
            event.get("authorization_sha256") != capability.authorization_sha256 or
            event.get("first_label_access") != context["first_access"] or
            event.get("original_r4_package") != context["r4_package"] or
            event.get("recovery_package") != context["recovery_package"] or
            event.get("original_raw_sources_reopened_during_recovery") is not False):
        raise PermissionError("recovery event binding mismatch")
    if (provenance.get("schema_version") != core.MATERIALIZATION_SCHEMA or
            provenance.get("status") != "PB1_R5_TARGETS_RECOVERED_FROM_EXISTING_R4_SNAPSHOTS" or
            provenance.get("reason") != core.RECOVERY_REASON or
            provenance.get("execution_id") != capability.execution_id or
            provenance.get("authorization_sha256") != capability.authorization_sha256 or
            provenance.get("target_authority_sha256") != capability.target_authority_sha256 or
            provenance.get("recovery_event", {}).get("sha256") != core.sha256_file(event_path)):
        raise PermissionError("recovery materialization binding mismatch")
    authority_value = core.read_json(
        stage_root / context["target_authority"]["path"])
    snapshots = _verify_snapshot_records(stage_root, authority_value, provenance)
    if event.get("verified_snapshot_root_sha256") != provenance["verified_snapshot_root_sha256"]:
        raise PermissionError("recovery event snapshot root mismatch")
    target = provenance["materialized_targets"]
    target_path = core.no_symlink_path(Path(target["path"]))
    if target_path.stat().st_size != target["bytes"] or core.sha256_file(target_path) != target["sha256"]:
        raise PermissionError("recovered target artifact changed")
    if (recovery_value.get("schema_version") != core.RECOVERY_PROVENANCE_SCHEMA or
            recovery_value.get("status") != "PB1_R5_OPERATIONAL_RECOVERY_COMPLETE" or
            recovery_value.get("reason") != core.RECOVERY_REASON or
            recovery_value.get("authorization_sha256") != capability.authorization_sha256 or
            recovery_value.get("materialized_targets") != target or
            recovery_value.get("materialization_provenance", {}).get("sha256") !=
            core.sha256_file(materialization_path) or
            recovery_value.get("verified_snapshot_root_sha256") !=
            provenance["verified_snapshot_root_sha256"] or
            recovery_value.get("original_raw_sources_reopened_during_recovery") is not False or
            recovery_value.get("scientific_semantics_changed") is not False):
        raise PermissionError("recovery completion provenance mismatch")
    return target_path, provenance, snapshots


def validate_results(authorization_path: Path, stage_root: Path,
                     raw_data_root: Path, target_authority_path: Path,
                     r4_package_manifest: Path, r4_package_sha256: str,
                     recovery_package_manifest: Path,
                     recovery_package_sha256: str) -> dict:
    context = recovery.validate_recovery_context(
        authorization_path, stage_root, raw_data_root, target_authority_path,
        r4_package_manifest, r4_package_sha256, recovery_package_manifest,
        recovery_package_sha256)
    stage_root = context["stage_root"]
    staged = staging.validate(stage_root)
    phase_a = r3_validator._verify_phase_a_commitment(stage_root)
    target_path, provenance, snapshots = _verify_materialization(stage_root, context)
    output_root = stage_root / core.PB1_OUT
    expected_metrics, expected_bootstrap, expected_comparisons, expected_replicates = \
        scorer.compute_outputs(stage_root, target_path)
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
        raise PermissionError("bootstrap reporting structure mismatch")
    if bootstrap.get("undefined_replicates_redrawn") is not False or bootstrap.get("training_seed_randomness_inside_bootstrap") is not False:
        raise PermissionError("bootstrap firewall mismatch")
    if comparisons.get("gain_definition") != "error(reference)-error(method)" or comparisons.get("positive_favors_method") is not True:
        raise PermissionError("paired gain orientation mismatch")
    forbidden = [path for path in output_root.rglob("*") if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt"}]
    if forbidden:
        raise PermissionError("recovery created an unauthorized checkpoint")
    return {
        "schema_version": core.VALIDATION_SCHEMA,
        "status": "PASS_PB1_R5_RECOVERY_FINAL_RESULTS_PRE_FREEZE",
        "execution_id": context["capability"].execution_id,
        "phase_a_files_verified": staged["phase_a_files"],
        "completions_verified": phase_a["completions"],
        "predictions_verified": phase_a["predictions"],
        "verified_snapshots": len(snapshots),
        "snapshot_only_parsing": True,
        "evaluation_passes": 468,
        "registered_reporting_cells": 660,
        "aggregated_city_cells": 180,
        "equal_city_macro_cells": 45,
        "bootstrap_replicates": 2000,
        "target_cities": list(core.TARGETS),
        "learned_seeds": list(core.FINAL_SEEDS),
        "authorization_predates_first_access": True,
        "original_raw_sources_reopened_during_recovery": False,
        "phase_a_evidence_modified": False,
        "predictions_modified_after_commitment": False,
        "models_retrained_after_commitment": False,
        "new_learned_checkpoints": 0,
        "metrics_reproduced": True,
        "bootstrap_reproduced": True,
        "comparisons_reproduced": True,
    }


def validate_frozen(stage_root: Path) -> dict:
    stage_root = core.no_symlink_path(stage_root, require_file=False)
    output_root = stage_root / core.PB1_OUT
    manifest_path = core.no_symlink_path(output_root / "final_result_manifest.json")
    manifest = core.read_json(manifest_path)
    if (manifest.get("schema_version") != core.FINAL_MANIFEST_SCHEMA or
            manifest.get("status") != "FINAL_V2_2_RESULTS_FROZEN" or
            manifest.get("recovery_revision") != core.RECOVERY_REVISION):
        raise PermissionError("recovery final-result manifest state mismatch")
    for item in manifest["bound_artifacts"].values():
        core.verify_entry(stage_root, item)
    provenance = core.read_json(
        stage_root / manifest["bound_artifacts"]["target_materialization_provenance"]["path"])
    authority_value = core.read_json(
        stage_root / manifest["bound_artifacts"]["target_authority"]["path"])
    snapshots = _verify_snapshot_records(stage_root, authority_value, provenance)
    if manifest.get("verified_snapshot_root_sha256") != provenance["verified_snapshot_root_sha256"]:
        raise PermissionError("final manifest does not bind recovery snapshots")
    recovery_value = core.read_json(
        stage_root / manifest["bound_artifacts"]["recovery_provenance"]["path"])
    if recovery_value.get("verified_snapshot_root_sha256") != provenance["verified_snapshot_root_sha256"]:
        raise PermissionError("final manifest recovery provenance mismatch")
    archive_entry = manifest["final_result_archive"]
    archive_path = core.verify_entry(stage_root, archive_entry)
    payload_manifest = core.read_json(
        stage_root / manifest["bound_artifacts"]["frozen_payload_manifest"]["path"])
    if (payload_manifest.get("verified_snapshot_root_sha256") != provenance["verified_snapshot_root_sha256"] or
            payload_manifest.get("recovery_revision") != core.RECOVERY_REVISION):
        raise PermissionError("frozen payload/recovery snapshot binding mismatch")
    expected = {item["path"]: item for item in payload_manifest["files"]}
    with zipfile.ZipFile(archive_path) as archive:
        entries = [item for item in archive.infolist() if not item.is_dir()]
        if len(entries) != len(expected) or {item.filename for item in entries} != set(expected):
            raise PermissionError("final archive member-set mismatch")
        for item in entries:
            payload = archive.read(item)
            registered = expected[item.filename]
            if len(payload) != registered["bytes"] or core.sha256_bytes(payload) != registered["sha256"]:
                raise PermissionError("final archive member tampering: " + item.filename)
    return {
        "status": "PASS_PB1_R5_RECOVERY_FINAL_RESULT_FROZEN",
        "archive": archive_entry,
        "manifest": core.entry(manifest_path, relative_to=stage_root),
        "verified_snapshots": len(snapshots),
        "original_raw_sources_reopened_during_recovery": False,
        "phase_a_evidence_modified": False,
        "models_retrained_after_commitment": False,
    }
