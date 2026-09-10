"""Resume one interrupted R4 materialization from verified snapshots only."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from research.final_v2_2_r7_r1_pb1 import authorization
from research.final_v2_2_r7_r1_pb1_r4 import materializer as r4_materializer

from . import core


def _identity(stat_result) -> dict:
    return {
        "st_dev": int(stat_result.st_dev),
        "st_ino": int(stat_result.st_ino),
        "st_size": int(stat_result.st_size),
        "st_mtime_ns": int(stat_result.st_mtime_ns),
        "st_ctime_ns": int(stat_result.st_ctime_ns),
        "st_mode": int(stat_result.st_mode),
    }


def _verify_original_r4_package(stage_root: Path, path: Path, supplied_sha256: str) -> dict:
    path = core.no_symlink_path(path)
    expected_path = core.no_symlink_path(stage_root / core.R4_PACKAGE_RELATIVE)
    if path != expected_path:
        raise PermissionError("recovery requires the staged original R4 package manifest")
    digest = core.sha256_file(path)
    if digest != core.EXPECTED_R4_PACKAGE_SHA256 or supplied_sha256 != digest:
        raise PermissionError("original R4 package identity mismatch")
    value = core.read_json(path)
    if (value.get("revision") != core.R4_REVISION or
            value.get("status") != "PB1_PACKAGE_FROZEN_NOT_EXECUTED"):
        raise PermissionError("original R4 package state mismatch")
    return core.entry(path, relative_to=stage_root)


def _verify_recovery_package(stage_root: Path, path: Path, supplied_sha256: str) -> dict:
    path = core.no_symlink_path(path)
    digest = core.sha256_file(path)
    if digest != supplied_sha256:
        raise PermissionError("recovery package manifest hash mismatch")
    value = core.read_json(path)
    if (value.get("schema_version") != core.RECOVERY_REVISION + ".package.1" or
            value.get("status") != "PB1_R5_RECOVERY_PACKAGE_FROZEN_NOT_EXECUTED" or
            value.get("original_r4_package_sha256") != core.EXPECTED_R4_PACKAGE_SHA256):
        raise PermissionError("recovery package state mismatch")
    for name in ("package_member_manifest", "executable_code_manifest", "implementation_contract"):
        core.verify_entry(stage_root, value[name])
    return core.entry(path, relative_to=stage_root)


def _verify_first_access(stage_root: Path, capability: authorization.PhaseBCapability) -> dict:
    path = core.no_symlink_path(stage_root / core.PB1_OUT / "first_label_access.json")
    digest = core.sha256_file(path)
    if digest != core.EXPECTED_FIRST_ACCESS_SHA256:
        raise PermissionError("existing R4 first-access artifact identity mismatch")
    value = core.read_json(path)
    if (value.get("schema_version") != core.ACCESS_SCHEMA or
            value.get("status") != "FIRST_HELD_OUT_BYTE_ACCESS_ABOUT_TO_BEGIN" or
            value.get("execution_id") != capability.execution_id or
            value.get("authorization", {}).get("sha256") != capability.authorization_sha256 or
            value.get("target_authority", {}).get("sha256") != capability.target_authority_sha256 or
            value.get("raw_bytes_read_before_this_event") is not False):
        raise PermissionError("existing R4 first-access event mismatch")
    return core.entry(path, relative_to=stage_root)


def verify_existing_snapshots(stage_root: Path, authority_value: dict):
    snapshot_root = core.no_symlink_path(
        stage_root / core.PB1_OUT / "verified_raw_snapshots", require_file=False)
    children = list(snapshot_root.iterdir())
    if len(children) != 11 or any(path.is_symlink() or not path.is_file() for path in children):
        raise PermissionError("recovery requires exactly eleven regular snapshot files")
    expected_paths = []
    records = []
    trips, status = [], []
    for index, registered in enumerate(authority_value.get("files", [])):
        relative = r4_materializer._snapshot_relative(index, registered)
        path = core.no_symlink_path(stage_root / relative)
        path.relative_to(snapshot_root)
        expected_paths.append(path)
        stat_result = path.stat()
        digest = core.sha256_file(path)
        if stat_result.st_size != registered["bytes"] or digest != registered["sha256"]:
            raise PermissionError("existing R4 snapshot differs from registered authority")
        snapshot = core.entry(path, relative_to=stage_root)
        record = {
            "source_relative_path": registered["path"],
            "registered_source_bytes": registered["bytes"],
            "registered_source_sha256": registered["sha256"],
            "role": registered["role"],
            "deterministic_snapshot_path": relative.as_posix(),
            "snapshot": snapshot,
            "snapshot_identity_at_recovery": _identity(stat_result),
            "verification_status": "PASS_EXISTING_R4_SNAPSHOT_REVERIFIED_FOR_RECOVERY",
            "original_raw_source_opened": False,
        }
        records.append(record)
        (trips if registered["role"] == "trips" else status).append(path)
    if len(records) != 11 or len(trips) != 6 or len(status) != 5:
        raise PermissionError("snapshot authority role/count mismatch")
    if {path.name for path in children} != {path.name for path in expected_paths}:
        raise PermissionError("unregistered or missing recovery snapshot")
    return trips, status, records


def validate_recovery_context(authorization_path: Path, stage_root: Path,
                              raw_data_root: Path, target_authority_path: Path,
                              r4_package_manifest: Path, r4_package_sha256: str,
                              recovery_package_manifest: Path,
                              recovery_package_sha256: str) -> dict:
    stage_root = core.no_symlink_path(stage_root, require_file=False)
    capability = authorization.validate(
        authorization_path, stage_root, raw_data_root, target_authority_path,
        r4_package_manifest, r4_package_sha256)
    if (capability.authorization_sha256 != core.EXPECTED_AUTHORIZATION_SHA256 or
            capability.execution_id != core.EXPECTED_EXECUTION_ID):
        raise PermissionError("recovery authorization/execution identity mismatch")
    staging_path = core.no_symlink_path(stage_root / core.PB1_OUT / "staging_manifest.json")
    if core.sha256_file(staging_path) != core.EXPECTED_STAGING_MANIFEST_SHA256:
        raise PermissionError("recovery staging manifest identity mismatch")
    r4_package = _verify_original_r4_package(
        stage_root, r4_package_manifest, r4_package_sha256)
    recovery_package = _verify_recovery_package(
        stage_root, recovery_package_manifest, recovery_package_sha256)
    authority_path = core.no_symlink_path(target_authority_path)
    if (capability.target_authority_sha256 != core.EXPECTED_TARGET_AUTHORITY_SHA256 or
            core.sha256_file(authority_path) != core.EXPECTED_TARGET_AUTHORITY_SHA256):
        raise PermissionError("recovery target authority identity mismatch")
    first_access = _verify_first_access(stage_root, capability)
    authority_value = core.read_json(authority_path)
    trips, status, records = verify_existing_snapshots(stage_root, authority_value)
    return {
        "stage_root": stage_root,
        "capability": capability,
        "target_authority": core.entry(authority_path, relative_to=stage_root),
        "r4_package": r4_package,
        "recovery_package": recovery_package,
        "first_access": first_access,
        "trips": trips,
        "status": status,
        "snapshot_records": records,
        "snapshot_root_sha256": core.canonical_hash(records),
    }


def recover(authorization_path: Path, stage_root: Path, raw_data_root: Path,
            target_authority_path: Path, r4_package_manifest: Path,
            r4_package_sha256: str, recovery_package_manifest: Path,
            recovery_package_sha256: str) -> dict:
    stage_root = core.no_symlink_path(stage_root, require_file=False)
    output_root = stage_root / core.PB1_OUT
    target_path = output_root / "materialized_targets.npz"
    materialization_path = output_root / "target_materialization_provenance.json"
    event_path = output_root / "recovery_event.json"
    recovery_path = output_root / "recovery_provenance.json"
    for path in (target_path, materialization_path, event_path, recovery_path):
        if path.exists():
            raise FileExistsError("repeat/conflicting recovery publication: " + str(path))
    context = validate_recovery_context(
        authorization_path, stage_root, raw_data_root, target_authority_path,
        r4_package_manifest, r4_package_sha256, recovery_package_manifest,
        recovery_package_sha256)
    capability = context["capability"]
    parser_paths = context["trips"] + context["status"]
    event = {
        "schema_version": core.RECOVERY_EVENT_SCHEMA,
        "status": "PB1_R5_RECOVERY_ABOUT_TO_PARSE_VERIFIED_R4_SNAPSHOTS",
        "reason": core.RECOVERY_REASON,
        "execution_id": capability.execution_id,
        "authorization_sha256": capability.authorization_sha256,
        "first_label_access": context["first_access"],
        "original_r4_package": context["r4_package"],
        "recovery_package": context["recovery_package"],
        "verified_snapshot_root_sha256": context["snapshot_root_sha256"],
        "parsed_snapshot_paths": [path.relative_to(stage_root).as_posix() for path in parser_paths],
        "original_raw_sources_reopened_during_recovery": False,
        "created_utc": core.utc(),
    }
    event_entry = core.atomic_json(event_path, event)
    arrays = r4_materializer._materialize_arrays(
        stage_root, context["trips"], context["status"])
    if set(map(int, np.unique(arrays["city_id"]))) != set(core.TARGETS):
        raise RuntimeError("recovered target-city set mismatch")
    if (np.any(arrays["count"][arrays["label_valid"]] < 0) or
            np.any(arrays["count"][~arrays["label_valid"]] != -1)):
        raise RuntimeError("recovered label-valid/sentinel semantics mismatch")
    target_entry = core.atomic_bytes(target_path, core.deterministic_npz(arrays))
    materialization = {
        "schema_version": core.MATERIALIZATION_SCHEMA,
        "status": "PB1_R5_TARGETS_RECOVERED_FROM_EXISTING_R4_SNAPSHOTS",
        "reason": core.RECOVERY_REASON,
        "execution_id": capability.execution_id,
        "authorization_sha256": capability.authorization_sha256,
        "target_authority_sha256": capability.target_authority_sha256,
        "first_label_access": context["first_access"],
        "recovery_event": event_entry,
        "original_r4_package": context["r4_package"],
        "recovery_package": context["recovery_package"],
        "verified_snapshots": context["snapshot_records"],
        "parsed_snapshot_paths": event["parsed_snapshot_paths"],
        "original_raw_paths_parsed": False,
        "original_raw_sources_reopened_during_recovery": False,
        "verified_snapshot_root_sha256": context["snapshot_root_sha256"],
        "materialized_targets": target_entry,
        "target_cities": list(core.TARGETS),
        "evaluation_hours_per_city": core.EVAL_HOURS,
        "scientific_summaries_emitted": False,
        "raw_rows_printed": False,
        "created_utc": core.utc(),
    }
    materialization_entry = core.atomic_json(materialization_path, materialization)
    recovery_provenance = {
        "schema_version": core.RECOVERY_PROVENANCE_SCHEMA,
        "status": "PB1_R5_OPERATIONAL_RECOVERY_COMPLETE",
        "reason": core.RECOVERY_REASON,
        "execution_id": capability.execution_id,
        "authorization_sha256": capability.authorization_sha256,
        "first_label_access": context["first_access"],
        "target_authority": context["target_authority"],
        "original_r4_package": context["r4_package"],
        "recovery_package": context["recovery_package"],
        "verified_snapshot_root_sha256": context["snapshot_root_sha256"],
        "materialized_targets": target_entry,
        "materialization_provenance": materialization_entry,
        "original_raw_sources_reopened_during_recovery": False,
        "scientific_semantics_changed": False,
        "created_utc": core.utc(),
    }
    recovery_entry = core.atomic_json(recovery_path, recovery_provenance)
    return {
        "status": "PASS_PB1_R5_OPERATIONAL_RECOVERY",
        "materialized_targets": target_entry,
        "materialization_provenance": materialization_entry,
        "recovery_provenance": recovery_entry,
        "verified_snapshots": 11,
        "original_raw_sources_reopened_during_recovery": False,
    }
