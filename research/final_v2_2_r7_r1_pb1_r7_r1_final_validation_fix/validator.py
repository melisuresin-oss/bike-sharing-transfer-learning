"""Exact validator for the immutable R6 final archive and manifest."""
from __future__ import annotations

import zipfile
from pathlib import Path

from research.final_v2_2_r7_r1_pb1_r5_recovery import validator as r5_validator

from . import core


REQUIRED_BOUND = {
    "authorization", "target_authority", "first_label_access",
    "target_materialization_provenance", "recovery_event",
    "recovery_provenance", "scoring_fix_provenance",
    "prediction_commitment", "prediction_manifest",
    "r6_scoring_fix_package_manifest", "metrics", "bootstrap_summary",
    "bootstrap_replicates", "paired_comparisons", "final_validation_report",
    "frozen_payload_manifest",
}
REQUIRED_ARCHIVE = {
    "research/results/final_v2_2_r7_r1_pb1/authorization.json",
    "research/results/final_v2_2_r7_r1_pb1/first_label_access.json",
    "research/results/final_v2_2_r7_r1_pb1/materialized_targets.npz",
    "research/results/final_v2_2_r7_r1_pb1/target_materialization_provenance.json",
    "research/results/final_v2_2_r7_r1_pb1/recovery_event.json",
    "research/results/final_v2_2_r7_r1_pb1/recovery_provenance.json",
    "research/results/final_v2_2_r7_r1_pb1/scoring_fix_provenance.json",
    "research/results/final_v2_2_r7_r1_pb1/metrics.json",
    "research/results/final_v2_2_r7_r1_pb1/bootstrap_summary.json",
    "research/results/final_v2_2_r7_r1_pb1/bootstrap_replicates.npz",
    "research/results/final_v2_2_r7_r1_pb1/paired_comparisons.json",
    "research/results/final_v2_2_r7_r1_pb1/final_validation_report.json",
    "research/results/final_v2_2_r7_r1_pb1/frozen_result_payload_manifest.json",
    "research/results/final_v2_2_r7_r1_pb1_execution_seal/target_authority.json",
}


def _verify_archive_members(archive_path: Path, stage_root: Path,
                            payload_files: list[dict], payload_manifest: dict) -> dict:
    expected = {item["path"]: item for item in payload_files}
    if payload_manifest["path"] in expected:
        raise PermissionError("payload manifest duplicated inside its files list")
    expected[payload_manifest["path"]] = payload_manifest
    with zipfile.ZipFile(archive_path) as archive:
        entries = [item for item in archive.infolist() if not item.is_dir()]
        names = [item.filename for item in entries]
        if len(names) != len(set(names)):
            raise PermissionError("duplicate frozen archive member")
        missing = set(expected) - set(names)
        extra = set(names) - set(expected)
        if missing or extra:
            raise PermissionError(
                f"frozen archive member mismatch: missing={sorted(missing)} extra={sorted(extra)}")
        for item in entries:
            payload = archive.read(item)
            registered = expected[item.filename]
            if (len(payload) != registered["bytes"] or
                    core.sha256_bytes(payload) != registered["sha256"]):
                raise PermissionError("frozen archive member hash mismatch: " + item.filename)
            stage_path = core.no_symlink_path(stage_root / item.filename)
            if (stage_path.stat().st_size != len(payload) or
                    core.sha256_file(stage_path) != registered["sha256"]):
                raise PermissionError("archive/stage artifact mismatch: " + item.filename)
    return {"actual_members": names, "expected_members": sorted(expected),
            "missing_members": [], "extra_members": []}


def validate(stage_root: Path, validation_package_manifest: Path,
             validation_package_sha256: str) -> dict:
    stage_root = core.no_symlink_path(stage_root, require_file=False)
    package = core.verify_validation_package(
        stage_root, validation_package_manifest, validation_package_sha256)
    output_root = stage_root / core.PB1_OUT
    manifest_path = core.no_symlink_path(output_root / "final_result_manifest.json")
    archive_path = core.no_symlink_path(output_root / "final_result_archive.zip")
    if core.sha256_file(manifest_path) != core.EXPECTED_FINAL_MANIFEST_SHA256:
        raise PermissionError("frozen final manifest changed")
    if core.sha256_file(archive_path) != core.EXPECTED_FINAL_ARCHIVE_SHA256:
        raise PermissionError("frozen final archive changed")
    manifest = core.read_json(manifest_path)
    if (manifest.get("schema_version") != core.FINAL_MANIFEST_SCHEMA or
            manifest.get("status") != "FINAL_V2_2_RESULTS_FROZEN" or
            manifest.get("recovery_revision") != core.RECOVERY_REVISION or
            manifest.get("scoring_revision") != core.SCORING_REVISION):
        raise PermissionError("frozen final manifest state mismatch")
    bound = manifest.get("bound_artifacts", {})
    if not REQUIRED_BOUND <= set(bound):
        raise PermissionError("required frozen binding missing")
    for item in bound.values():
        core.verify_entry(stage_root, item)
    fixed = {
        "authorization": core.EXPECTED_AUTHORIZATION_SHA256,
        "first_label_access": core.EXPECTED_FIRST_ACCESS_SHA256,
        "recovery_provenance": core.EXPECTED_RECOVERY_PROVENANCE_SHA256,
        "prediction_commitment": core.PREDICTION_COMMITMENT_SHA,
        "prediction_manifest": core.PREDICTION_MANIFEST_SHA,
        "r6_scoring_fix_package_manifest": core.EXPECTED_R6_PACKAGE_SHA256,
    }
    for key, expected_sha in fixed.items():
        if bound[key]["sha256"] != expected_sha:
            raise PermissionError("fixed frozen binding mismatch: " + key)
    materialization = core.read_json(stage_root / bound["target_materialization_provenance"]["path"])
    target_entry = materialization.get("materialized_targets", {})
    target_path = core.no_symlink_path(Path(target_entry.get("path", "")))
    if (target_entry.get("sha256") != core.EXPECTED_MATERIALIZED_TARGETS_SHA256 or
            core.sha256_file(target_path) != core.EXPECTED_MATERIALIZED_TARGETS_SHA256 or
            target_path.stat().st_size != target_entry.get("bytes")):
        raise PermissionError("frozen materialized-target binding mismatch")
    authority = core.read_json(stage_root / bound["target_authority"]["path"])
    snapshots = r5_validator._verify_snapshot_records(stage_root, authority, materialization)
    scoring = core.read_json(stage_root / bound["scoring_fix_provenance"]["path"])
    if (scoring.get("status") != "PB1_R6_SCORING_FIX_APPLIED" or
            scoring.get("scientific_protocol_changed") is not False or
            scoring.get("predictions_changed") is not False or
            scoring.get("models_retrained") is not False or
            scoring.get("labels_changed") is not False or
            scoring.get("masks_changed") is not False or
            scoring.get("recovery_rerun") is not False or
            scoring.get("raw_sources_reopened") is not False):
        raise PermissionError("frozen R6 scoring provenance mismatch")
    payload_entry = bound["frozen_payload_manifest"]
    payload = core.read_json(stage_root / payload_entry["path"])
    member_state = _verify_archive_members(
        archive_path, stage_root, payload["files"], payload_entry)
    if not REQUIRED_ARCHIVE <= set(member_state["actual_members"]):
        raise PermissionError("scientifically required archive member missing")
    archive_entry = manifest.get("final_result_archive")
    if (archive_entry.get("sha256") != core.EXPECTED_FINAL_ARCHIVE_SHA256 or
            core.verify_entry(stage_root, archive_entry) != archive_path):
        raise PermissionError("final manifest/archive binding mismatch")
    return {
        "schema_version": core.VALIDATION_REPORT_SCHEMA,
        "status": "PASS_PB1_R7_R1_FINAL_VALIDATION_FIX",
        "validation_revision": core.VALIDATION_REVISION,
        "validation_fix_package": package,
        "frozen_final_manifest": core.entry(manifest_path, relative_to=stage_root),
        "frozen_final_archive": core.entry(archive_path, relative_to=stage_root),
        "actual_archive_members": member_state["actual_members"],
        "expected_archive_members": member_state["expected_members"],
        "missing_archive_members": [],
        "extra_archive_members": [],
        "verified_snapshots": len(snapshots),
        "required_scientific_members_present": True,
        "all_bound_artifacts_verified": True,
        "archive_members_match_stage_byte_for_byte": True,
        "scientific_outputs_modified": False,
        "recovery_rerun": False,
        "scoring_rerun": False,
        "raw_sources_reopened": False,
        "created_utc": core.utc(),
    }


def validate_and_record(stage_root: Path, validation_package_manifest: Path,
                        validation_package_sha256: str) -> dict:
    result = validate(stage_root, validation_package_manifest,
                      validation_package_sha256)
    report_path = Path(stage_root).resolve() / core.VALIDATION_REPORT_RELATIVE
    report = core.atomic_json(report_path, result)
    return result | {"validation_recovery_report": report}



