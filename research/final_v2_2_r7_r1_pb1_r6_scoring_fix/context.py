"""Read-only binding of R6 scoring to the already recovered real execution."""
from __future__ import annotations

from pathlib import Path

from research.final_v2_2_r7_r1_pb1_r5_recovery import recovery as r5_recovery

from . import core


def validate_scoring_context(
        authorization_path: Path, stage_root: Path, raw_data_root: Path,
        target_authority_path: Path, r4_package_manifest: Path,
        r4_package_sha256: str, recovery_package_manifest: Path,
        recovery_package_sha256: str, scoring_package_manifest: Path,
        scoring_package_sha256: str) -> dict:
    if recovery_package_sha256 != core.EXPECTED_R5_PACKAGE_SHA256:
        raise PermissionError("R5 recovery-package hash is not the frozen authority")
    inherited = r5_recovery.validate_recovery_context(
        authorization_path, stage_root, raw_data_root, target_authority_path,
        r4_package_manifest, r4_package_sha256, recovery_package_manifest,
        recovery_package_sha256)
    stage_root = inherited["stage_root"]
    scoring_package = core.verify_scoring_package(
        stage_root, scoring_package_manifest, scoring_package_sha256)
    output_root = stage_root / core.PB1_OUT
    materialization = core.read_json(core.no_symlink_path(
        output_root / "target_materialization_provenance.json"))
    recovery_path = core.no_symlink_path(output_root / "recovery_provenance.json")
    if core.sha256_file(recovery_path) != core.EXPECTED_RECOVERY_PROVENANCE_SHA256:
        raise PermissionError("R5 recovery provenance changed")
    recovery = core.read_json(recovery_path)
    if (materialization.get("status") !=
            "PB1_R5_TARGETS_RECOVERED_FROM_EXISTING_R4_SNAPSHOTS" or
            materialization.get("original_raw_sources_reopened_during_recovery") is not False or
            recovery.get("status") != "PB1_R5_OPERATIONAL_RECOVERY_COMPLETE" or
            recovery.get("original_raw_sources_reopened_during_recovery") is not False or
            recovery.get("scientific_semantics_changed") is not False):
        raise PermissionError("R5 recovery state is not the frozen successful state")
    target_path = core.no_symlink_path(Path(materialization["materialized_targets"]["path"]))
    if (target_path.stat().st_size != materialization["materialized_targets"]["bytes"] or
            core.sha256_file(target_path) != core.EXPECTED_MATERIALIZED_TARGETS_SHA256 or
            materialization["materialized_targets"]["sha256"] !=
            core.EXPECTED_MATERIALIZED_TARGETS_SHA256):
        raise PermissionError("materialized targets changed")
    commitment_path = core.no_symlink_path(
        stage_root / core.PHASE_A_OUT / "predictions/prediction_commitment.json")
    manifest_path = core.no_symlink_path(
        stage_root / core.PHASE_A_OUT / "predictions/prediction_manifest.json")
    if core.sha256_file(commitment_path) != core.PREDICTION_COMMITMENT_SHA:
        raise PermissionError("prediction commitment changed")
    if core.sha256_file(manifest_path) != core.PREDICTION_MANIFEST_SHA:
        raise PermissionError("prediction manifest changed")
    return inherited | {
        "stage_root": stage_root,
        "target_path": target_path,
        "materialization": materialization,
        "recovery_provenance": core.entry(recovery_path, relative_to=stage_root),
        "scoring_package": scoring_package,
        "scoring_package_sha256": scoring_package_sha256,
        "prediction_commitment": core.entry(commitment_path, relative_to=stage_root),
        "prediction_manifest": core.entry(manifest_path, relative_to=stage_root),
    }

