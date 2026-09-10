"""Append-only final freeze bound to the PB1 R6 scoring correction."""
from __future__ import annotations

from pathlib import Path

from research.final_v2_2_r7_r1_pb1_r5_recovery.freeze import _deterministic_archive

from . import core, validator


def freeze(
        authorization_path: Path, stage_root: Path, raw_data_root: Path,
        target_authority_path: Path, r4_package_manifest: Path,
        r4_package_sha256: str, recovery_package_manifest: Path,
        recovery_package_sha256: str, scoring_package_manifest: Path,
        scoring_package_sha256: str) -> dict:
    stage_root = core.no_symlink_path(stage_root, require_file=False)
    output_root = stage_root / core.PB1_OUT
    final_paths = (
        output_root / "final_validation_report.json",
        output_root / "frozen_result_payload_manifest.json",
        output_root / "final_result_archive.zip",
        output_root / "final_result_manifest.json",
    )
    if any(path.exists() for path in final_paths):
        raise FileExistsError("repeat/conflicting R6 final freeze publication")
    report = validator.validate_results(
        authorization_path, stage_root, raw_data_root, target_authority_path,
        r4_package_manifest, r4_package_sha256, recovery_package_manifest,
        recovery_package_sha256, scoring_package_manifest,
        scoring_package_sha256)
    report_entry = core.atomic_json(final_paths[0], report)
    provenance = core.read_json(output_root / "target_materialization_provenance.json")
    snapshot_root = provenance["verified_snapshot_root_sha256"]
    names = [
        "authorization.json", "first_label_access.json", "recovery_event.json",
        "materialized_targets.npz", "target_materialization_provenance.json",
        "recovery_provenance.json", "scoring_fix_provenance.json",
        "metrics.json", "bootstrap_summary.json", "bootstrap_replicates.npz",
        "paired_comparisons.json", "final_validation_report.json",
    ]
    files = [core.entry(core.no_symlink_path(output_root / name), relative_to=stage_root)
             for name in names]
    target_authority_entry = core.entry(
        core.no_symlink_path(target_authority_path), relative_to=stage_root)
    files.append(target_authority_entry)
    payload_manifest = {
        "schema_version": core.PAYLOAD_MANIFEST_SCHEMA,
        "status": "PB1_R6_SCORING_FIX_RESULT_PAYLOAD_FROZEN",
        "recovery_revision": core.RECOVERY_REVISION,
        "scoring_revision": core.SCORING_REVISION,
        "files": sorted(files, key=lambda item: item["path"]),
        "verified_snapshot_root_sha256": snapshot_root,
        "prediction_commitment_sha256": core.PREDICTION_COMMITMENT_SHA,
        "prediction_manifest_sha256": core.PREDICTION_MANIFEST_SHA,
        "materialized_targets_sha256": core.EXPECTED_MATERIALIZED_TARGETS_SHA256,
        "r5_recovery_provenance_sha256": core.EXPECTED_RECOVERY_PROVENANCE_SHA256,
        "phase_a_archive_sha256": core.PHASE_A_ARCHIVE_SHA,
        "r7_package_manifest_sha256": core.R7_PACKAGE_SHA,
        "original_r4_package_manifest_sha256": r4_package_sha256,
        "r5_recovery_package_manifest_sha256": recovery_package_sha256,
        "r6_scoring_fix_package_manifest_sha256": scoring_package_sha256,
        "final_protocol_hashes": [core.FINAL_PROTOCOL_MD_SHA,
                                  core.FINAL_PROTOCOL_JSON_SHA],
        "grl_clarification_hashes": [core.GRL_MD_SHA, core.GRL_JSON_SHA],
        "recovery_rerun": False,
        "raw_sources_reopened": False,
        "scientific_protocol_changed": False,
    }
    payload_entry = core.atomic_json(final_paths[1], payload_manifest)
    archive_files = files + [core.entry(final_paths[1], relative_to=stage_root)]
    core.atomic_bytes(final_paths[2], _deterministic_archive(stage_root, archive_files))
    archive_entry = core.entry(final_paths[2], relative_to=stage_root)
    bound = {
        "authorization": core.entry(core.no_symlink_path(authorization_path), relative_to=stage_root),
        "target_authority": target_authority_entry,
        "first_label_access": core.entry(output_root / "first_label_access.json", relative_to=stage_root),
        "recovery_event": core.entry(output_root / "recovery_event.json", relative_to=stage_root),
        "target_materialization_provenance": core.entry(output_root / "target_materialization_provenance.json", relative_to=stage_root),
        "recovery_provenance": core.entry(output_root / "recovery_provenance.json", relative_to=stage_root),
        "scoring_fix_provenance": core.entry(output_root / "scoring_fix_provenance.json", relative_to=stage_root),
        "metrics": core.entry(output_root / "metrics.json", relative_to=stage_root),
        "bootstrap_summary": core.entry(output_root / "bootstrap_summary.json", relative_to=stage_root),
        "bootstrap_replicates": core.entry(output_root / "bootstrap_replicates.npz", relative_to=stage_root),
        "paired_comparisons": core.entry(output_root / "paired_comparisons.json", relative_to=stage_root),
        "final_validation_report": report_entry | {"path": final_paths[0].relative_to(stage_root).as_posix()},
        "frozen_payload_manifest": payload_entry | {"path": final_paths[1].relative_to(stage_root).as_posix()},
        "prediction_commitment": core.entry(stage_root / core.PHASE_A_OUT / "predictions/prediction_commitment.json", relative_to=stage_root),
        "prediction_manifest": core.entry(stage_root / core.PHASE_A_OUT / "predictions/prediction_manifest.json", relative_to=stage_root),
        "r7_package_manifest": core.entry(stage_root / core.R7_PATHS["r7_package_manifest"][0], relative_to=stage_root),
        "r7_member_manifest": core.entry(stage_root / core.R7_PATHS["r7_member_manifest"][0], relative_to=stage_root),
        "r7_code_manifest": core.entry(stage_root / core.R7_PATHS["r7_code_manifest"][0], relative_to=stage_root),
        "r7_implementation_contract": core.entry(stage_root / core.R7_PATHS["r7_implementation_contract"][0], relative_to=stage_root),
        "job_manifest": core.entry(stage_root / core.R7_PATHS["job_manifest"][0], relative_to=stage_root),
        "evaluation_manifest": core.entry(stage_root / core.R7_PATHS["evaluation_manifest"][0], relative_to=stage_root),
        "final_data_manifest": core.entry(stage_root / core.R7_PATHS["final_data_manifest"][0], relative_to=stage_root),
        "final_protocol_markdown": core.entry(stage_root / core.R7_PATHS["final_protocol_markdown"][0], relative_to=stage_root),
        "final_protocol_json": core.entry(stage_root / core.R7_PATHS["final_protocol_json"][0], relative_to=stage_root),
        "grl_clarification_markdown": core.entry(stage_root / core.R7_PATHS["grl_clarification_markdown"][0], relative_to=stage_root),
        "grl_clarification_json": core.entry(stage_root / core.R7_PATHS["grl_clarification_json"][0], relative_to=stage_root),
        "original_r4_package_manifest": core.entry(core.no_symlink_path(r4_package_manifest), relative_to=stage_root),
        "r5_recovery_package_manifest": core.entry(core.no_symlink_path(recovery_package_manifest), relative_to=stage_root),
        "r6_scoring_fix_package_manifest": core.entry(core.no_symlink_path(scoring_package_manifest), relative_to=stage_root),
    }
    final_manifest = {
        "schema_version": core.FINAL_MANIFEST_SCHEMA,
        "status": "FINAL_V2_2_RESULTS_FROZEN",
        "recovery_revision": core.RECOVERY_REVISION,
        "scoring_revision": core.SCORING_REVISION,
        "recovery_reason": core.RECOVERY_REASON,
        "bound_artifacts": bound,
        "verified_snapshot_root_sha256": snapshot_root,
        "final_result_archive": archive_entry,
        "phase_a_archive": core.read_json(
            stage_root / core.PB1_OUT / "staging_manifest.json")["phase_a_archive"],
        "training_performed_by_pb1": False,
        "predictions_modified_by_pb1": False,
        "new_learned_checkpoints": 0,
        "recovery_rerun": False,
        "raw_sources_reopened": False,
        "scientific_protocol_changed": False,
    }
    manifest_entry = core.atomic_json(final_paths[3], final_manifest)
    return {
        "status": "FINAL_V2_2_RESULTS_FROZEN",
        "recovery_revision": core.RECOVERY_REVISION,
        "scoring_revision": core.SCORING_REVISION,
        "final_result_manifest": manifest_entry,
        "final_result_archive": archive_entry,
        "verified_snapshot_root_sha256": snapshot_root,
    }

