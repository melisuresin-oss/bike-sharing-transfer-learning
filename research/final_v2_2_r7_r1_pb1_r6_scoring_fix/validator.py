"""Strict read-only verifier for PB1 R6 scoring-fix outputs."""
from __future__ import annotations

import zipfile
from pathlib import Path

from research.final_v2_2_r7_r1_pb1 import staging
from research.final_v2_2_r7_r1_pb1 import validator as r3_validator
from research.final_v2_2_r7_r1_pb1_r5_recovery import validator as r5_validator

from . import context, core, scorer


def _verify_scoring_provenance(stage_root: Path, bound: dict,
                               metrics: dict, comparisons: dict) -> dict:
    output_root = stage_root / core.PB1_OUT
    path = core.no_symlink_path(output_root / "scoring_fix_provenance.json")
    value = core.read_json(path)
    if (value.get("schema_version") != core.SCORING_PROVENANCE_SCHEMA or
            value.get("status") != "PB1_R6_SCORING_FIX_APPLIED" or
            value.get("scoring_revision") != core.SCORING_REVISION or
            value.get("r6_scoring_fix_package") != bound["scoring_package"] or
            value.get("r5_recovery_package") != bound["recovery_package"] or
            value.get("original_r4_package") != bound["r4_package"] or
            value.get("prediction_commitment") != bound["prediction_commitment"] or
            value.get("prediction_manifest") != bound["prediction_manifest"] or
            value.get("r5_recovery_provenance") != bound["recovery_provenance"] or
            value.get("registered_zero_reference_candidates") !=
            metrics.get("registered_parameter_zero_candidates") or
            value.get("selected_deterministic_reference") !=
            metrics.get("selected_parameter_zero_reference") or
            comparisons.get("selected_parameter_zero_reference") !=
            metrics.get("selected_parameter_zero_reference")):
        raise PermissionError("R6 scoring-fix provenance binding mismatch")
    if (value.get("scientific_protocol_changed") is not False or
            value.get("predictions_changed") is not False or
            value.get("models_retrained") is not False or
            value.get("labels_changed") is not False or
            value.get("masks_changed") is not False or
            value.get("recovery_rerun") is not False or
            value.get("raw_sources_reopened") is not False):
        raise PermissionError("R6 scoring-fix firewall mismatch")
    target = value.get("materialized_targets")
    if target != core.entry(bound["target_path"], relative_to=stage_root):
        raise PermissionError("R6 target binding mismatch")
    expected_outputs = {
        "metrics": "metrics.json",
        "bootstrap": "bootstrap_summary.json",
        "comparisons": "paired_comparisons.json",
        "replicates": "bootstrap_replicates.npz",
    }
    if set(value.get("scoring_outputs", {})) != set(expected_outputs):
        raise PermissionError("R6 scoring-output provenance set mismatch")
    for key, name in expected_outputs.items():
        expected = core.entry(core.no_symlink_path(output_root / name),
                              relative_to=stage_root)
        if value["scoring_outputs"][key] != expected:
            raise PermissionError("R6 scoring-output provenance mismatch: " + name)
    return core.entry(path, relative_to=stage_root)


def validate_results(
        authorization_path: Path, stage_root: Path, raw_data_root: Path,
        target_authority_path: Path, r4_package_manifest: Path,
        r4_package_sha256: str, recovery_package_manifest: Path,
        recovery_package_sha256: str, scoring_package_manifest: Path,
        scoring_package_sha256: str) -> dict:
    bound = context.validate_scoring_context(
        authorization_path, stage_root, raw_data_root, target_authority_path,
        r4_package_manifest, r4_package_sha256, recovery_package_manifest,
        recovery_package_sha256, scoring_package_manifest,
        scoring_package_sha256)
    stage_root = bound["stage_root"]
    staged = staging.validate(stage_root)
    phase_a = r3_validator._verify_phase_a_commitment(stage_root)
    target_path, _, snapshots = r5_validator._verify_materialization(stage_root, bound)
    if target_path != bound["target_path"]:
        raise PermissionError("R5/R6 materialized-target disagreement")
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
        if (core.sha256_file(path) != core.sha256_bytes(payload) or
                path.stat().st_size != len(payload)):
            raise PermissionError("R6 scoring output does not reproduce: " + name)
    metrics = core.read_json(output_root / "metrics.json")
    bootstrap = core.read_json(output_root / "bootstrap_summary.json")
    comparisons = core.read_json(output_root / "paired_comparisons.json")
    statuses = metrics.get("registered_parameter_zero_candidates")
    if ([item.get("method") for item in statuses or []] !=
            list(core.REGISTERED_ZERO_REFERENCES) or
            any(set(item) != {"method", "equal_city_macro_mae_estimable"}
                for item in statuses or [])):
        raise PermissionError("registered zero-reference status mismatch")
    estimable = {item["method"] for item in statuses
                 if item["equal_city_macro_mae_estimable"]}
    selected = metrics.get("selected_parameter_zero_reference")
    if (selected is None and estimable) or (selected is not None and selected not in estimable):
        raise PermissionError("selected zero reference is inconsistent with estimability")
    if (metrics.get("evaluation_pass_count") != 468 or
            metrics.get("registered_reporting_cell_count") != 660 or
            metrics.get("aggregated_city_cell_count") != 180 or
            metrics.get("equal_city_macro_cell_count") != 45):
        raise PermissionError("R6 reporting structure mismatch")
    if (bootstrap.get("replicates") != 2000 or
            bootstrap.get("block_lengths") != [168] * 9 + [53] or
            bootstrap.get("undefined_replicates_redrawn") is not False or
            bootstrap.get("training_seed_randomness_inside_bootstrap") is not False):
        raise PermissionError("R6 bootstrap contract mismatch")
    if (len(comparisons.get("comparison_families", [])) != 27 or
            len(comparisons.get("city_comparisons", [])) != 108 or
            len(comparisons.get("equal_city_macro_comparisons", [])) != 27 or
            comparisons.get("gain_definition") != "error(reference)-error(method)" or
            comparisons.get("positive_favors_method") is not True):
        raise PermissionError("R6 comparison structure mismatch")
    if selected is None:
        unavailable = [row for row in comparisons["city_comparisons"]
                       if row["family"] in {"ordinary_zero_transfer",
                                            "grl_zero_transfer"}]
        if (len(unavailable) != 8 or
                any(row.get("availability_status") !=
                    "UNAVAILABLE_NO_ESTIMABLE_DETERMINISTIC_ZERO_REFERENCE"
                    for row in unavailable)):
            raise PermissionError("undefined zero-reference gains not explicit")
    provenance = _verify_scoring_provenance(stage_root, bound, metrics, comparisons)
    forbidden = [path for path in output_root.rglob("*")
                 if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt"}]
    if forbidden:
        raise PermissionError("R6 scoring created an unauthorized checkpoint")
    return {
        "schema_version": core.VALIDATION_SCHEMA,
        "status": "PASS_PB1_R6_SCORING_FIX_RESULTS_PRE_FREEZE",
        "execution_id": bound["capability"].execution_id,
        "phase_a_files_verified": staged["phase_a_files"],
        "completions_verified": phase_a["completions"],
        "predictions_verified": phase_a["predictions"],
        "verified_snapshots": len(snapshots),
        "evaluation_passes": 468,
        "registered_reporting_cells": 660,
        "aggregated_city_cells": 180,
        "equal_city_macro_cells": 45,
        "bootstrap_replicates": 2000,
        "registered_zero_reference_candidates": list(core.REGISTERED_ZERO_REFERENCES),
        "scoring_fix_provenance": provenance,
        "recovery_rerun": False,
        "raw_sources_reopened": False,
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
            manifest.get("recovery_revision") != core.RECOVERY_REVISION or
            manifest.get("scoring_revision") != core.SCORING_REVISION):
        raise PermissionError("R6 final-result manifest state mismatch")
    for item in manifest["bound_artifacts"].values():
        core.verify_entry(stage_root, item)
    scoring_provenance = core.read_json(
        stage_root / manifest["bound_artifacts"]["scoring_fix_provenance"]["path"])
    if (scoring_provenance.get("scoring_revision") != core.SCORING_REVISION or
            scoring_provenance.get("raw_sources_reopened") is not False or
            scoring_provenance.get("recovery_rerun") is not False):
        raise PermissionError("R6 frozen scoring provenance mismatch")
    payload_manifest = core.read_json(
        stage_root / manifest["bound_artifacts"]["frozen_payload_manifest"]["path"])
    if (payload_manifest.get("recovery_revision") != core.RECOVERY_REVISION or
            payload_manifest.get("scoring_revision") != core.SCORING_REVISION):
        raise PermissionError("R6 frozen payload revision mismatch")
    archive_entry = manifest["final_result_archive"]
    archive_path = core.verify_entry(stage_root, archive_entry)
    expected = {item["path"]: item for item in payload_manifest["files"]}
    with zipfile.ZipFile(archive_path) as archive:
        entries = [item for item in archive.infolist() if not item.is_dir()]
        if len(entries) != len(expected) or {item.filename for item in entries} != set(expected):
            raise PermissionError("R6 final archive member-set mismatch")
        for item in entries:
            payload = archive.read(item)
            registered = expected[item.filename]
            if (len(payload) != registered["bytes"] or
                    core.sha256_bytes(payload) != registered["sha256"]):
                raise PermissionError("R6 final archive member tampering: " + item.filename)
    return {
        "status": "PASS_PB1_R6_SCORING_FIX_FINAL_RESULT_FROZEN",
        "archive": archive_entry,
        "manifest": core.entry(manifest_path, relative_to=stage_root),
        "recovery_rerun": False,
        "raw_sources_reopened": False,
        "phase_a_evidence_modified": False,
        "models_retrained_after_commitment": False,
    }

