"""Recovery-aware scorer using the unchanged frozen PB1 computations."""
from __future__ import annotations

from pathlib import Path

from research.final_v2_2_r7_r1_pb1_r4.scorer import compute_outputs

from . import core, recovery


def score(authorization_path: Path, stage_root: Path, raw_data_root: Path,
          target_authority_path: Path, r4_package_manifest: Path,
          r4_package_sha256: str, recovery_package_manifest: Path,
          recovery_package_sha256: str) -> dict:
    context = recovery.validate_recovery_context(
        authorization_path, stage_root, raw_data_root, target_authority_path,
        r4_package_manifest, r4_package_sha256, recovery_package_manifest,
        recovery_package_sha256)
    output_root = context["stage_root"] / core.PB1_OUT
    provenance = core.read_json(core.no_symlink_path(
        output_root / "target_materialization_provenance.json"))
    if (provenance.get("schema_version") != core.MATERIALIZATION_SCHEMA or
            provenance.get("status") != "PB1_R5_TARGETS_RECOVERED_FROM_EXISTING_R4_SNAPSHOTS" or
            provenance.get("authorization_sha256") != context["capability"].authorization_sha256 or
            provenance.get("original_raw_sources_reopened_during_recovery") is not False):
        raise PermissionError("recovery materialization provenance mismatch")
    target_path = core.no_symlink_path(Path(provenance["materialized_targets"]["path"]))
    if (core.sha256_file(target_path) != provenance["materialized_targets"]["sha256"] or
            target_path.stat().st_size != provenance["materialized_targets"]["bytes"]):
        raise PermissionError("recovered materialized targets changed")
    paths = {
        "metrics": output_root / "metrics.json",
        "bootstrap": output_root / "bootstrap_summary.json",
        "comparisons": output_root / "paired_comparisons.json",
        "replicates": output_root / "bootstrap_replicates.npz",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("repeat/conflicting score publication")
    metrics, bootstrap, comparisons, replicates = compute_outputs(
        context["stage_root"], target_path)
    published = {
        "metrics": core.atomic_json(paths["metrics"], metrics),
        "bootstrap": core.atomic_json(paths["bootstrap"], bootstrap),
        "comparisons": core.atomic_json(paths["comparisons"], comparisons),
        "replicates": core.atomic_bytes(paths["replicates"], core.deterministic_npz(replicates)),
    }
    return {
        "status": "PASS_PB1_R5_RECOVERY_SCORING",
        "outputs": published,
        "evaluation_passes": 468,
        "registered_reporting_cells": 660,
        "training_started": False,
        "predictions_written": False,
        "checkpoints_written": False,
    }
