"""R4 scoring entry; mathematical implementation remains the sealed PB1 scorer."""
from __future__ import annotations

from pathlib import Path

from research.final_v2_2_r7_r1_pb1 import authorization
from research.final_v2_2_r7_r1_pb1 import scorer as frozen_scorer

from . import core

compute_outputs = frozen_scorer.compute_outputs


def score(authorization_path: Path, stage_root: Path, raw_data_root: Path,
          target_authority_path: Path, pb1_package_manifest: Path, pb1_package_sha: str) -> dict:
    capability = authorization.validate(authorization_path, stage_root, raw_data_root,
                                        target_authority_path, pb1_package_manifest, pb1_package_sha)
    output_root = Path(capability.stage_root_canonical_path) / core.PB1_OUT
    provenance = core.read_json(output_root / "target_materialization_provenance.json")
    if provenance.get("schema_version") != core.MATERIALIZATION_SCHEMA or provenance.get("authorization_sha256") != capability.authorization_sha256:
        raise PermissionError("R4 materialization provenance mismatch")
    target_path = core.no_symlink_path(Path(provenance["materialized_targets"]["path"]))
    if core.sha256_file(target_path) != provenance["materialized_targets"]["sha256"]:
        raise PermissionError("materialized targets changed")
    paths = {"metrics": output_root / "metrics.json", "bootstrap": output_root / "bootstrap_summary.json",
             "comparisons": output_root / "paired_comparisons.json", "replicates": output_root / "bootstrap_replicates.npz"}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("repeat/conflicting score publication")
    metrics, bootstrap, comparisons, replicates = compute_outputs(Path(capability.stage_root_canonical_path), target_path)
    published = {
        "metrics": core.atomic_json(paths["metrics"], metrics),
        "bootstrap": core.atomic_json(paths["bootstrap"], bootstrap),
        "comparisons": core.atomic_json(paths["comparisons"], comparisons),
        "replicates": core.atomic_bytes(paths["replicates"], core.deterministic_npz(replicates)),
    }
    return {"status": "PASS_PB1_R4_SCORING", "outputs": published, "evaluation_passes": 468,
            "registered_reporting_cells": 660, "training_started": False,
            "predictions_written": False, "checkpoints_written": False}
