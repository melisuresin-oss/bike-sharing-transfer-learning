from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import authority, core, staging


@dataclass(frozen=True)
class PhaseBCapability:
    authorization_sha256: str
    target_authority_sha256: str
    prediction_commitment_sha256: str
    prediction_manifest_sha256: str
    stage_root_canonical_path: str
    raw_data_root_canonical_path: str
    execution_id: str
    authorization_created_utc: str


def _phase_a_bindings(stage_root: Path) -> dict:
    result = {}
    for name, (relative, expected) in core.R7_PATHS.items():
        result[name] = core.verify_fixed_file(stage_root, relative, expected)
    commitment_path = stage_root / core.PHASE_A_OUT / "predictions/prediction_commitment.json"
    prediction_path = stage_root / core.PHASE_A_OUT / "predictions/prediction_manifest.json"
    if core.sha256_file(commitment_path) != core.PREDICTION_COMMITMENT_SHA:
        raise PermissionError("prediction commitment changed")
    if core.sha256_file(prediction_path) != core.PREDICTION_MANIFEST_SHA:
        raise PermissionError("prediction manifest changed")
    commitment = core.read_json(commitment_path)
    if commitment.get("schema_version") != "final_v2_2_r7_r1.prediction_commitment.1" or commitment.get("status") != "PHASE_A_PREDICTIONS_STRONGLY_COMMITTED":
        raise PermissionError("prediction commitment schema/state mismatch")
    if commitment.get("bound_root_sha256") != core.canonical_hash(commitment.get("bound")):
        raise PermissionError("prediction commitment root mismatch")
    bound_prediction = commitment["bound"]["prediction_manifest"]
    if bound_prediction.get("sha256") != core.PREDICTION_MANIFEST_SHA:
        raise PermissionError("commitment does not bind frozen prediction manifest")
    prediction = core.read_json(prediction_path)
    if prediction.get("schema_version") != "final_v2_2_r7_r1.predictions.1" or prediction.get("count") != 468 or prediction.get("final_labels_joined") is not False:
        raise PermissionError("prediction manifest schema/firewall mismatch")
    result["prediction_commitment"] = core.entry(commitment_path, relative_to=stage_root)
    result["prediction_manifest"] = core.entry(prediction_path, relative_to=stage_root)
    return result


def create(stage_root: Path, raw_data_root: Path, target_authority_path: Path,
           pb1_package_manifest: Path, pb1_package_sha: str, output: Path) -> dict:
    stage_root = core.no_symlink_path(stage_root, require_file=False)
    raw_data_root = core.no_symlink_path(raw_data_root, require_file=False)
    staging.validate(stage_root)
    final_data = stage_root / core.R7_PATHS["final_data_manifest"][0]
    authority.validate(target_authority_path, final_data)
    package_path = core.no_symlink_path(pb1_package_manifest)
    if core.sha256_file(package_path) != pb1_package_sha:
        raise PermissionError("PB1 package manifest hash mismatch")
    package = core.read_json(package_path)
    if package.get("schema_version") != core.REVISION + ".package.1" or package.get("status") != "PB1_PACKAGE_FROZEN_NOT_EXECUTED":
        raise PermissionError("PB1 package state mismatch")
    phase_a = _phase_a_bindings(stage_root)
    target_entry = core.entry(core.no_symlink_path(target_authority_path), relative_to=stage_root)
    package_entry = core.entry(package_path, relative_to=stage_root)
    bindings = {
        "phase_a": phase_a,
        "phase_a_archive": core.read_json(stage_root / core.PB1_OUT / "staging_manifest.json")["phase_a_archive"],
        "target_authority": target_entry,
        "pb1_package_manifest": package_entry,
        "pb1_member_manifest": package["package_member_manifest"],
        "pb1_code_manifest": package["executable_code_manifest"],
        "pb1_implementation_contract": package["implementation_contract"],
        "stage_root_canonical_path": str(stage_root),
        "raw_data_root_canonical_path": str(raw_data_root),
    }
    execution_id = core.canonical_hash(bindings)
    payload = {
        "schema_version": core.AUTH_SCHEMA,
        "status": "FINAL_V2_2_R7_R1_PB1_EXPLICITLY_AUTHORIZED",
        "execution_id": execution_id,
        "bindings": bindings,
        "declarations": {
            "prediction_commitment_precedes_authorization": True,
            "raw_content_opened_during_authorization": False,
            "raw_hashes_recomputed_during_authorization": False,
            "training_authorized": False,
            "prediction_generation_authorized": False,
            "phase_b_scoring_authorized": True,
        },
        "created_utc": core.utc(),
    }
    result = core.atomic_json(output, payload)
    return {"status": "PB1_PHASE_B_AUTHORIZATION_PUBLISHED", "authorization": result,
            "execution_id": execution_id, "raw_content_opened": False}


def validate(authorization_path: Path, stage_root: Path, raw_data_root: Path,
             target_authority_path: Path, pb1_package_manifest: Path,
             pb1_package_sha: str) -> PhaseBCapability:
    stage_root = core.no_symlink_path(stage_root, require_file=False)
    raw_data_root = core.no_symlink_path(raw_data_root, require_file=False)
    auth_path = core.no_symlink_path(authorization_path)
    auth = core.read_json(auth_path)
    if auth.get("schema_version") != core.AUTH_SCHEMA or auth.get("status") != "FINAL_V2_2_R7_R1_PB1_EXPLICITLY_AUTHORIZED":
        raise PermissionError("PB1 authorization schema/state mismatch")
    declarations = auth.get("declarations", {})
    if declarations != {"prediction_commitment_precedes_authorization": True,
                        "raw_content_opened_during_authorization": False,
                        "raw_hashes_recomputed_during_authorization": False,
                        "training_authorized": False,
                        "prediction_generation_authorized": False,
                        "phase_b_scoring_authorized": True}:
        raise PermissionError("authorization declarations mismatch")
    if core.sha256_file(pb1_package_manifest) != pb1_package_sha:
        raise PermissionError("PB1 package manifest changed")
    package = core.read_json(pb1_package_manifest)
    final_data = stage_root / core.R7_PATHS["final_data_manifest"][0]
    authority.validate(target_authority_path, final_data)
    expected_bindings = {
        "phase_a": _phase_a_bindings(stage_root),
        "phase_a_archive": core.read_json(stage_root / core.PB1_OUT / "staging_manifest.json")["phase_a_archive"],
        "target_authority": core.entry(target_authority_path, relative_to=stage_root),
        "pb1_package_manifest": core.entry(pb1_package_manifest, relative_to=stage_root),
        "pb1_member_manifest": package["package_member_manifest"],
        "pb1_code_manifest": package["executable_code_manifest"],
        "pb1_implementation_contract": package["implementation_contract"],
        "stage_root_canonical_path": str(stage_root),
        "raw_data_root_canonical_path": str(raw_data_root),
    }
    if auth.get("bindings") != expected_bindings or auth.get("execution_id") != core.canonical_hash(expected_bindings):
        raise PermissionError("PB1 authorization binding mismatch")
    datetime.fromisoformat(auth["created_utc"])
    return PhaseBCapability(core.sha256_file(auth_path), expected_bindings["target_authority"]["sha256"],
                            core.PREDICTION_COMMITMENT_SHA, core.PREDICTION_MANIFEST_SHA,
                            str(stage_root), str(raw_data_root), auth["execution_id"], auth["created_utc"])

