from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "research/results/stage2b_vanilla_fairness_v2_1_a40_complete.zip"
FROZEN_MANIFEST = ROOT / "research/results/stage2b_vanilla_fairness_v2_1/stage2b_fairness_manifest.json"
R2_ROOT = ROOT / "research/results/stage2b_execution_package_v2_1_r2"
JOB_MAP = R2_ROOT / "stage2b_remote_job_map.csv"
R2_PACKAGE_MANIFEST = R2_ROOT / "stage2b_execution_package_manifest.json"
R2_AUTHORIZATION = R2_ROOT / "STAGE2B_EXECUTION_AUTHORIZATION.json"
R0_INCIDENT = ROOT / "research/results/stage2b_execution_package_v2_1_r1/REMOTE_STAGE2B_ABORTED_PRETRAIN_CACHE_BUILD.json"
R1_INCIDENT = R2_ROOT / "REMOTE_STAGE2B_R1_ABORTED_FIRST_FORWARD.json"
CLOSURE_ROOT = ROOT / "research/results/stage2b_vanilla_fairness_v2_1_a40_closure"
COMPLETION_ARTIFACT = CLOSURE_ROOT / "stage2b_completion_provenance.json"
COMPLETION_REPORT = CLOSURE_ROOT / "STAGE2B_COMPLETION_PROVENANCE.md"
HASH_LEDGER = CLOSURE_ROOT / "STAGE2B_COMPLETION_PROVENANCE.sha256"

EXPECTED = {
    "archive_sha256": "306f61dfd842b15269ac22f4c84e31dbd97029fb3036a320116749104406a89c",
    "frozen_manifest_sha256": "6b3f7e4936e201c91023145a73e36b4a122e8a97305ef408bc14a6c727c6259f",
    "job_map_sha256": "72db61af7a3ea7ad33dd92b6f6fe00bf29d0ee93196840768a56222ab0291302",
    "r2_package_manifest_sha256": "daa47d7e91df010828fedf3a2e2f73c8666cda21fde49f0f12a24810c480f548",
    "r2_authorization_sha256": "3b1a98da6568faddd275df3b122e6e5d7d140e2f52c7e6a142caa947b70152fb",
    "r0_incident_sha256": "508ba7eee08b2a961a56cc3859884bc6c6ca574079a030c81cb0bb19831547d3",
    "r1_incident_sha256": "f87b84651d65b328467eb49ae7e4c7fda38ac2cba9501335db0695c9b4507f7f",
    "summary_sha256": "d41811d1b6103f8b577da55a51e3caf6879c0f14f4967f96ab92a3daaa7628da",
}
FINAL_TARGET_IDS = {195, 199, 237, 617}
PSEUDO_TARGET_IDS = {476, 532, 619, 658}
EXPECTED_DETERMINISM = {
    "cublas_workspace_config": ":4096:8",
    "deterministic_algorithms": True,
    "cuda_matmul_allow_tf32": False,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cudnn_allow_tf32": False,
}
EXPECTED_SUMMARY = [
    {
        "config_id": "VGRU_H032_D00", "equal_fold_mae": "1.5373305020034158",
        "equal_fold_mae_4dp": "1.5373", "fold_population_std": "1.7883942767239285",
        "hidden_size": "32", "dropout": "0.0",
    },
    {
        "config_id": "VGRU_H032_D10", "equal_fold_mae": "1.5379446526640719",
        "equal_fold_mae_4dp": "1.5379", "fold_population_std": "1.7880526623732274",
        "hidden_size": "32", "dropout": "0.1",
    },
    {
        "config_id": "VGRU_H064_D00", "equal_fold_mae": "1.552849683865485",
        "equal_fold_mae_4dp": "1.5528", "fold_population_std": "1.7798397390605007",
        "hidden_size": "64", "dropout": "0.0",
    },
    {
        "config_id": "VGRU_H064_D10", "equal_fold_mae": "1.5530258456955182",
        "equal_fold_mae_4dp": "1.5530", "fold_population_std": "1.7797421993108733",
        "hidden_size": "64", "dropout": "0.1",
    },
]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def read_json_file(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _zip_index(handle: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    result: dict[str, zipfile.ZipInfo] = {}
    for item in handle.infolist():
        if item.is_dir():
            continue
        name = item.filename.replace("\\", "/")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or name in result:
            raise RuntimeError(f"Unsafe or duplicate archive member: {name}")
        result[name] = item
    return result


def _member_bytes(handle: zipfile.ZipFile, index: dict[str, zipfile.ZipInfo], name: str) -> bytes:
    if name not in index:
        raise RuntimeError(f"Missing archive member: {name}")
    return handle.read(index[name])


def _member_json(handle: zipfile.ZipFile, index: dict[str, zipfile.ZipInfo], name: str) -> dict[str, Any]:
    value = json.loads(_member_bytes(handle, index, name).decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected archive JSON object: {name}")
    return value


def _archive_member_for_repository_path(value: str) -> str:
    prefix = "research/results/stage2b_vanilla_fairness_v2_1_a40/"
    normalized = value.replace("\\", "/")
    if not normalized.startswith(prefix):
        raise RuntimeError(f"Scientific artifact escaped the isolated Stage-2B result root: {value}")
    return normalized[len(prefix):]


def collect_existing_evidence() -> dict[str, Any]:
    """Read and hash completed artifacts; never import model/evaluation code."""
    file_bindings = {
        "archive_sha256": sha256_file(ARCHIVE),
        "frozen_manifest_sha256": sha256_file(FROZEN_MANIFEST),
        "job_map_sha256": sha256_file(JOB_MAP),
        "r2_package_manifest_sha256": sha256_file(R2_PACKAGE_MANIFEST),
        "r2_authorization_sha256": sha256_file(R2_AUTHORIZATION),
        "r0_incident_sha256": sha256_file(R0_INCIDENT),
        "r1_incident_sha256": sha256_file(R1_INCIDENT),
    }
    for key, observed in file_bindings.items():
        if observed != EXPECTED[key]:
            raise RuntimeError(f"Frozen/archived binding mismatch for {key}: {observed}")

    frozen = read_json_file(FROZEN_MANIFEST)
    if frozen.get("status") != "FROZEN_VERIFIED" or frozen.get("workload") != {
        "configurations": 4, "pseudo_target_folds": 4, "seeds": 3,
        "source_fits": 48, "target_adaptation_fits": 0, "zero_shot_evaluations": 48,
    }:
        raise RuntimeError("Frozen Stage-2B scientific manifest changed")
    package = read_json_file(R2_PACKAGE_MANIFEST)
    authorization = read_json_file(R2_AUTHORIZATION)
    if package.get("package_revision") != "R2_CUBLAS_DETERMINISM_FIX":
        raise RuntimeError("R2 execution package revision changed")
    if package.get("frozen_stage2b_manifest_sha256") != EXPECTED["frozen_manifest_sha256"]:
        raise RuntimeError("R2 package is not bound to the frozen scientific manifest")
    if package.get("job_map", {}).get("sha256") != EXPECTED["job_map_sha256"]:
        raise RuntimeError("R2 package is not bound to the immutable job map")
    if authorization.get("execution_package_manifest_sha256") != EXPECTED["r2_package_manifest_sha256"]:
        raise RuntimeError("R2 authorization package binding changed")
    if authorization.get("job_map_sha256") != EXPECTED["job_map_sha256"]:
        raise RuntimeError("R2 authorization job-map binding changed")
    if authorization.get("stage4_or_final_execution_authorized") is not False:
        raise RuntimeError("R2 authorization scope expanded")

    r0 = read_json_file(R0_INCIDENT)
    r1 = read_json_file(R1_INCIDENT)
    if r0.get("REMOTE_STAGE2B_ATTEMPT") != "ABORTED_PRETRAIN_CACHE_BUILD_IMPLEMENTATION_ERROR":
        raise RuntimeError("R0 incident record changed")
    if any(r0.get(key) is not False for key in (
        "scientific_training_started", "evaluation_started", "scientific_results_generated",
        "pseudo_target_mae_produced", "final_target_labels_accessed",
    )):
        raise RuntimeError("R0 incident is not a zero-result pretraining abort")
    if r1.get("REMOTE_STAGE2B_R1_ATTEMPT") != "ABORTED_FIRST_FORWARD_MISSING_CUBLAS_WORKSPACE_CONFIG":
        raise RuntimeError("R1 incident record changed")
    if r1.get("optimizer_updates_completed") != 0 or r1.get("scientific_fit_completed") is not False:
        raise RuntimeError("R1 incident is not a zero-update/zero-fit abort")
    if any(r1.get(key) is not False for key in (
        "evaluation_started", "pseudo_target_mae_produced", "model_selection_performed",
        "scientific_results_generated", "final_target_labels_accessed",
    )):
        raise RuntimeError("R1 incident generated a forbidden scientific result")

    with JOB_MAP.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 48 or len({row["job_key"] for row in rows}) != 48:
        raise RuntimeError("Immutable job map is not 48 unique jobs")
    expected_by_key = {row["job_key"]: row for row in rows}

    with zipfile.ZipFile(ARCHIVE, "r") as handle:
        index = _zip_index(handle)
        if len(index) != 199:
            raise RuntimeError(f"Unexpected completed-archive entry count: {len(index)}")
        preflight_bytes = _member_bytes(handle, index, "REMOTE_PREFLIGHT_R2.json")
        validator_bytes = _member_bytes(handle, index, "STAGE2B_REMOTE_VALIDATION.json")
        summary_bytes = _member_bytes(handle, index, "stage2b_config_summary.csv")
        preflight = json.loads(preflight_bytes.decode("utf-8"))
        validator = json.loads(validator_bytes.decode("utf-8"))
        summary = list(csv.DictReader(io.StringIO(summary_bytes.decode("utf-8"))))
        if sha256_bytes(summary_bytes) != EXPECTED["summary_sha256"] or summary != EXPECTED_SUMMARY:
            raise RuntimeError("Stage-2B configuration summary content/hash changed")
        expected_validator = {
            "status": "PASS", "source_fits": 48, "zero_shot_evaluations": 48,
            "fine_tuning_fits": 0, "validated_remote_jobs": 48,
            "selected_config_id": "VGRU_H032_D00",
            "selection_rule_applied_exactly_as_frozen": True,
            "final_label_firewall_passed": True, "final_target_labels_accessed": False,
            "does_not_authorize_stage4_or_final_execution": True,
            "summary": "research/results/stage2b_vanilla_fairness_v2_1_a40/stage2b_config_summary.csv",
            "summary_sha256": EXPECTED["summary_sha256"],
        }
        if validator != expected_validator:
            raise RuntimeError("Official Stage-2B remote validator record changed")
        if preflight.get("status") != "PASS" or preflight.get("gate") != "STRICT_REMOTE_RUNTIME":
            raise RuntimeError("R2 strict remote preflight did not pass")
        if preflight.get("execution_package_manifest_sha256") != EXPECTED["r2_package_manifest_sha256"]:
            raise RuntimeError("R2 remote preflight package binding changed")
        if preflight.get("execution_authorization_sha256") != EXPECTED["r2_authorization_sha256"]:
            raise RuntimeError("R2 remote preflight authorization binding changed")
        if preflight.get("job_map_sha256") != EXPECTED["job_map_sha256"]:
            raise RuntimeError("R2 remote preflight job-map binding changed")
        if any(preflight.get("deterministic_settings", {}).get(key) != value for key, value in EXPECTED_DETERMINISM.items()):
            raise RuntimeError("R2 deterministic runtime settings changed")
        fixture = preflight.get("cuda_determinism_fixture", {})
        if fixture.get("passed") is not True or fixture.get("repeated_outputs_bitwise_equal") is not True:
            raise RuntimeError("R2 CUDA determinism fixture did not pass")
        if fixture.get("maximum_absolute_difference") != 0.0 or fixture.get("training_or_evaluation_performed") is not False:
            raise RuntimeError("R2 CUDA fixture provenance changed")
        if preflight.get("final_target_labels_accessed") is not False:
            raise RuntimeError("R2 preflight final-label firewall changed")

        category_counts = {
            category: len([name for name in index if name.startswith(category + "/")])
            for category in ("jobs", "checkpoints", "predictions", "prediction_manifests")
        }
        if set(category_counts.values()) != {48}:
            raise RuntimeError(f"Completed scientific artifact counts are not all 48: {category_counts}")
        job_record_hashes: list[dict[str, str]] = []
        scientific_file_hashes: list[dict[str, str]] = []
        seen_remote_ids: set[int] = set()
        seen_job_keys: set[str] = set()
        per_config: Counter[str] = Counter()
        per_target: Counter[int] = Counter()
        per_seed: Counter[int] = Counter()
        for job_key, row in expected_by_key.items():
            job_member = _archive_member_for_repository_path(row["expected_job_result"])
            job_bytes = _member_bytes(handle, index, job_member)
            job = json.loads(job_bytes.decode("utf-8"))
            target = int(row["pseudo_target_city_id"])
            seed = int(row["seed"])
            remote_id = int(row["remote_job_id"])
            expected_sources = [int(item) for item in row["source_city_ids"].split(";")]
            if target not in PSEUDO_TARGET_IDS or target in FINAL_TARGET_IDS or set(expected_sources) & FINAL_TARGET_IDS:
                raise RuntimeError(f"Final-target firewall failed for {job_key}")
            immutable = {
                "status": "completed", "job_key": job_key,
                "remote_job_id": remote_id, "config_id": row["config_id"],
                "pseudo_target_city_id": target, "seed": seed,
                "source_city_ids": expected_sources, "source_updates_completed": 12000,
                "fine_tuning_fits": 0, "target_adaptation_fits": 0,
                "evaluation_regime": "zero_shot", "final_target_labels_accessed": False,
            }
            if any(job.get(key) != value for key, value in immutable.items()):
                raise RuntimeError(f"Completed job differs from immutable map/contract: {job_key}")
            if sum(int(value) for value in job.get("source_exposure", {}).values()) != 12000:
                raise RuntimeError(f"Source exposure does not total 12000 for {job_key}")
            if any(job.get("deterministic_settings", {}).get(key) != value for key, value in EXPECTED_DETERMINISM.items()):
                raise RuntimeError(f"Deterministic settings changed for {job_key}")
            metrics = job.get("metrics", {})
            if not isinstance(metrics.get("mae"), (int, float)) or not math.isfinite(float(metrics["mae"])):
                raise RuntimeError(f"Completed job lacks a finite recorded MAE: {job_key}")
            for key, map_field in (
                ("checkpoint", "expected_checkpoint"),
                ("prediction", "expected_prediction"),
                ("prediction_manifest", "expected_prediction_manifest"),
            ):
                item = job.get(key, {})
                if item.get("path") != row[map_field]:
                    raise RuntimeError(f"{key} path changed for {job_key}")
                member = _archive_member_for_repository_path(item["path"])
                digest = sha256_bytes(_member_bytes(handle, index, member))
                if digest != item.get("sha256"):
                    raise RuntimeError(f"{key} hash mismatch for {job_key}")
                scientific_file_hashes.append({"path": member, "sha256": digest})
            prediction_manifest = _member_json(
                handle, index, _archive_member_for_repository_path(job["prediction_manifest"]["path"])
            )
            if prediction_manifest.get("job_key") != job_key:
                raise RuntimeError(f"Prediction-manifest identity changed for {job_key}")
            if prediction_manifest.get("prediction_artifact_sha256") != job["prediction"]["sha256"]:
                raise RuntimeError(f"Prediction-manifest artifact binding changed for {job_key}")
            key_contract = prediction_manifest.get("precommitted_key_contract", {})
            if key_contract.get("city_id") != target or key_contract.get("precommitted_before_evaluation_labels_opened") is not True:
                raise RuntimeError(f"Prediction-key precommitment changed for {job_key}")
            job_digest = sha256_bytes(job_bytes)
            job_record_hashes.append({"job_key": job_key, "sha256": job_digest})
            scientific_file_hashes.append({"path": job_member, "sha256": job_digest})
            seen_remote_ids.add(remote_id)
            seen_job_keys.add(job_key)
            per_config[row["config_id"]] += 1
            per_target[target] += 1
            per_seed[seed] += 1
        if seen_job_keys != set(expected_by_key) or seen_remote_ids != set(range(48)):
            raise RuntimeError("Completed job identity coverage is incomplete")
        if set(per_config.values()) != {12} or set(per_target.values()) != {12} or set(per_seed.values()) != {16}:
            raise RuntimeError("Completed job marginal counts changed")

    return {
        **file_bindings,
        "archive_entry_count": 199,
        "remote_preflight_r2_sha256": sha256_bytes(preflight_bytes),
        "official_validator_sha256": sha256_bytes(validator_bytes),
        "summary_sha256": sha256_bytes(summary_bytes),
        "job_records": 48,
        "checkpoints": 48,
        "predictions": 48,
        "prediction_manifests": 48,
        "job_records_inventory_sha256": canonical_hash(sorted(job_record_hashes, key=lambda item: item["job_key"])),
        "scientific_artifact_inventory_sha256": canonical_hash(sorted(scientific_file_hashes, key=lambda item: item["path"])),
        "source_fits": 48,
        "zero_shot_evaluations": 48,
        "fine_tuning_fits": 0,
        "selected_config_id": "VGRU_H032_D00",
        "selection_rule_applied_exactly_as_frozen": True,
        "final_label_firewall_passed": True,
        "final_target_labels_accessed": False,
        "scientific_computation_performed_by_verifier": False,
    }


def verify() -> dict[str, Any]:
    evidence = collect_existing_evidence()
    completion = read_json_file(COMPLETION_ARTIFACT)
    if completion.get("status") != "CLOSED_FROZEN" or completion.get("stage") != "2B":
        raise RuntimeError("Stage-2B completion artifact is not CLOSED_FROZEN")
    bindings = completion.get("verified_bindings", {})
    for key, value in evidence.items():
        if key.endswith("_sha256") and bindings.get(key) != value:
            raise RuntimeError(f"Completion artifact binding mismatch: {key}")
    if bindings.get("stage2b_completion_verifier_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("Completion artifact is not bound to this verifier")
    chronology = completion.get("execution_chronology", [])
    if len(chronology) != 3 or [item.get("attempt") for item in chronology] != ["R0", "R1", "R2"]:
        raise RuntimeError("Completion chronology changed")
    r0, r1, r2 = chronology
    if r0.get("status") != "ABORTED_PRETRAIN_CACHE_BUILD_IMPLEMENTATION_ERROR":
        raise RuntimeError("Completion R0 status changed")
    if any(r0.get(key) != 0 for key in ("optimizer_updates_completed", "completed_source_fits", "completed_evaluations")):
        raise RuntimeError("Completion R0 counts are not zero")
    if r1.get("status") != "ABORTED_FIRST_FORWARD_MISSING_CUBLAS_WORKSPACE_CONFIG":
        raise RuntimeError("Completion R1 status changed")
    if any(r1.get(key) != 0 for key in ("optimizer_updates_completed", "completed_source_fits", "completed_evaluations")):
        raise RuntimeError("Completion R1 counts are not zero")
    if r2.get("status") != "COMPLETED_VALIDATED" or r2.get("official_validator_status") != "PASS":
        raise RuntimeError("Completion R2 status changed")
    if r2.get("completed_registered_jobs") != 48 or r2.get("completed_source_fits") != 48:
        raise RuntimeError("Completion R2 fit/job counts changed")
    if r2.get("completed_zero_shot_evaluations") != 48 or r2.get("fine_tuning_fits") != 0:
        raise RuntimeError("Completion R2 evaluation/adaptation counts changed")
    if r2.get("process_observations", {}).get("termination_source") != "UNRESOLVED_NOT_PROVEN":
        raise RuntimeError("Completion overclaims the abnormal-process termination source")
    if r2.get("operational_execution_workaround", {}).get("scientific_or_job_definition_change") is not False:
        raise RuntimeError("Completion workaround provenance changed")
    result = completion.get("scientific_result", {})
    if result.get("selected_config_id") != evidence["selected_config_id"]:
        raise RuntimeError("Completion selected configuration changed")
    if result.get("source_fits") != 48 or result.get("zero_shot_evaluations") != 48 or result.get("fine_tuning_fits") != 0:
        raise RuntimeError("Completion workload counts changed")
    if result.get("selection_rule_applied_exactly_as_frozen") is not True or result.get("tie_breaker_required") is not False:
        raise RuntimeError("Completion selection-rule provenance changed")
    expected_completion_summary = [
        {
            "config_id": row["config_id"],
            "equal_fold_mae": float(row["equal_fold_mae"]),
            "equal_fold_mae_4dp": row["equal_fold_mae_4dp"],
            "fold_population_std": float(row["fold_population_std"]),
        }
        for row in EXPECTED_SUMMARY
    ]
    if result.get("configuration_summary") != expected_completion_summary:
        raise RuntimeError("Completion configuration summary differs from the hashed remote summary")
    if result.get("summary_sha256") != evidence["summary_sha256"]:
        raise RuntimeError("Completion summary hash changed")
    if result.get("official_validator_sha256") != evidence["official_validator_sha256"]:
        raise RuntimeError("Completion official-validator hash changed")
    firewall = completion.get("firewall", {})
    if firewall != {
        "final_label_firewall_passed": True,
        "final_target_labels_accessed": False,
        "stage4_authorized": False,
        "final_execution_authorized": False,
    }:
        raise RuntimeError("Completion firewall/scope changed")
    if completion.get("closure_operations", {}).get("scientific_computation_performed") is not False:
        raise RuntimeError("Closure artifact claims scientific computation")
    backup = completion.get("backup", {})
    if backup.get("sha256") != evidence["archive_sha256"] or backup.get("used_as_scientific_selection_input") is not False:
        raise RuntimeError("Completion backup role/hash changed")
    expected_ledger: dict[str, str] = {}
    for line in HASH_LEDGER.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in expected_ledger:
            raise RuntimeError(f"Duplicate closure-ledger path: {relative}")
        expected_ledger[relative] = digest
    for relative, digest in expected_ledger.items():
        path = ROOT.joinpath(*PurePosixPath(relative).parts)
        if sha256_file(path) != digest:
            raise RuntimeError(f"Closure hash-ledger mismatch: {relative}")
    return {
        "status": "PASS",
        "read_only_verifier": True,
        "completion_status": "CLOSED_FROZEN",
        "completion_artifact_sha256": sha256_file(COMPLETION_ARTIFACT),
        "hash_ledger_sha256": sha256_file(HASH_LEDGER),
        "archive_sha256": evidence["archive_sha256"],
        "validated_remote_jobs": 48,
        "source_fits": 48,
        "zero_shot_evaluations": 48,
        "fine_tuning_fits": 0,
        "selected_config_id": "VGRU_H032_D00",
        "selection_rule_applied_exactly_as_frozen": True,
        "final_label_firewall_passed": True,
        "final_target_labels_accessed": False,
        "scientific_computation_performed": False,
        "stage4_or_final_execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Stage-2B completion/provenance verifier")
    parser.add_argument("--evidence-only", action="store_true", help="verify and print existing evidence without requiring closure files")
    args = parser.parse_args()
    result = collect_existing_evidence() if args.evidence_only else verify()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
