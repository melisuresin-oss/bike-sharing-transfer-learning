from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
FROZEN_SEAL_DIR = ROOT / "research" / "results" / "stage2b_vanilla_fairness_v2_1"
FROZEN_MANIFEST = FROZEN_SEAL_DIR / "stage2b_fairness_manifest.json"
FROZEN_RUN_PLAN = FROZEN_SEAL_DIR / "stage2b_run_plan.csv"
PACKAGE_ROOT = ROOT / "research" / "results" / "stage2b_execution_package_v2_1_r2"
REMOTE_OUTPUT = ROOT / "research" / "results" / "stage2b_vanilla_fairness_v2_1_a40"
REMOTE_CACHE = ROOT / "tmp" / "stage2b_vanilla_feature_cache_v2_1_a40"
JOB_MAP = PACKAGE_ROOT / "stage2b_remote_job_map.csv"
PACKAGE_PREFLIGHT = PACKAGE_ROOT / "STAGE2B_PACKAGE_PREFLIGHT.json"
PACKAGE_MANIFEST = PACKAGE_ROOT / "stage2b_execution_package_manifest.json"
AUTHORIZATION = PACKAGE_ROOT / "STAGE2B_EXECUTION_AUTHORIZATION.json"
REMOTE_PREFLIGHT = REMOTE_OUTPUT / "REMOTE_PREFLIGHT_R2.json"
INCIDENT_RECORD = PACKAGE_ROOT / "REMOTE_STAGE2B_R1_ABORTED_FIRST_FORWARD.json"

FROZEN_SEAL_SHA256 = "6b3f7e4936e201c91023145a73e36b4a122e8a97305ef408bc14a6c727c6259f"
EXECUTION_ENVIRONMENT = "UNIVERSITY_A40"
DEVELOPMENT_CITIES = {
    129: "Dortmund",
    194: "Heidelberg",
    438: "Marburg",
    467: "Gießen",
    476: "Cardiff",
    532: "Bilbao",
    619: "Freiburg",
    658: "Göteborg",
}
DEVELOPMENT_STATION_FILES = {
    129: "city_129_dortmund_stations.parquet",
    194: "city_194_heidelberg_stations.parquet",
    438: "city_438_marburg_stations.parquet",
    467: "city_467_gießen_stations.parquet",
    476: "city_476_cardiff_stations.parquet",
    532: "city_532_bilbao_stations.parquet",
    619: "city_619_freiburg_stations.parquet",
    658: "city_658_göteborg_stations.parquet",
}
PSEUDO_TARGETS = (532, 476, 619, 658)
SEEDS = (17, 29, 43)
FINAL_TARGETS = (195, 199, 237, 617)
CONFIGURATIONS = (
    ("VGRU_H032_D00", 32, 0.0),
    ("VGRU_H032_D10", 32, 0.1),
    ("VGRU_H064_D00", 64, 0.0),
    ("VGRU_H064_D10", 64, 0.1),
)
H0 = "2022-08-29T05:00:00Z"
HD = "2023-02-15T22:00:00Z"
HF = "2023-04-16T22:00:00Z"
SOURCE_UPDATES = 12_000
BATCH_SIZE = 16
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

REQUIRED_A40_RUNTIME = {
    "python": "3.10",
    "torch": "2.0.1+cu118",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "duckdb": "1.5.5",
    "psutil": "5.9.8",
    "gpu": "NVIDIA A40",
}


def establish_cublas_workspace_config() -> str:
    """Set the registered deterministic CuBLAS workspace before torch is imported.

    An already-defined conflicting value is rejected so a remote shell cannot
    silently weaken or alter the sealed runtime contract.
    """
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    elif current != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG conflicts with the sealed Stage-2B value: "
            f"expected {CUBLAS_WORKSPACE_CONFIG!r}, observed {current!r}"
        )
    return os.environ["CUBLAS_WORKSPACE_CONFIG"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_json_once_or_verify(path: Path, value: dict[str, Any]) -> None:
    """Append-only JSON creation: identical reruns verify; changes are rejected."""
    if path.is_file():
        if read_json(path) != value:
            raise RuntimeError(f"Append-only artifact differs from existing file: {path}")
        return
    write_json_atomic(path, value)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def load_and_verify_frozen_seal() -> dict[str, Any]:
    if sha256_file(FROZEN_MANIFEST) != FROZEN_SEAL_SHA256:
        raise RuntimeError("Frozen Stage-2B fairness manifest hash mismatch")
    manifest = read_json(FROZEN_MANIFEST)
    if manifest.get("status") != "FROZEN_VERIFIED":
        raise RuntimeError("Stage-2B fairness protocol is not FROZEN_VERIFIED")
    if manifest.get("frozen_stage1_decision") != "LOG1P_TARGET_MAE":
        raise RuntimeError("Stage-1 decision changed")
    if manifest.get("frozen_stage2_decision_unchanged") != "GGRU_K04_H032_D00":
        raise RuntimeError("Frozen Stage-2 decision changed")
    workload = manifest.get("workload", {})
    expected = {"source_fits": 48, "target_adaptation_fits": 0, "zero_shot_evaluations": 48}
    if any(int(workload.get(key, -1)) != value for key, value in expected.items()):
        raise RuntimeError("Frozen Stage-2B workload changed")
    return manifest


def split_roles() -> dict[str, Any]:
    split_path = ROOT / "processed" / "protocol_v2_1" / "manifests" / "split_manifest.json"
    split = read_json(split_path)
    roles = split.get("city_roles", [])
    development = {int(row["city_id"]): str(row["city"]) for row in roles if row["development_source"]}
    pseudo = tuple(int(row["city_id"]) for row in roles if row["pseudo_target"])
    final = tuple(int(row["city_id"]) for row in roles if row["final_target"])
    if development != DEVELOPMENT_CITIES or set(pseudo) != set(PSEUDO_TARGETS):
        raise RuntimeError("Authoritative development roles differ from the frozen Stage-2B contract")
    if set(final) != set(FINAL_TARGETS):
        raise RuntimeError("Authoritative final targets differ from the frozen firewall")
    boundaries = split.get("boundaries", {})
    if boundaries.get("H0") != H0 or boundaries.get("HD") != HD or boundaries.get("HF") != HF:
        raise RuntimeError("Authoritative development boundaries changed")
    return {"split": split, "development": development, "pseudo": pseudo, "final": final}


def station_entries() -> dict[int, dict[str, Any]]:
    path = ROOT / "processed" / "protocol_v2_1" / "station_manifests" / "station_manifest.json"
    manifest = read_json(path)
    entries = {int(item["city_id"]): item for item in manifest["cities"]}
    missing = set(DEVELOPMENT_CITIES) - set(entries)
    if missing:
        raise RuntimeError(f"Missing development station manifests: {sorted(missing)}")
    return entries


def deterministic_seed(label: str, target_id: int, seed: int) -> int:
    # This is the already-registered Stage-2 development seed mapping used by the seal.
    payload = f"stage2-v2.1|{label}|{int(target_id)}|{int(seed)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def source_ids_for(target_id: int) -> tuple[int, ...]:
    result = tuple(city_id for city_id in DEVELOPMENT_CITIES if city_id != int(target_id))
    if len(result) != 7 or target_id in result or set(result) & set(FINAL_TARGETS):
        raise RuntimeError("Pseudo-target source exclusion failed")
    return result


def prediction_key_contract(target_id: int) -> dict[str, Any]:
    return {
        "columns": ["city_id", "station_id", "timestamp_utc"],
        "city_id": int(target_id),
        "interval": {"start": HD, "end": HF, "semantics": "UTC left-closed right-open"},
        "mask": "coverage_observed_12h == true",
        "order": ["timestamp_utc", "station_id"],
        "duplicates_allowed": False,
        "precommitted_before_evaluation_labels_opened": True,
    }


def build_job_rows() -> list[dict[str, Any]]:
    load_and_verify_frozen_seal()
    split_roles()
    entries = station_entries()
    panel = ROOT / "processed" / "protocol_v2_1" / "development" / "development_panel.parquet"
    feature = ROOT / "processed" / "protocol_v2_1" / "manifests" / "feature_manifest.json"
    protocol_manifest = ROOT / "processed" / "protocol_v2_1" / "PROTOCOL_DATASET_MANIFEST.json"
    panel_hash, feature_hash, protocol_hash = map(sha256_file, (panel, feature, protocol_manifest))
    rows: list[dict[str, Any]] = []
    for config_id, hidden_size, dropout in CONFIGURATIONS:
        for target_id in PSEUDO_TARGETS:
            sources = source_ids_for(target_id)
            source_rosters = {str(city_id): entries[city_id]["roster_hash_sha256"] for city_id in sources}
            source_roster_bundle = canonical_hash([
                {"city_id": city_id, "sha256": entries[city_id]["roster_hash_sha256"]}
                for city_id in sorted(sources)
            ])
            target_roster = str(entries[target_id]["roster_hash_sha256"])
            cache_signatures = {
                str(city_id): canonical_hash({
                    "stage": "2B",
                    "city_id": city_id,
                    "development_panel_sha256": panel_hash,
                    "feature_manifest_sha256": feature_hash,
                    "roster_hash_sha256": entries[city_id]["roster_hash_sha256"],
                    "model_family": "station_independent_vanilla_gru",
                })
                for city_id in DEVELOPMENT_CITIES
            }
            for seed in SEEDS:
                job_key = f"{config_id.lower()}_target-{target_id}_seed-{seed}"
                rows.append({
                    "remote_job_id": len(rows),
                    "job_key": job_key,
                    "config_id": config_id,
                    "model_family": "genuine_vanilla_gru",
                    "pseudo_target_city_id": target_id,
                    "pseudo_target": DEVELOPMENT_CITIES[target_id],
                    "source_city_ids": ";".join(map(str, sources)),
                    "source_cities": ";".join(DEVELOPMENT_CITIES[x] for x in sources),
                    "seed": seed,
                    "initialization_seed": seed,
                    "source_city_schedule_seed": deterministic_seed("source-city-schedule", target_id, seed),
                    "source_anchor_sampling_seed": deterministic_seed("source-anchor-sampling", target_id, seed),
                    "hidden_size": hidden_size,
                    "dropout": dropout,
                    "input_size": 2,
                    "context_size": 10,
                    "recurrent_layers": 1,
                    "output_mode": "log1p_target",
                    "source_updates": SOURCE_UPDATES,
                    "batch_size_city_hours": BATCH_SIZE,
                    "optimizer": "AdamW",
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "betas": "0.9;0.999",
                    "epsilon": 1e-8,
                    "global_gradient_clip": 1.0,
                    "early_stopping": False,
                    "checkpoint": "final_update",
                    "fine_tuning_fits": 0,
                    "evaluation_regime": "zero_shot",
                    "development_panel_sha256": panel_hash,
                    "feature_manifest_sha256": feature_hash,
                    "protocol_dataset_manifest_sha256": protocol_hash,
                    "source_roster_bundle_sha256": source_roster_bundle,
                    "target_roster_sha256": target_roster,
                    "development_rosters_json": json.dumps(
                        {str(city_id): entries[city_id]["roster_hash_sha256"] for city_id in DEVELOPMENT_CITIES},
                        sort_keys=True, separators=(",", ":"),
                    ),
                    "cache_signatures_json": json.dumps(cache_signatures, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    "evaluation_start_utc": HD,
                    "evaluation_end_utc": HF,
                    "prediction_key_contract_json": json.dumps(prediction_key_contract(target_id), sort_keys=True, separators=(",", ":")),
                    "expected_checkpoint": f"research/results/stage2b_vanilla_fairness_v2_1_a40/checkpoints/{job_key}.pt",
                    "expected_prediction": f"research/results/stage2b_vanilla_fairness_v2_1_a40/predictions/{job_key}.npz",
                    "expected_prediction_manifest": f"research/results/stage2b_vanilla_fairness_v2_1_a40/prediction_manifests/{job_key}.json",
                    "expected_job_result": f"research/results/stage2b_vanilla_fairness_v2_1_a40/jobs/{job_key}.json",
                })
    validate_job_rows(rows)
    _validate_against_frozen_run_plan(rows)
    return rows


def _validate_against_frozen_run_plan(rows: list[dict[str, Any]]) -> None:
    with FROZEN_RUN_PLAN.open("r", encoding="utf-8", newline="") as handle:
        frozen_rows = list(csv.DictReader(handle))
    if len(frozen_rows) != 48:
        raise RuntimeError("Frozen Stage-2B run plan no longer contains 48 rows")
    by_key = {str(row["job_id"]): row for row in frozen_rows}
    if len(by_key) != 48:
        raise RuntimeError("Frozen Stage-2B run plan contains duplicate job ids")
    exact_fields = (
        "config_id", "pseudo_target_city_id", "source_city_ids", "seed",
        "initialization_seed", "source_city_schedule_seed", "source_anchor_sampling_seed",
        "hidden_size", "dropout", "source_updates", "optimizer", "learning_rate",
        "weight_decay", "global_gradient_clip", "early_stopping",
    )
    for row in rows:
        frozen = by_key.get(str(row["job_key"]))
        if frozen is None:
            raise RuntimeError(f"Generated job absent from frozen plan: {row['job_key']}")
        for field in exact_fields:
            left = str(row[field]).lower()
            right = str(frozen[field]).lower()
            if left != right:
                raise RuntimeError(f"Generated {field} differs from frozen run plan for {row['job_key']}: {left} != {right}")


def validate_job_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    identities = {(r["config_id"], int(r["pseudo_target_city_id"]), int(r["seed"])) for r in rows}
    if len(rows) != 48 or len(identities) != 48:
        raise RuntimeError("Stage-2B job map must contain 48 unique jobs")
    if [int(r["remote_job_id"]) for r in rows] != list(range(48)):
        raise RuntimeError("Stage-2B remote job ids must be contiguous 0..47")
    per_config = Counter(str(r["config_id"]) for r in rows)
    per_target = Counter(int(r["pseudo_target_city_id"]) for r in rows)
    per_seed = Counter(int(r["seed"]) for r in rows)
    if set(per_config.values()) != {12} or set(per_target.values()) != {12} or set(per_seed.values()) != {16}:
        raise RuntimeError("Stage-2B marginal job counts are incorrect")
    if any(int(r["pseudo_target_city_id"]) in FINAL_TARGETS for r in rows):
        raise RuntimeError("A final target entered the Stage-2B job map")
    for row in rows:
        sources = tuple(int(x) for x in str(row["source_city_ids"]).split(";"))
        if sources != source_ids_for(int(row["pseudo_target_city_id"])):
            raise RuntimeError("Source membership changed")
        if not str(row["expected_job_result"]).startswith("research/results/stage2b_vanilla_fairness_v2_1_a40/"):
            raise RuntimeError("Job path escapes the isolated Stage-2B output root")
    return {
        "jobs": 48,
        "per_configuration": {str(key): value for key, value in per_config.items()},
        "per_target": {str(key): value for key, value in per_target.items()},
        "per_seed": {str(key): value for key, value in per_seed.items()},
    }


def job_map_csv_text(rows: Iterable[dict[str, Any]]) -> str:
    rows = list(rows)
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def write_or_verify_job_map(path: Path = JOB_MAP) -> Path:
    text = job_map_csv_text(build_job_rows())
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError("Existing Stage-2B job map differs from the frozen contract")
    if not path.exists():
        path.write_text(text, encoding="utf-8")
    return path


def read_job_map(path: Path = JOB_MAP) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    validate_job_rows(rows)
    return rows


def scientific_artifacts(root: Path = REMOTE_OUTPUT) -> list[Path]:
    patterns = ("jobs/*.json", "checkpoints/*.pt", "predictions/*.npz", "prediction_manifests/*.json", "stage2b_*summary*.csv", "stage2b_decision_manifest.json")
    return sorted({p.resolve() for pattern in patterns for p in root.glob(pattern) if p.is_file()}) if root.exists() else []


def assert_output_isolation(*, require_empty: bool) -> None:
    allowed = REMOTE_OUTPUT.resolve()
    forbidden = [
        ROOT / "research" / "results" / "stage2_graphgru_selection_v2_1",
        ROOT / "research" / "results" / "stage2_graphgru_selection_v2_1_a40",
        ROOT / "research" / "results" / "stage4_dann_development_v2_1",
    ]
    if any(allowed == x.resolve() or allowed in x.resolve().parents or x.resolve() in allowed.parents for x in forbidden):
        raise RuntimeError("Stage-2B output is not isolated")
    if require_empty and scientific_artifacts():
        raise RuntimeError("Stage-2B scientific artifacts exist before authorization/preflight")


def assert_final_firewall_path(path: Path) -> None:
    resolved = path.resolve()
    allowed_exact = {
        (ROOT / "processed/protocol_v2_1/development/development_panel.parquet").resolve(),
        (ROOT / "processed/protocol_v2_1/manifests/split_manifest.json").resolve(),
        (ROOT / "processed/protocol_v2_1/manifests/feature_manifest.json").resolve(),
        JOB_MAP.resolve(), AUTHORIZATION.resolve(),
        *{
            (ROOT / "processed/protocol_v2_1/station_manifests" / filename).resolve()
            for filename in DEVELOPMENT_STATION_FILES.values()
        },
    }
    if resolved not in allowed_exact:
        raise PermissionError(f"Stage-2B firewall forbids content access: {path}")
    lower = resolved.as_posix().lower()
    forbidden_tokens = ("sealed_final", "final_target", "final_evaluation", "stage4", "dann")
    if any(token in lower for token in forbidden_tokens):
        raise PermissionError(f"Stage-2B firewall rejected final/Stage-4 path: {path}")


def load_authorization() -> dict[str, Any]:
    auth = read_json(AUTHORIZATION)
    if auth.get("status") != "AUTHORIZED_CONDITIONAL_UNIVERSITY_A40":
        raise RuntimeError("Stage-2B execution authorization is absent")
    if auth.get("local_execution_authorized") is not False or auth.get("training_started") is not False:
        raise RuntimeError("Stage-2B authorization provenance is invalid")
    if auth.get("frozen_stage2b_manifest_sha256") != FROZEN_SEAL_SHA256:
        raise RuntimeError("Authorization is not bound to the frozen Stage-2B seal")
    if auth.get("package_revision") != "R2_CUBLAS_DETERMINISM_FIX":
        raise RuntimeError("Authorization is not the corrected R2 package")
    if sha256_file(JOB_MAP) != auth.get("job_map_sha256"):
        raise RuntimeError("Authorized Stage-2B job map hash mismatch")
    return auth


def assert_remote_runtime(preflight: dict[str, Any]) -> None:
    if preflight.get("status") != "PASS" or preflight.get("execution_environment") != EXECUTION_ENVIRONMENT:
        raise RuntimeError("Strict UNIVERSITY_A40 preflight has not passed")
    if preflight.get("final_label_firewall_passed") is not True:
        raise RuntimeError("Final-label firewall did not pass")
    if preflight.get("scientific_results_present") is not False or preflight.get("training_started") is not False:
        raise RuntimeError("Remote preflight was not performed before scientific execution")
    if preflight.get("runtime") != REQUIRED_A40_RUNTIME:
        raise RuntimeError("Remote runtime does not match the frozen A40 environment")
    settings = preflight.get("deterministic_settings", {})
    expected_settings = {
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
        "deterministic_algorithms": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cudnn_allow_tf32": False,
    }
    if any(settings.get(key) != value for key, value in expected_settings.items()):
        raise RuntimeError("Strict remote preflight did not verify the sealed deterministic settings")
    fixture = preflight.get("cuda_determinism_fixture", {})
    if fixture.get("passed") is not True or fixture.get("operation") != "torch.matmul":
        raise RuntimeError("Strict remote preflight did not pass the CUDA CuBLAS fixture")
    expected_bindings = {
        "execution_package_manifest_sha256": sha256_file(PACKAGE_MANIFEST),
        "execution_authorization_sha256": sha256_file(AUTHORIZATION),
        "incident_record_sha256": sha256_file(INCIDENT_RECORD),
        "stage2b_implementation_sha256": sha256_file(
            ROOT / "research/development/stage2b_vanilla_execution.py"
        ),
    }
    if any(preflight.get(key) != value for key, value in expected_bindings.items()):
        raise RuntimeError("Remote preflight is not bound to the corrected R2 package")


def validate_resume_job(job: dict[str, Any], row: dict[str, str]) -> bool:
    if job.get("status") != "completed":
        return False
    immutable = {
        "job_key": row["job_key"],
        "config_id": row["config_id"],
        "pseudo_target_city_id": int(row["pseudo_target_city_id"]),
        "seed": int(row["seed"]),
        "source_updates_completed": SOURCE_UPDATES,
        "fine_tuning_fits": 0,
        "evaluation_regime": "zero_shot",
    }
    if any(job.get(key) != value for key, value in immutable.items()):
        return False
    for key in ("checkpoint", "prediction", "prediction_manifest"):
        item = job.get(key, {})
        path = ROOT / str(item.get("path", ""))
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            return False
    return True
