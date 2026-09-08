from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

from research.development.stage1_scale_loss import (
    DEVELOPMENT_SEEDS,
    FINAL_TARGETS,
    PSEUDO_TARGETS,
)
from research.development.stage2_graphgru_selection import (
    STAGE2_REGIMES,
    Stage2Configuration,
)


ROOT = Path(__file__).resolve().parents[2]
LOCAL_ABORTED_OUTPUT = ROOT / "research" / "results" / "stage2_graphgru_selection_v2_1"
LOCAL_ABORT_MARKER = LOCAL_ABORTED_OUTPUT / "ABORTED_COMPUTE_MIGRATION.json"
FROZEN_STAGE2_RUN_PLAN = LOCAL_ABORTED_OUTPUT / "stage2_run_plan.json"
REMOTE_OUTPUT = ROOT / "research" / "results" / "stage2_graphgru_selection_v2_1_a40"
REMOTE_CACHE = ROOT / "tmp" / "stage1_scale_loss_cache_v2_1_a40"
REMOTE_GRAPH_CACHE = ROOT / "tmp" / "stage2_graph_grid_cache_v2_1_a40"
REPOSITORY_JOB_MAP = ROOT / "research" / "stage2_remote_job_map.csv"
REMOTE_PROVENANCE = REMOTE_OUTPUT / "REMOTE_EXECUTION_PROVENANCE.json"
REMOTE_PREFLIGHT = REMOTE_OUTPUT / "REMOTE_PREFLIGHT.json"
VALID_EXECUTION_ENVIRONMENT = "UNIVERSITY_A40"
LOCAL_ABORT_STATUS = "ABORTED_COMPUTE_MIGRATION"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_frozen_stage2_plan() -> dict[str, Any]:
    abort = read_json(LOCAL_ABORT_MARKER)
    if abort.get("status") != LOCAL_ABORT_STATUS:
        raise RuntimeError("The local Stage-2 attempt is not sealed as compute-migration aborted")
    if abort.get("execution_snapshot", {}).get("scientific_decision_modified") is not False:
        raise RuntimeError("The local abort marker does not preserve the scientific decision")
    plan = read_json(FROZEN_STAGE2_RUN_PLAN)
    if plan.get("STAGE2_REGIMES") != ["zero_shot"]:
        raise RuntimeError("Frozen Stage-2 run plan is not zero-shot only")
    if tuple(plan.get("development_seeds", [])) != tuple(DEVELOPMENT_SEEDS):
        raise RuntimeError("Frozen Stage-2 development seeds changed")
    target_ids = tuple(int(item["city_id"]) for item in plan.get("pseudo_targets", []))
    if target_ids != tuple(PSEUDO_TARGETS):
        raise RuntimeError("Frozen Stage-2 pseudo-target order changed")
    configurations = plan.get("configurations", [])
    if len(configurations) != 12:
        raise RuntimeError("Frozen Stage-2 run plan must contain exactly 12 configurations")
    parsed = [Stage2Configuration(**item) for item in configurations]
    if len({item.config_id for item in parsed}) != 12:
        raise RuntimeError("Frozen Stage-2 configuration identifiers are not unique")
    counts = plan.get("expected_counts", {})
    required = {
        "configurations": 12,
        "pseudo_target_folds": 4,
        "development_seeds": 3,
        "source_fits": 144,
        "fine_tuning_fits": 0,
        "total_unique_neural_fits": 144,
        "evaluations": 144,
    }
    if any(int(counts.get(key, -1)) != value for key, value in required.items()):
        raise RuntimeError("Frozen Stage-2 run counts differ from the registered 144-fit plan")
    if any(int(item) in FINAL_TARGETS for item in target_ids):
        raise RuntimeError("A sealed final target entered the frozen Stage-2 plan")
    return plan


def build_job_rows(plan: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    plan = load_frozen_stage2_plan() if plan is None else plan
    rows: list[dict[str, Any]] = []
    for configuration in plan["configurations"]:
        for target in plan["pseudo_targets"]:
            for seed in plan["development_seeds"]:
                config_id = str(configuration["config_id"])
                target_id = int(target["city_id"])
                registered_seed = int(seed)
                rows.append(
                    {
                        "job_id": len(rows),
                        "config_id": config_id,
                        "pseudo_target_city_id": target_id,
                        "pseudo_target": str(target["city"]),
                        "seed": registered_seed,
                        "job_key": (
                            f"{config_id.lower()}_target-{target_id}_seed-{registered_seed}"
                        ),
                    }
                )
    identities = {
        (row["config_id"], row["pseudo_target_city_id"], row["seed"])
        for row in rows
    }
    if len(rows) != 144 or len(identities) != 144:
        raise RuntimeError("Remote job map must contain 144/144 unique registered jobs")
    if [row["job_id"] for row in rows] != list(range(144)):
        raise RuntimeError("Remote job identifiers must be the contiguous range 0..143")
    return rows


def job_map_csv_text(rows: Iterable[dict[str, Any]]) -> str:
    handle = io.StringIO()
    fields = (
        "job_id",
        "config_id",
        "pseudo_target_city_id",
        "pseudo_target",
        "seed",
        "job_key",
    )
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row[key] for key in fields})
    return handle.getvalue()


def write_or_verify_job_map(path: Path = REPOSITORY_JOB_MAP) -> Path:
    expected = job_map_csv_text(build_job_rows())
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"Existing remote job map differs from the frozen plan: {path}")
    else:
        path.write_text(expected, encoding="utf-8")
    return path


def validate_task_job_mapping(tasks: list[dict[str, Any]]) -> None:
    rows = build_job_rows()
    if len(tasks) != len(rows):
        raise RuntimeError("Task list and frozen remote job map have different lengths")
    for task, row in zip(tasks, rows, strict=True):
        observed = (
            task["configuration"]["config_id"],
            int(task["target_id"]),
            int(task["seed"]),
        )
        expected = (row["config_id"], row["pseudo_target_city_id"], row["seed"])
        if observed != expected:
            raise RuntimeError(
                f"Task ordering differs from remote job_id {row['job_id']}: {observed} != {expected}"
            )
        task["remote_job_id"] = int(row["job_id"])


def remote_scientific_artifacts(output: Path = REMOTE_OUTPUT) -> list[Path]:
    if not output.exists():
        return []
    patterns = (
        "jobs/*.json",
        "jobs/*.failure.json",
        "stage2_checkpoints/**/*.pt",
        "stage2_predictions/**/*.npz",
        "stage2_prediction_manifests/*.json",
        "stage2_individual_runs.csv",
        "stage2_fold_summary.csv",
        "stage2_config_summary.csv",
        "stage2_decision_manifest.json",
    )
    found: set[Path] = set()
    for pattern in patterns:
        found.update(path.resolve() for path in output.glob(pattern) if path.is_file())
    return sorted(found)


def assert_remote_output_isolation(
    output: Path = REMOTE_OUTPUT, *, require_no_scientific_results: bool
) -> None:
    output = output.resolve()
    local = LOCAL_ABORTED_OUTPUT.resolve()
    if output == local or local in output.parents or output in local.parents:
        raise RuntimeError("Remote Stage-2 output is not isolated from the aborted local output")
    if output != REMOTE_OUTPUT.resolve():
        raise RuntimeError(f"University A40 output must be exactly {REMOTE_OUTPUT}")
    if require_no_scientific_results:
        existing = remote_scientific_artifacts(output)
        if existing:
            raise RuntimeError(
                "Remote output already contains scientific Stage-2 artifacts: "
                + ", ".join(path.name for path in existing[:5])
            )


def expected_provenance() -> dict[str, Any]:
    return {
        "VALID_EXECUTION_ENVIRONMENT": VALID_EXECUTION_ENVIRONMENT,
        "LOCAL_STAGE2_ATTEMPT": LOCAL_ABORT_STATUS,
        "LOCAL_RESULTS_INCLUDED_IN_SELECTION": False,
        "REMOTE_RUN_INITIALIZATION": "FRESH_FROM_INITIALIZATION_ALL_144_SOURCE_FITS",
        "STAGE2_REGIMES": list(STAGE2_REGIMES),
        "EXPECTED_SOURCE_FITS": 144,
        "EXPECTED_FINE_TUNING_FITS": 0,
        "LOCAL_ABORT_MARKER": LOCAL_ABORT_MARKER.relative_to(ROOT).as_posix(),
        "LOCAL_ABORT_MARKER_SHA256": sha256_file(LOCAL_ABORT_MARKER),
        "FROZEN_STAGE2_RUN_PLAN": FROZEN_STAGE2_RUN_PLAN.relative_to(ROOT).as_posix(),
        "FROZEN_STAGE2_RUN_PLAN_SHA256": sha256_file(FROZEN_STAGE2_RUN_PLAN),
        "SCIENTIFIC_RESULTS_PRESENT_AT_PREPARATION": False,
    }


def write_or_verify_provenance(path: Path = REMOTE_PROVENANCE) -> Path:
    expected = expected_provenance()
    if path.is_file():
        existing = read_json(path)
        if existing != expected:
            raise RuntimeError("Remote execution provenance marker differs from the frozen source")
    else:
        write_json_atomic(path, expected)
    return path


def nvidia_smi_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("nvidia-smi could not verify the university GPU") from exc
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one visible A40 GPU, found {len(rows)}")
    values = [item.strip() for item in rows[0].split(",")]
    if len(values) != 5:
        raise RuntimeError("Unexpected nvidia-smi query output")
    return {
        "gpu_name": values[0],
        "driver_version": values[1],
        "memory_total_mib": int(values[2]),
        "gpu_utilization_percent": int(values[3]),
        "memory_used_mib": int(values[4]),
    }


def require_passing_remote_preflight(path: Path = REMOTE_PREFLIGHT) -> dict[str, Any]:
    value = read_json(path)
    if value.get("status") != "PASS":
        raise RuntimeError("Remote Stage-2 preflight has not passed")
    if value.get("VALID_EXECUTION_ENVIRONMENT") != VALID_EXECUTION_ENVIRONMENT:
        raise RuntimeError("Remote preflight was not produced on UNIVERSITY_A40")
    if value.get("LOCAL_RESULTS_INCLUDED_IN_SELECTION") is not False:
        raise RuntimeError("Remote preflight did not exclude local aborted results")
    if value.get("scientific_results_present") is not False:
        raise RuntimeError("Remote preflight root was not scientifically empty")
    if value.get("frozen_run_plan_sha256") != sha256_file(FROZEN_STAGE2_RUN_PLAN):
        raise RuntimeError("Frozen Stage-2 plan changed after remote preflight")
    if value.get("job_map_sha256") != sha256_file(REPOSITORY_JOB_MAP):
        raise RuntimeError("Remote job map changed after remote preflight")
    for relative, expected in value.get("migration_source_sha256", {}).items():
        source = ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(f"Migration source changed after remote preflight: {relative}")
    return value
