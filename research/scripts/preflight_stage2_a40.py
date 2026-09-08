from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from research.development.stage2_graphgru_selection import (
    Stage2ArtifactRegistry,
    build_stage2_graph_cache,
    configure_deterministic_device,
    registered_stage2_grid,
    runtime_environment,
    verify_stage2_artifacts,
)
from research.remote.stage2_a40 import (
    FROZEN_STAGE2_RUN_PLAN,
    LOCAL_ABORTED_OUTPUT,
    REMOTE_GRAPH_CACHE,
    REMOTE_OUTPUT,
    REMOTE_PREFLIGHT,
    REPOSITORY_JOB_MAP,
    VALID_EXECUTION_ENVIRONMENT,
    assert_remote_output_isolation,
    build_job_rows,
    load_frozen_stage2_plan,
    nvidia_smi_snapshot,
    read_json,
    remote_scientific_artifacts,
    sha256_file,
    write_json_atomic,
    write_or_verify_job_map,
    write_or_verify_provenance,
)
from research.scripts.run_stage2_graphgru_selection import (
    _source_hashes,
    verify_governing_contract,
)
from research.training.checkpointing import neural_code_hash


EXPECTED_PYTHON = "3.10.9"
EXPECTED_TORCH = "2.0.1+cu118"
EXPECTED_TORCH_CUDA = "11.8"
EXPECTED_GPU = "NVIDIA A40"
EXPECTED_DRIVER = "538.78"


def _migration_source_hashes() -> dict[str, str]:
    relatives = (
        "research/development/stage1_scale_loss.py",
        "research/development/stage2_graphgru_selection.py",
        "research/remote/stage2_a40.py",
        "research/scripts/preflight_stage2_a40.py",
        "research/scripts/run_stage2_graphgru_selection.py",
        "research/scripts/run_stage2_remote_a40.py",
        "research/stage2_remote_job_map.csv",
    )
    return {relative: sha256_file(ROOT / relative) for relative in relatives}


def _verify_graph_reproduction(
    graph_metadata: dict[str, dict[str, Any]]
) -> dict[str, str]:
    local_manifest = read_json(LOCAL_ABORTED_OUTPUT / "stage2_run_manifest.json")
    expected = local_manifest.get("graph_cache_metadata", {})
    if set(expected) != set(graph_metadata):
        raise RuntimeError("Remote graph-cache keys differ from the frozen local graph evidence")
    hashes: dict[str, str] = {}
    for key, item in graph_metadata.items():
        observed_graph = item["graph_config"]
        expected_graph = expected[key]["graph_config"]
        for field in (
            "roster_hash_sha256",
            "coordinate_hash_sha256",
            "adjacency_hash_sha256",
        ):
            if observed_graph[field] != expected_graph[field]:
                raise RuntimeError(
                    f"Registered graph/station identity field {field} did not reproduce for {key}"
                )
        hashes[key] = observed_graph["adjacency_hash_sha256"]
    if len(hashes) != 24:
        raise RuntimeError("Remote preflight must reproduce exactly 24 city-k graphs")
    return hashes


def run_preflight() -> dict[str, Any]:
    assert_remote_output_isolation(REMOTE_OUTPUT, require_no_scientific_results=True)
    write_or_verify_provenance()
    job_map = write_or_verify_job_map(REPOSITORY_JOB_MAP)
    remote_job_map = write_or_verify_job_map(REMOTE_OUTPUT / "stage2_remote_job_map.csv")
    frozen = load_frozen_stage2_plan()

    device, deterministic = configure_deterministic_device(17, "cuda")
    environment = runtime_environment(device)
    if environment["python_version"] != EXPECTED_PYTHON:
        raise RuntimeError(
            f"Expected Python {EXPECTED_PYTHON}, got {environment['python_version']}"
        )
    if str(environment["torch_version"]) != EXPECTED_TORCH:
        raise RuntimeError(f"Expected torch {EXPECTED_TORCH}, got {environment['torch_version']}")
    if environment["torch_cuda_version"] != EXPECTED_TORCH_CUDA:
        raise RuntimeError(
            f"Expected torch CUDA {EXPECTED_TORCH_CUDA}, got {environment['torch_cuda_version']}"
        )
    if environment["cuda_available"] is not True:
        raise RuntimeError("CUDA is unavailable")
    if environment["gpu_name"] != EXPECTED_GPU:
        raise RuntimeError(f"Expected {EXPECTED_GPU}, got {environment['gpu_name']}")
    smi = nvidia_smi_snapshot()
    if smi["gpu_name"] != EXPECTED_GPU:
        raise RuntimeError(f"nvidia-smi did not report {EXPECTED_GPU}")
    if smi["driver_version"] != EXPECTED_DRIVER:
        raise RuntimeError(
            f"Expected NVIDIA driver {EXPECTED_DRIVER}, got {smi['driver_version']}"
        )

    contract, grid = verify_governing_contract()
    frozen_grid = [item for item in frozen["configurations"]]
    if [item.as_dict() for item in grid] != frozen_grid:
        raise RuntimeError("Current Stage-2 grid differs from the frozen run plan")
    if len(build_job_rows(frozen)) != 144:
        raise RuntimeError("Frozen remote job map does not contain exactly 144 fits")

    registry = Stage2ArtifactRegistry()
    firewall = verify_stage2_artifacts(registry)
    if firewall["sealed_final_label_content_accessed"] is not False:
        raise RuntimeError("Final-label firewall failed")
    graph_metadata = build_stage2_graph_cache(
        registry, REMOTE_GRAPH_CACHE, grid, log=None
    )
    graph_hashes = _verify_graph_reproduction(graph_metadata)
    scientific = remote_scientific_artifacts(REMOTE_OUTPUT)
    if scientific:
        raise RuntimeError("Scientific Stage-2 artifacts exist before remote execution")

    return {
        "status": "PASS",
        "VALID_EXECUTION_ENVIRONMENT": VALID_EXECUTION_ENVIRONMENT,
        "LOCAL_STAGE2_ATTEMPT": "ABORTED_COMPUTE_MIGRATION",
        "LOCAL_RESULTS_INCLUDED_IN_SELECTION": False,
        "scientific_results_present": False,
        "training_started": False,
        "environment": environment,
        "nvidia_smi": smi,
        "deterministic_settings": deterministic,
        "dataset_artifact_verification": firewall,
        "governing_contract_passed": bool(contract["passed"]),
        "stage1_decision": contract["stage1_decision"],
        "stage2_configuration_count": len(grid),
        "stage2_source_fit_count": 144,
        "stage2_fine_tuning_fit_count": 0,
        "stage2_regimes": ["zero_shot"],
        "frozen_run_plan": FROZEN_STAGE2_RUN_PLAN.relative_to(ROOT).as_posix(),
        "frozen_run_plan_sha256": sha256_file(FROZEN_STAGE2_RUN_PLAN),
        "job_map": job_map.relative_to(ROOT).as_posix(),
        "job_map_sha256": sha256_file(job_map),
        "remote_job_map": remote_job_map.relative_to(ROOT).as_posix(),
        "remote_job_map_sha256": sha256_file(remote_job_map),
        "graph_hashes": graph_hashes,
        "neural_code_sha256": neural_code_hash(),
        "stage2_source_code_sha256": _source_hashes(),
        "migration_source_sha256": _migration_source_hashes(),
        "final_label_firewall_passed": True,
        "remote_output": REMOTE_OUTPUT.relative_to(ROOT).as_posix(),
        "remote_output_isolated": True,
    }


def main() -> int:
    result = run_preflight()
    write_json_atomic(REMOTE_PREFLIGHT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
