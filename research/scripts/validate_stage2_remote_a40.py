from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.development.stage2_graphgru_selection import Stage2ArtifactRegistry
from research.remote.stage2_a40 import (
    REMOTE_CACHE,
    REMOTE_GRAPH_CACHE,
    REMOTE_OUTPUT,
    read_json,
    sha256_file,
    validate_task_job_mapping,
    write_json_atomic,
)
from research.scripts.run_stage2_graphgru_selection import (
    _read_completed_job,
    build_tasks,
    verify_governing_contract,
)


def main() -> int:
    manifest_path = REMOTE_OUTPUT / "stage2_run_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "valid_and_frozen":
        raise RuntimeError("Remote Stage-2 run is not valid_and_frozen")
    if manifest.get("execution_environment") != "UNIVERSITY_A40":
        raise RuntimeError("Completed manifest is not from UNIVERSITY_A40")
    if int(manifest.get("completed_source_fits", -1)) != 144:
        raise RuntimeError("Remote Stage-2 did not complete 144 source fits")
    if int(manifest.get("completed_fine_tuning_fits", -1)) != 0:
        raise RuntimeError("Remote Stage-2 unexpectedly contains fine-tuning fits")

    contract, grid = verify_governing_contract()
    registry = Stage2ArtifactRegistry()
    plan_hash = sha256_file(REMOTE_OUTPUT / "stage2_run_plan.json")
    requested_device = str(manifest.get("runtime", {}).get("device_requested", "cuda"))
    tasks = build_tasks(
        REMOTE_OUTPUT,
        REMOTE_CACHE,
        REMOTE_GRAPH_CACHE,
        registry,
        grid,
        plan_hash,
        device=requested_device,
        execution_environment="UNIVERSITY_A40",
    )
    validate_task_job_mapping(tasks)
    completed = []
    for task in tasks:
        config_id = task["configuration"]["config_id"].lower()
        job_id = f"{config_id}_target-{task['target_id']}_seed-{task['seed']}"
        value = _read_completed_job(
            REMOTE_OUTPUT / "jobs" / f"{job_id}.json", task, REMOTE_OUTPUT
        )
        if value is None:
            raise RuntimeError(f"Missing completed remote job: {job_id}")
        completed.append(value)
    if len(completed) != 144:
        raise RuntimeError("Independent remote job validation did not reach 144/144")

    decision_path = REMOTE_OUTPUT / "stage2_decision_manifest.json"
    decision = read_json(decision_path)
    if decision.get("execution_environment") != "UNIVERSITY_A40":
        raise RuntimeError("Decision manifest does not identify UNIVERSITY_A40")
    if decision.get("local_results_included_in_selection") is not False:
        raise RuntimeError("Decision manifest does not exclude local aborted results")
    if decision.get("sealed_final_evaluation_labels_accessed") is not False:
        raise RuntimeError("Final-label firewall failed in the decision manifest")

    result = {
        "status": "PASS",
        "validated_remote_jobs": len(completed),
        "execution_environment": "UNIVERSITY_A40",
        "local_stage2_attempt": "ABORTED_COMPUTE_MIGRATION",
        "local_results_included_in_selection": False,
        "fine_tuning_fits": 0,
        "final_label_firewall_passed": True,
        "run_manifest_sha256": sha256_file(manifest_path),
        "decision_manifest_sha256": sha256_file(decision_path),
        "contract_expected_counts": contract["expected_counts"],
    }
    output = REMOTE_OUTPUT / "REMOTE_POSTRUN_VALIDATION.json"
    write_json_atomic(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
