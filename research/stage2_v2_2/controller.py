"""Detached shared operational scheduler for frozen Stage-2/Stage-2B jobs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from typing import Any

from research.stage2_v2_2 import a40
from research.stage2_v2_2 import core as c


ROOT = Path(__file__).resolve().parents[2]


def _interleaved_jobs() -> list[tuple[str, dict[str, Any]]]:
    stage2 = list(c.read_json(c.STAGE2_JOB_MAP)["jobs"])
    stage2b = list(c.read_json(c.STAGE2B_JOB_MAP)["jobs"])
    queue: list[tuple[str, dict[str, Any]]] = []
    for index, baseline in enumerate(stage2b):
        queue.extend(("stage2", job) for job in stage2[index * 3:(index + 1) * 3])
        queue.append(("stage2b", baseline))
    if len(queue) != 192 or len({(stage, job["job_id"]) for stage, job in queue}) != 192:
        raise RuntimeError("Shared 3:1 Stage-2/Stage-2B queue is incomplete or duplicated")
    return queue


def _worker_command(stage: str, job_id: str, manifest_sha256: str,
                    preflight_path: str, workers: int) -> list[str]:
    return [
        sys.executable, "-X", "utf8", "-B", "-m", "research.stage2_v2_2.a40", "job",
        "--manifest-sha256", manifest_sha256,
        "--workers", str(workers),
        "--stage", stage,
        "--job-id", job_id,
        "--preflight", preflight_path,
    ]


def _postrun_command(mode: str, stage: str, manifest_sha256: str, workers: int,
                     *, record: bool = False) -> list[str]:
    command = [
        sys.executable, "-X", "utf8", "-B", "-m", "research.stage2_v2_2.a40", mode,
        "--manifest-sha256", manifest_sha256, "--workers", str(workers), "--stage", stage,
    ]
    if record:
        command.append("--record")
    return command


def launch(manifest_sha256: str, workers: int) -> None:
    if not 1 <= workers <= a40.MAX_WORKERS:
        raise RuntimeError("Operational workers must be in 1..6")
    a40.verify_package(manifest_sha256)
    preflight = a40.preflight(manifest_sha256, workers)
    preflight_path = preflight["artifact"]["path"]
    queue = _interleaved_jobs()
    run_id = uuid.uuid4().hex
    run_root = c.JOINT_OUT + "controller_runs/" + run_id + "/"
    a40.write_json_atomic(run_root + "runtime_provenance.json", {
        "status": "RUNNING",
        "protocol_version": "2.2",
        "execution_environment": c.ENVIRONMENT,
        "run_id": run_id,
        "coordinator_pid": os.getpid(),
        "bundle_manifest_sha256": manifest_sha256,
        "stage1_decision_sha256": c.STAGE1_DECISION_SHA256,
        "stage2_job_map_sha256": c.sha256_file(c.STAGE2_JOB_MAP),
        "stage2b_job_map_sha256": c.sha256_file(c.STAGE2B_JOB_MAP),
        "operational_worker_count": workers,
        "worker_count_is_scientific_hyperparameter": False,
        "threads_per_worker": a40.THREADS_PER_WORKER,
        "one_fresh_python_process_per_immutable_fit": True,
        "queue_policy": "three Stage-2 jobs then one Stage-2B job; robust fair 3:1 interleave",
        "stage2_jobs": 144,
        "stage2b_jobs": 48,
        "target_adaptation_fits": 0,
        "strict_preflight": preflight["artifact"],
        "resource_check": preflight["record"]["resource_check"],
        "scientific_parameters_changed_by_scheduler": False,
        "final_target_labels_accessed": False,
        "stage3_started": False,
        "stage4_started": False,
        "created_utc": a40.utc(),
    })
    pending = list(queue)
    active: dict[tuple[str, str], tuple[subprocess.Popen, object, object]] = {}
    failures: list[dict[str, Any]] = []
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    try:
        while pending or active:
            while pending and len(active) < workers and not failures:
                stage, job = pending.pop(0)
                job_id = job["job_id"]
                worker_root = run_root + "workers/" + stage + "/"
                stdout_relative = worker_root + job_id + ".stdout.log"
                stderr_relative = worker_root + job_id + ".stderr.log"
                stdout_path = a40.output_path(stdout_relative)
                stderr_path = a40.output_path(stderr_relative)
                stdout_path.parent.mkdir(parents=True, exist_ok=True)
                stdout_handle = stdout_path.open("xb")
                stderr_handle = stderr_path.open("xb")
                command = _worker_command(stage, job_id, manifest_sha256, preflight_path, workers)
                try:
                    process = subprocess.Popen(
                        command, cwd=ROOT, stdout=stdout_handle, stderr=stderr_handle,
                        creationflags=flags,
                    )
                except BaseException:
                    stdout_handle.close()
                    stderr_handle.close()
                    raise
                a40.write_json_atomic(worker_root + job_id + ".launch.json", {
                    "status": "LAUNCHED",
                    "stage": stage,
                    "job_id": job_id,
                    "worker_pid": process.pid,
                    "coordinator_pid": os.getpid(),
                    "worker_limit": workers,
                    "worker_count_is_scientific_hyperparameter": False,
                    "fresh_python_process": True,
                    "stdout": stdout_relative,
                    "stderr": stderr_relative,
                    "created_utc": a40.utc(),
                })
                active[(stage, job_id)] = (process, stdout_handle, stderr_handle)
            if not active:
                break
            finished: list[tuple[str, str]] = []
            while not finished:
                finished = [key for key, (process, _, _) in active.items() if process.poll() is not None]
                if not finished:
                    time.sleep(1.0)
            for stage, job_id in finished:
                process, stdout_handle, stderr_handle = active.pop((stage, job_id))
                stdout_handle.close()
                stderr_handle.close()
                exit_code = int(process.returncode)
                completed_relative = (
                    (c.STAGE2_OUT if stage == "stage2" else c.STAGE2B_OUT)
                    + "completed/" + job_id + ".json"
                )
                completed = c.repository_path(completed_relative).is_file()
                worker_root = run_root + "workers/" + stage + "/"
                status = {
                    "status": "EXITED_ZERO_WITH_COMPLETION" if exit_code == 0 and completed else "FAILED",
                    "stage": stage,
                    "job_id": job_id,
                    "worker_pid": process.pid,
                    "exit_code": exit_code,
                    "completed_record_present": completed,
                    "completed_record": a40.entry(completed_relative) if completed else None,
                    "stdout": a40.entry(worker_root + job_id + ".stdout.log"),
                    "stderr": a40.entry(worker_root + job_id + ".stderr.log"),
                    "finished_utc": a40.utc(),
                }
                a40.write_json_atomic(worker_root + job_id + ".exit.json", status)
                if exit_code != 0 or not completed:
                    failures.append({"stage": stage, "job_id": job_id, "exit_code": exit_code,
                                     "completed_record_present": completed})
        if failures:
            raise RuntimeError("Worker failure; no new jobs dispatched after first observed failure: " + json.dumps(failures))
        if pending:
            raise RuntimeError("Controller ended with undispatched jobs")
        for stage in ("stage2", "stage2b"):
            subprocess.run(_postrun_command("validate", stage, manifest_sha256, workers, record=True), cwd=ROOT, check=True)
        for stage in ("stage2", "stage2b"):
            subprocess.run(_postrun_command("freeze", stage, manifest_sha256, workers), cwd=ROOT, check=True)
        stage2_decision = c.STAGE2_OUT + "stage2_decision_manifest.json"
        stage2b_decision = c.STAGE2B_OUT + "stage2b_decision_manifest.json"
        a40.write_json_atomic(run_root + "controller_complete.json", {
            "status": "STAGE2_AND_STAGE2B_V2_2_FROZEN_STOP",
            "run_id": run_id,
            "operational_worker_count": workers,
            "completed_jobs": 192,
            "stage2_decision": a40.entry(stage2_decision),
            "stage2b_decision": a40.entry(stage2b_decision),
            "stage3_started": False,
            "stage4_started": False,
            "final_target_labels_accessed": False,
            "completed_utc": a40.utc(),
        })
    except BaseException as error:
        a40.write_json_atomic(run_root + "controller_failed.json", {
            "status": "FAILED_OR_INTERRUPTED_BEFORE_JOINT_FREEZE",
            "run_id": run_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "new_jobs_not_dispatched_after_observed_failure": True,
            "scientific_decision_produced_by_failure_record": False,
            "final_target_labels_accessed": False,
            "created_utc": a40.utc(),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared Stage-2/Stage-2B V2.2 scheduler")
    parser.add_argument("mode", choices=("launch",))
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--workers", type=int, default=a40.DEFAULT_WORKERS)
    args = parser.parse_args()
    with a40.execution_lock("stage2-stage2b-controller"):
        launch(args.manifest_sha256, args.workers)


if __name__ == "__main__":
    main()

