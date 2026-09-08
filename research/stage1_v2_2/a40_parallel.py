"""Operational four-worker A40 controller for the frozen Stage-1 V2.2 jobs.

Scientific workers remain the byte-identical ``research.stage1_v2_2.a40 job``
entry point.  This module changes only process scheduling, logging, and runtime
provenance.  It never executes a scientific fit inside the controller process.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

if os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8") != ":4096:8":
    raise RuntimeError("Conflicting CUBLAS_WORKSPACE_CONFIG")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

from research.stage1_v2_2 import a40 as base


ROOT = Path(__file__).resolve().parents[2]
THROUGHPUT_MANIFEST = "deployment/stage1_v2_2_a40/BUNDLE_MANIFEST_THROUGHPUT_R1.json"
THROUGHPUT_AUTHORIZATION = "deployment/stage1_v2_2_a40/EXECUTION_AUTHORIZATION_THROUGHPUT_R1.json"
ORIGINAL_MANIFEST_SHA256 = "357f93afb5279099af00334b339860b557531fd2284b5e05bbedef9aa380dbe9"
DEFAULT_WORKERS = 4
MAX_WORKERS = 4
THREADS_PER_WORKER = 2


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_throughput_package(expected_sha256: str) -> dict:
    require(base.c.sha(THROUGHPUT_MANIFEST) == expected_sha256, "External throughput-manifest binding")
    manifest = base.c.read(THROUGHPUT_MANIFEST)
    require(manifest["execution_environment"] == base.ENVIRONMENT, "Wrong execution environment")
    require(manifest["operational_worker_policy"] == {
        "default_workers": DEFAULT_WORKERS,
        "maximum_workers": MAX_WORKERS,
        "threads_per_worker": THREADS_PER_WORKER,
        "one_fresh_python_process_per_immutable_fit": True,
        "scientific_hyperparameter": False,
    }, "Throughput worker policy changed")
    for relative, digest in manifest["files"].items():
        require(base.c.sha(relative) == digest, "Throughput bundle member changed: " + relative)
    require(base.c.sha(base.MANIFEST) == ORIGINAL_MANIFEST_SHA256, "Original sealed A40 manifest changed")
    base.verify_package(ORIGINAL_MANIFEST_SHA256)
    authorization = base.c.read(THROUGHPUT_AUTHORIZATION)
    require(authorization["operational_worker_count_default"] == DEFAULT_WORKERS, "Default worker count changed")
    require(authorization["operational_worker_count_maximum"] == MAX_WORKERS, "Maximum worker count changed")
    require(authorization["worker_count_is_scientific_hyperparameter"] is False, "Worker count became scientific")
    require(authorization["job_map_sha256"] == base.JOBMAP_SHA, "Immutable job map changed")
    require(authorization["original_bundle_manifest_sha256"] == ORIGINAL_MANIFEST_SHA256, "Original execution binding changed")
    for key in (
        "batch_size_changed", "updates_changed", "optimizer_changed", "architecture_changed",
        "seeds_changed", "data_changed", "selection_rule_changed",
    ):
        require(authorization[key] is False, "Scientific field changed: " + key)
    return manifest


def throughput_preflight(manifest_sha256: str) -> dict:
    verify_throughput_package(manifest_sha256)
    scientific_preflight = base.preflight(ORIGINAL_MANIFEST_SHA256)
    record = {
        "status": "PASS",
        "scope": "OPERATIONAL_THROUGHPUT_PREFLIGHT_PLUS_ORIGINAL_STRICT_A40_PREFLIGHT",
        "throughput_manifest_sha256": manifest_sha256,
        "original_bundle_manifest_sha256": ORIGINAL_MANIFEST_SHA256,
        "scientific_preflight": scientific_preflight,
        "default_workers": DEFAULT_WORKERS,
        "maximum_workers": MAX_WORKERS,
        "threads_per_worker": THREADS_PER_WORKER,
        "one_fresh_python_process_per_immutable_fit": True,
        "scientific_hyperparameters_changed": False,
        "scientific_training_steps": 0,
        "scientific_evaluation_performed": False,
        "final_target_labels_accessed": False,
        "created_utc": base.c.utc(),
    }
    saved = base.write(base.OUT + "throughput_preflight/" + uuid.uuid4().hex + ".json", record)
    print(json.dumps({"status": "PASS", "throughput_preflight": saved}, indent=2), flush=True)
    return {"record": record, "artifact": saved}


def _cache_payload_bytes() -> int:
    cache = base.c.read(base.c.CACHE + "cache_manifest.json")
    return sum(
        base.path(array["path"]).stat().st_size
        for city in cache["cities"].values()
        for array in city["arrays"].values()
    )


def _worker_command(job_id: str, manifest_sha256: str, scientific_preflight_path: str) -> list[str]:
    return [
        sys.executable, "-X", "utf8", "-B", "-m", "research.stage1_v2_2.a40", "job",
        "--manifest-sha256", ORIGINAL_MANIFEST_SHA256,
        "--preflight", scientific_preflight_path,
        "--job-id", job_id,
    ]


def _run_phase(
    jobs: list[dict],
    *,
    phase: str,
    workers: int,
    manifest_sha256: str,
    scientific_preflight_path: str,
    run_root: str,
) -> None:
    pending = list(jobs)
    active: dict[str, tuple[subprocess.Popen, object, object]] = {}
    failures: list[dict] = []
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    while pending or active:
        while pending and len(active) < workers and not failures:
            job = pending.pop(0)
            job_id = job["id"]
            worker_root = run_root + "workers/"
            stdout_rel = worker_root + job_id + ".stdout.log"
            stderr_rel = worker_root + job_id + ".stderr.log"
            stdout_path = base.path(stdout_rel, output=True)
            stderr_path = base.path(stderr_rel, output=True)
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_handle = stdout_path.open("xb")
            stderr_handle = stderr_path.open("xb")
            command = _worker_command(job_id, manifest_sha256, scientific_preflight_path)
            try:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    creationflags=creation_flags,
                )
            except BaseException:
                stdout_handle.close()
                stderr_handle.close()
                raise
            base.write(worker_root + job_id + ".launch.json", {
                "status": "LAUNCHED",
                "job_id": job_id,
                "phase": phase,
                "worker_pid": process.pid,
                "coordinator_pid": os.getpid(),
                "worker_limit": workers,
                "threads_per_worker": THREADS_PER_WORKER,
                "scientific_hyperparameter": False,
                "fresh_python_process": True,
                "stdout": stdout_rel,
                "stderr": stderr_rel,
                "created_utc": base.c.utc(),
            })
            active[job_id] = (process, stdout_handle, stderr_handle)
        if not active:
            break
        finished: list[str] = []
        while not finished:
            finished = [job_id for job_id, (process, _, _) in active.items() if process.poll() is not None]
            if not finished:
                time.sleep(1.0)
        for job_id in finished:
            process, stdout_handle, stderr_handle = active.pop(job_id)
            exit_code = int(process.returncode)
            stdout_handle.close()
            stderr_handle.close()
            completed_rel = base.OUT + "completed/" + job_id + ".json"
            completed = base.path(completed_rel).is_file()
            status = {
                "status": "EXITED_ZERO_VALIDATED_COMPLETION" if exit_code == 0 and completed else "FAILED",
                "job_id": job_id,
                "phase": phase,
                "worker_pid": process.pid,
                "exit_code": exit_code,
                "completed_record_present": completed,
                "completed_record": base.entry(completed_rel) if completed else None,
                "stdout": base.entry(run_root + "workers/" + job_id + ".stdout.log"),
                "stderr": base.entry(run_root + "workers/" + job_id + ".stderr.log"),
                "finished_utc": base.c.utc(),
            }
            base.write(run_root + "workers/" + job_id + ".exit.json", status)
            if exit_code != 0 or not completed:
                failures.append({"job_id": job_id, "exit_code": exit_code, "completed_record_present": completed})
    if failures:
        raise RuntimeError("One or more isolated workers failed; no new jobs were dispatched after failure: " + json.dumps(failures))
    require(not pending, "Phase ended with undispatched immutable jobs")


def launch(manifest_sha256: str, workers: int) -> None:
    require(1 <= workers <= MAX_WORKERS, f"Operational workers must be between 1 and {MAX_WORKERS}")
    verify_throughput_package(manifest_sha256)
    preflight = throughput_preflight(manifest_sha256)
    scientific_preflight_path = preflight["record"]["scientific_preflight"]["path"]
    job_map = base.c.read(base.JOBMAP)
    source_jobs = [job for job in job_map["jobs"] if job["phase"] == "source"]
    adaptation_jobs = [job for job in job_map["jobs"] if job["phase"] == "adaptation"]
    require(len(source_jobs) == 24 and len(adaptation_jobs) == 24, "Frozen workload changed")
    run_id = uuid.uuid4().hex
    run_root = base.OUT + "controller_runs/" + run_id + "/"
    payload_bytes = _cache_payload_bytes()
    base.write(run_root + "runtime_provenance.json", {
        "status": "RUNNING",
        "run_id": run_id,
        "coordinator_pid": os.getpid(),
        "throughput_manifest_sha256": manifest_sha256,
        "original_bundle_manifest_sha256": ORIGINAL_MANIFEST_SHA256,
        "job_map_sha256": base.JOBMAP_SHA,
        "operational_worker_count": workers,
        "operational_worker_count_default": DEFAULT_WORKERS,
        "operational_worker_count_maximum": MAX_WORKERS,
        "threads_per_worker": THREADS_PER_WORKER,
        "worker_count_is_scientific_hyperparameter": False,
        "one_fresh_python_process_per_immutable_fit": True,
        "phase_order": ["all_source_fits", "all_adaptation_fits", "validate", "freeze", "stop"],
        "source_worker_read_only_cache_payload_bytes_upper_bound": payload_bytes,
        "concurrent_source_worker_cache_payload_bytes_upper_bound": payload_bytes * workers,
        "scientific_hyperparameters_changed": False,
        "scientific_preflight": preflight["record"]["scientific_preflight"],
        "created_utc": base.c.utc(),
    })
    try:
        _run_phase(
            source_jobs, phase="source", workers=workers, manifest_sha256=manifest_sha256,
            scientific_preflight_path=scientific_preflight_path, run_root=run_root,
        )
        _run_phase(
            adaptation_jobs, phase="adaptation", workers=workers, manifest_sha256=manifest_sha256,
            scientific_preflight_path=scientific_preflight_path, run_root=run_root,
        )
        subprocess.run(
            [sys.executable, "-X", "utf8", "-B", "-m", "research.stage1_v2_2.a40", "validate",
             "--manifest-sha256", ORIGINAL_MANIFEST_SHA256],
            cwd=ROOT, check=True,
        )
        subprocess.run(
            [sys.executable, "-X", "utf8", "-B", "-m", "research.stage1_v2_2.a40", "freeze",
             "--manifest-sha256", ORIGINAL_MANIFEST_SHA256],
            cwd=ROOT, check=True,
        )
        base.write(run_root + "controller_complete.json", {
            "status": "STAGE1_FROZEN_STOP",
            "run_id": run_id,
            "operational_worker_count": workers,
            "completed_jobs": 48,
            "stage2_started": False,
            "stage2b_started": False,
            "final_target_labels_accessed": False,
            "completed_utc": base.c.utc(),
        })
    except BaseException as error:
        base.write(run_root + "controller_failed.json", {
            "status": "FAILED_OR_INTERRUPTED_BEFORE_STAGE1_FREEZE",
            "run_id": run_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "scientific_decision_produced_by_controller": False,
            "created_utc": base.c.utc(),
        })
        raise


def validate(manifest_sha256: str) -> None:
    verify_throughput_package(manifest_sha256)
    subprocess.run(
        [sys.executable, "-X", "utf8", "-B", "-m", "research.stage1_v2_2.a40", "validate",
         "--manifest-sha256", ORIGINAL_MANIFEST_SHA256],
        cwd=ROOT, check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Operational parallel controller for frozen Stage-1 V2.2 A40 jobs")
    parser.add_argument("mode", choices=("package-check", "preflight", "launch", "validate"))
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    require(1 <= args.workers <= MAX_WORKERS, f"Operational workers must be between 1 and {MAX_WORKERS}")
    if args.mode == "package-check":
        verify_throughput_package(args.manifest_sha256)
        print(json.dumps({
            "status": "PASS", "scope": "PACKAGE_ONLY_NO_CUDA_CLAIM",
            "default_workers": DEFAULT_WORKERS, "maximum_workers": MAX_WORKERS,
            "scientific_training_steps": 0,
        }, indent=2))
    elif args.mode == "preflight":
        throughput_preflight(args.manifest_sha256)
    elif args.mode == "launch":
        with base.execution_lock("coordinator"):
            launch(args.manifest_sha256, args.workers)
    elif args.mode == "validate":
        validate(args.manifest_sha256)


if __name__ == "__main__":
    main()
