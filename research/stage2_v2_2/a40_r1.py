"""Operational R1 validation overlay for the frozen V2.2 A40 workload.

The original runner remains byte-identical because its hash is bound into both
scientific contracts.  This module delegates all scientific work to it and
changes only the representation-level checkpoint metadata comparison.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import uuid
from typing import Any

import numpy as np

from research.stage2_v2_2 import a40 as base
from research.stage2_v2_2 import core as c


BASE_BUNDLE_MANIFEST_SHA256 = "ce9f6c32ce0bcf0d07db3cf2e890df66fa96f6595f4220a7bf2819d64d936342"
OVERLAY_MANIFEST = "deployment/stage2_v2_2_a40/BUNDLE_MANIFEST_OPERATIONAL_R1.json"
OVERLAY_AUTHORIZATION = "deployment/stage2_v2_2_a40/EXECUTION_AUTHORIZATION_OPERATIONAL_R1.json"
ORIGINAL_A40_SHA256 = "847cdeb2524d88ddd2b8cd52f080d04b27782247ae15a5e55e164c2e70906559"
ORIGINAL_CONTROLLER_SHA256 = "d7f88397e8cc2b046de950c2cde9a9aaa4d7f8d68494c26e0d8745af544c0064"
STAGE2_CONTRACT_SHA256 = "55ebfa06ca49d77e5ae0b7b2e531d6bb14c26a2b93f478da7754c75fb1aae52f"
STAGE2B_CONTRACT_SHA256 = "2eb1a2a14dd1412f1286248c547814a2c7e04fad94edaace2084c7d78be0cad3"
STAGE2_JOB_MAP_SHA256 = "fd34f71b46eb6f9cfc9adb5ebfd44001a2e26df049b9bb197aff1d272e7a074a"
STAGE2B_JOB_MAP_SHA256 = "0cf72a4f66797bcc5625bd438046562c5fd444677676971e2697e1f9a3d0f23d"
EXPECTED_EXISTING = (
    ("stage2", "ggru_k04_h032_d00_target-532_seed-17"),
    ("stage2", "ggru_k04_h032_d00_target-532_seed-29"),
    ("stage2", "ggru_k04_h032_d00_target-532_seed-43"),
    ("stage2b", "vgru_h032_d00_target-532_seed-17"),
)

# Operational interface consumed by the unchanged shared controller.
DEFAULT_WORKERS = base.DEFAULT_WORKERS
MAX_WORKERS = base.MAX_WORKERS
THREADS_PER_WORKER = base.THREADS_PER_WORKER
utc = base.utc
require = base.require
entry = base.entry
output_path = base.output_path
write_json_atomic = base.write_json_atomic
execution_lock = base.execution_lock

_ORIGINAL_VALIDATE_COMPLETED = base.validate_completed
_ORIGINAL_LOAD_CHECKPOINT = base._load_checkpoint


def canonicalize_metadata(value: Any) -> Any:
    """Return an exact deterministic JSON-compatible representation.

    Lists and tuples retain element order but share the JSON array
    representation. NumPy scalar values become their exact native scalar and
    Path-like values become stable filesystem strings. No float rounding or
    field filtering is performed.
    """
    if isinstance(value, np.generic):
        return canonicalize_metadata(value.item())
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Metadata must be finite JSON data")
        return value
    if isinstance(value, (list, tuple)):
        return [canonicalize_metadata(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Metadata dictionary keys must be strings")
        return {key: canonicalize_metadata(item) for key, item in value.items()}
    raise TypeError("Unsupported metadata value: " + type(value).__name__)


def metadata_equal(left: Any, right: Any) -> bool:
    return canonicalize_metadata(left) == canonicalize_metadata(right)


def verify_overlay(expected_manifest_sha256: str) -> dict[str, Any]:
    require(c.sha256_file(OVERLAY_MANIFEST) == expected_manifest_sha256,
            "External R1 overlay-manifest binding failed")
    manifest = c.read_json(OVERLAY_MANIFEST)
    require(manifest["status"] == "OPERATIONAL_R1_PATCH" and manifest["protocol_version"] == "2.2",
            "Wrong R1 overlay identity")
    require(manifest["base_bundle_manifest_sha256"] == BASE_BUNDLE_MANIFEST_SHA256,
            "R1 base-package binding changed")
    for relative, digest in manifest["files"].items():
        require(c.sha256_file(relative) == digest, "R1 overlay member changed: " + relative)
    required = {
        "deployment/stage2_v2_2_a40/BUNDLE_MANIFEST.json": BASE_BUNDLE_MANIFEST_SHA256,
        "research/stage2_v2_2/a40.py": ORIGINAL_A40_SHA256,
        "research/stage2_v2_2/controller.py": ORIGINAL_CONTROLLER_SHA256,
        c.STAGE2_CONTRACT: STAGE2_CONTRACT_SHA256,
        c.STAGE2B_CONTRACT: STAGE2B_CONTRACT_SHA256,
        c.STAGE2_JOB_MAP: STAGE2_JOB_MAP_SHA256,
        c.STAGE2B_JOB_MAP: STAGE2B_JOB_MAP_SHA256,
        c.STAGE1_DECISION: c.STAGE1_DECISION_SHA256,
    }
    require(manifest["required_existing_files"] == required, "R1 required-file set changed")
    for relative, digest in required.items():
        require(c.sha256_file(relative) == digest, "R1 immutable dependency changed: " + relative)
    base.verify_package(BASE_BUNDLE_MANIFEST_SHA256)
    authorization = c.read_json(OVERLAY_AUTHORIZATION)
    require(authorization["scientific_changes"] == [] and authorization["workers"] == 4,
            "R1 authorization drift")
    require(authorization["resume_requires_existing_completion_validation_pass"] is True,
            "R1 retention gate disabled")
    require(authorization["final_target_label_access_authorized"] is False,
            "R1 final-target firewall disabled")
    return manifest


@contextmanager
def _canonical_validator_patch():
    previous = base.validate_completed
    base.validate_completed = validate_completed
    try:
        yield
    finally:
        base.validate_completed = previous


def validate_completed(job: dict[str, Any], manifest_sha256: str,
                       *, recompute_metrics: bool = True) -> dict[str, Any]:
    """Run the original validator after exact metadata canonical comparison."""
    stage = "stage2" if job["stage"] == 2 else "stage2b"
    completed_relative = base._output_root(stage) + "completed/" + job["job_id"] + ".json"
    record = c.read_json(completed_relative)
    checkpoint_relative = record["checkpoint"]["path"]
    payload = _ORIGINAL_LOAD_CHECKPOINT(checkpoint_relative)
    require(metadata_equal(payload["metadata"], record["checkpoint_metadata"]),
            "Checkpoint metadata canonical comparison")

    normalized_payload = dict(payload)
    normalized_payload["metadata"] = canonicalize_metadata(payload["metadata"])
    previous_loader = base._load_checkpoint

    def normalized_loader(relative: str):
        if relative == checkpoint_relative:
            return normalized_payload
        return previous_loader(relative)

    base._load_checkpoint = normalized_loader
    try:
        return _ORIGINAL_VALIDATE_COMPLETED(
            job, manifest_sha256, recompute_metrics=recompute_metrics
        )
    finally:
        base._load_checkpoint = previous_loader


def preflight(overlay_manifest_sha256: str, workers: int) -> dict[str, Any]:
    verify_overlay(overlay_manifest_sha256)
    result = base.preflight(BASE_BUNDLE_MANIFEST_SHA256, workers)
    overlay_record = {
        "status": "PASS",
        "scope": "STRICT_ZERO_TRAINING_OPERATIONAL_R1_A40_PREFLIGHT",
        "overlay_manifest_sha256": overlay_manifest_sha256,
        "base_bundle_manifest_sha256": BASE_BUNDLE_MANIFEST_SHA256,
        "base_preflight": result["artifact"],
        "operational_workers": workers,
        "canonical_metadata_validator": True,
        "scientific_parameter_changes": [],
        "scientific_training_steps": 0,
        "scientific_evaluations": 0,
        "final_target_labels_accessed": False,
        "stage3_started": False,
        "stage4_started": False,
        "created_utc": utc(),
    }
    saved = write_json_atomic(
        c.JOINT_OUT + "operational_r1/preflight/" + uuid.uuid4().hex + ".json",
        overlay_record,
    )
    print(json.dumps({"status": "PASS", "r1_preflight": saved,
                      "base_preflight": result["artifact"]}, indent=2), flush=True)
    return {"record": result["record"], "artifact": result["artifact"], "r1_artifact": saved}


def _job(stage: str, job_id: str) -> dict[str, Any]:
    job_map = c.read_json(c.STAGE2_JOB_MAP if stage == "stage2" else c.STAGE2B_JOB_MAP)
    job = next((candidate for candidate in job_map["jobs"] if candidate["job_id"] == job_id), None)
    require(job is not None, "Unregistered immutable R1 job")
    return job


def _original_failure_records(job_id: str) -> list[dict[str, Any]]:
    root = c.repository_path(c.JOINT_OUT + "controller_runs")
    matches: list[dict[str, Any]] = []
    if not root.exists():
        return matches
    for path in root.rglob(job_id + ".exit.json"):
        relative = path.relative_to(c.ROOT).as_posix()
        value = c.read_json(relative)
        if value.get("job_id") == job_id and value.get("status") == "FAILED" and value.get("exit_code") == 1:
            matches.append(entry(relative))
    return matches


def validate_existing(overlay_manifest_sha256: str) -> dict[str, Any]:
    """Validate and retain exactly the four pre-R1 completed jobs, without labels."""
    verify_overlay(overlay_manifest_sha256)
    rows = []
    for stage, job_id in EXPECTED_EXISTING:
        job = _job(stage, job_id)
        record = validate_completed(job, BASE_BUNDLE_MANIFEST_SHA256, recompute_metrics=False)
        require(record["source_updates_completed"] == 12_000, "R1 source-update check")
        require(record["job"] == job and record["stage"] == job["stage"], "R1 scientific binding check")
        require(record["final_target_labels_accessed"] is False, "R1 final-label firewall check")
        failures = _original_failure_records(job_id)
        require(failures, "Original post-completion worker failure record is missing: " + job_id)
        stage_name = "stage2" if job["stage"] == 2 else "stage2b"
        completion_relative = base._output_root(stage_name) + "completed/" + job_id + ".json"
        rows.append({
            "job_id": job_id,
            "stage": job["stage"],
            "config_id": job["config_id"],
            "pseudo_target_city_id": job["pseudo_target_city_id"],
            "seed": job["seed"],
            "completion_record": entry(completion_relative),
            "checkpoint": record["checkpoint"],
            "prediction": record["prediction"],
            "metadata_canonical_comparison": "PASS",
            "source_updates_12000": "PASS",
            "scientific_binding": "PASS",
            "checkpoint_hash": "PASS",
            "prediction_hash": "PASS",
            "final_label_firewall": "PASS",
            "original_worker_failure_records": failures,
            "classification": "AUTHORITATIVE_A40_COMPLETION_RETAINED_AFTER_OPERATIONAL_VALIDATOR_FIX",
        })
    report = {
        "status": "PASS",
        "classification": "AUTHORITATIVE_A40_COMPLETION_RETAINED_AFTER_OPERATIONAL_VALIDATOR_FIX",
        "incident": "Workers exited 1 only after training, evaluation, checkpoint, prediction, and completion-record persistence; raw tuple/list metadata equality then failed.",
        "root_cause": "optimizer.betas tuple in torch checkpoint versus JSON-normalized list in completion metadata",
        "scientific_parameters_or_results_changed": False,
        "original_exit_records_preserved": True,
        "validated_existing_completions": rows,
        "stage2_retained": 3,
        "stage2b_retained": 1,
        "remaining_stage2": 141,
        "remaining_stage2b": 47,
        "remaining_total": 188,
        "scientific_training_steps_performed": 0,
        "scientific_evaluations_performed": 0,
        "retrospective_development_labels_accessed": False,
        "final_target_labels_accessed": False,
        "created_utc": utc(),
    }
    relative = c.JOINT_OUT + "operational_r1/existing_completion_revalidation.json"
    path = c.repository_path(relative)
    if path.exists():
        existing = c.read_json(relative)
        require(existing["status"] == "PASS" and
                existing["classification"] == report["classification"],
                "Existing R1 retention report is invalid")
        return existing
    write_json_atomic(relative, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def run_job(stage: str, job_id: str, overlay_manifest_sha256: str,
            preflight_path: str, workers: int) -> None:
    verify_overlay(overlay_manifest_sha256)
    job = _job(stage, job_id)
    completed_relative = base._output_root(stage) + "completed/" + job_id + ".json"
    if c.repository_path(completed_relative).exists():
        validate_completed(job, BASE_BUNDLE_MANIFEST_SHA256, recompute_metrics=False)
        print("SKIP corrected-R1-validated completed immutable job " + job_id, flush=True)
        return
    with _canonical_validator_patch():
        base.run_job(stage, job_id, BASE_BUNDLE_MANIFEST_SHA256, preflight_path, workers)


def validate_stage(stage: str, overlay_manifest_sha256: str) -> dict[str, Any]:
    verify_overlay(overlay_manifest_sha256)
    with _canonical_validator_patch():
        return base.validate_stage(stage, BASE_BUNDLE_MANIFEST_SHA256)


def record_validation(stage: str, overlay_manifest_sha256: str) -> dict[str, Any]:
    verify_overlay(overlay_manifest_sha256)
    with _canonical_validator_patch():
        return base.record_validation(stage, BASE_BUNDLE_MANIFEST_SHA256)


def freeze(stage: str, overlay_manifest_sha256: str) -> dict[str, Any]:
    verify_overlay(overlay_manifest_sha256)
    with _canonical_validator_patch():
        return base.freeze(stage, BASE_BUNDLE_MANIFEST_SHA256)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2/Stage-2B operational R1 overlay")
    parser.add_argument("mode", choices=("package-check", "preflight", "validate-existing",
                                         "job", "validate", "freeze"))
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--stage", choices=("stage2", "stage2b"))
    parser.add_argument("--job-id")
    parser.add_argument("--preflight")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    if args.mode == "package-check":
        verify_overlay(args.manifest_sha256)
        print(json.dumps({"status": "PASS", "scope": "OPERATIONAL_R1_PACKAGE_ONLY",
                          "scientific_training_steps": 0, "scientific_evaluations": 0}, indent=2))
    elif args.mode == "preflight":
        preflight(args.manifest_sha256, args.workers)
    elif args.mode == "validate-existing":
        validate_existing(args.manifest_sha256)
    elif args.mode == "job":
        require(args.stage and args.job_id and args.preflight, "R1 job arguments required")
        with execution_lock(args.stage + "-" + args.job_id):
            run_job(args.stage, args.job_id, args.manifest_sha256, args.preflight, args.workers)
    elif args.mode == "validate":
        require(args.stage is not None, "R1 validate stage required")
        if args.record:
            saved = record_validation(args.stage, args.manifest_sha256)
            print(json.dumps({"status": "PASS", "validator_report": saved}, indent=2))
        else:
            validate_stage(args.stage, args.manifest_sha256)
    else:
        require(args.stage is not None, "R1 freeze stage required")
        freeze(args.stage, args.manifest_sha256)


if __name__ == "__main__":
    main()
