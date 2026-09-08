"""Operational R2 interface wrapper over unchanged base and R1 validation."""
from __future__ import annotations

import argparse
import json
import uuid
from typing import Any

from research.stage2_v2_2 import a40 as base
from research.stage2_v2_2 import a40_r1 as r1
from research.stage2_v2_2 import core as c


R1_OVERLAY_MANIFEST_SHA256 = "2b4114de5c6d85a3217f0c4c38dbf227790f4dbd3866667e02e5b6328cf41d75"
R1_A40_SHA256 = "3527fc15d69d0c7e6210822d4026b13ca4cd47883c2e88819fd7e5e99d6241d3"
R1_CONTROLLER_SHA256 = "eea360276f1d0faec0eac85389219543e787d0e6d4977406bf7040120ed6f2fe"
R2_MANIFEST = "deployment/stage2_v2_2_a40/BUNDLE_MANIFEST_OPERATIONAL_R2.json"
R2_AUTHORIZATION = "deployment/stage2_v2_2_a40/EXECUTION_AUTHORIZATION_OPERATIONAL_R2.json"

# Explicitly export the complete interface used by controller.py. Unknown names
# delegate to R1 first, then the byte-identical base module through __getattr__.
DEFAULT_WORKERS = base.DEFAULT_WORKERS
MAX_WORKERS = base.MAX_WORKERS
THREADS_PER_WORKER = base.THREADS_PER_WORKER
utc = base.utc
require = base.require
entry = base.entry
output_path = base.output_path
write_json_atomic = base.write_json_atomic
write_bytes_atomic = base.write_bytes_atomic
execution_lock = base.execution_lock

canonicalize_metadata = r1.canonicalize_metadata
metadata_equal = r1.metadata_equal
validate_completed = r1.validate_completed


def __getattr__(name: str):
    """Delegate every unchanged module attribute without copying its logic."""
    try:
        return getattr(r1, name)
    except AttributeError:
        return getattr(base, name)


def verify_overlay(expected_manifest_sha256: str) -> dict[str, Any]:
    require(c.sha256_file(R2_MANIFEST) == expected_manifest_sha256,
            "External R2 overlay-manifest binding failed")
    manifest = c.read_json(R2_MANIFEST)
    require(manifest["status"] == "OPERATIONAL_R2_CONTROLLER_INTERFACE_PATCH",
            "Wrong R2 overlay identity")
    require(manifest["r1_overlay_manifest_sha256"] == R1_OVERLAY_MANIFEST_SHA256,
            "R2-to-R1 binding changed")
    for relative, digest in manifest["files"].items():
        require(c.sha256_file(relative) == digest, "R2 overlay member changed: " + relative)
    required = {
        "deployment/stage2_v2_2_a40/BUNDLE_MANIFEST_OPERATIONAL_R1.json": R1_OVERLAY_MANIFEST_SHA256,
        "research/stage2_v2_2/a40_r1.py": R1_A40_SHA256,
        "research/stage2_v2_2/controller_r1.py": R1_CONTROLLER_SHA256,
        "research/stage2_v2_2/a40.py": r1.ORIGINAL_A40_SHA256,
        "research/stage2_v2_2/controller.py": r1.ORIGINAL_CONTROLLER_SHA256,
        c.STAGE2_CONTRACT: r1.STAGE2_CONTRACT_SHA256,
        c.STAGE2B_CONTRACT: r1.STAGE2B_CONTRACT_SHA256,
        c.STAGE2_JOB_MAP: r1.STAGE2_JOB_MAP_SHA256,
        c.STAGE2B_JOB_MAP: r1.STAGE2B_JOB_MAP_SHA256,
    }
    require(manifest["required_existing_files"] == required,
            "R2 required existing-file set changed")
    for relative, digest in required.items():
        require(c.sha256_file(relative) == digest, "R2 immutable dependency changed: " + relative)
    r1.verify_overlay(R1_OVERLAY_MANIFEST_SHA256)
    authorization = c.read_json(R2_AUTHORIZATION)
    require(authorization["scientific_changes"] == [] and authorization["workers"] == 4,
            "R2 authorization drift")
    require(authorization["resume_requires_retained_four_validation_pass"] is True,
            "R2 retention gate disabled")
    return manifest


# This exact name is required by the unchanged original controller.
def verify_package(expected_manifest_sha256: str) -> dict[str, Any]:
    return verify_overlay(expected_manifest_sha256)


def preflight(r2_manifest_sha256: str, workers: int) -> dict[str, Any]:
    verify_package(r2_manifest_sha256)
    result = r1.preflight(R1_OVERLAY_MANIFEST_SHA256, workers)
    record = {
        "status": "PASS",
        "scope": "STRICT_ZERO_TRAINING_OPERATIONAL_R2_A40_PREFLIGHT",
        "r2_manifest_sha256": r2_manifest_sha256,
        "r1_overlay_manifest_sha256": R1_OVERLAY_MANIFEST_SHA256,
        "base_preflight": result["artifact"],
        "r1_preflight": result["r1_artifact"],
        "complete_original_controller_interface_available": True,
        "corrected_worker_module": "research.stage2_v2_2.a40_r2",
        "operational_workers": workers,
        "scientific_changes": [],
        "scientific_training_steps": 0,
        "scientific_evaluations": 0,
        "final_target_labels_accessed": False,
        "stage3_started": False,
        "stage4_started": False,
        "created_utc": utc(),
    }
    saved = write_json_atomic(
        c.JOINT_OUT + "operational_r2/preflight/" + uuid.uuid4().hex + ".json", record
    )
    print(json.dumps({"status": "PASS", "r2_preflight": saved,
                      "base_preflight": result["artifact"]}, indent=2), flush=True)
    return {"record": result["record"], "artifact": result["artifact"],
            "r1_artifact": result["r1_artifact"], "r2_artifact": saved}


def validate_existing(r2_manifest_sha256: str) -> dict[str, Any]:
    verify_package(r2_manifest_sha256)
    result = r1.validate_existing(R1_OVERLAY_MANIFEST_SHA256)
    require(result["status"] == "PASS" and result["stage2_retained"] == 3
            and result["stage2b_retained"] == 1
            and result["classification"] ==
            "AUTHORITATIVE_A40_COMPLETION_RETAINED_AFTER_OPERATIONAL_VALIDATOR_FIX"
            and len(result["validated_existing_completions"]) == 4
            and result["remaining_total"] == 188
            and result["final_target_labels_accessed"] is False,
            "R2 requires the authoritative R1 four-completion retention PASS")
    return result


def run_job(stage: str, job_id: str, r2_manifest_sha256: str,
            preflight_path: str, workers: int) -> None:
    verify_package(r2_manifest_sha256)
    r1.run_job(stage, job_id, R1_OVERLAY_MANIFEST_SHA256, preflight_path, workers)


def validate_stage(stage: str, r2_manifest_sha256: str) -> dict[str, Any]:
    verify_package(r2_manifest_sha256)
    return r1.validate_stage(stage, R1_OVERLAY_MANIFEST_SHA256)


def record_validation(stage: str, r2_manifest_sha256: str) -> dict[str, Any]:
    verify_package(r2_manifest_sha256)
    return r1.record_validation(stage, R1_OVERLAY_MANIFEST_SHA256)


def freeze(stage: str, r2_manifest_sha256: str) -> dict[str, Any]:
    verify_package(r2_manifest_sha256)
    return r1.freeze(stage, R1_OVERLAY_MANIFEST_SHA256)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2/Stage-2B operational R2 overlay")
    parser.add_argument("mode", choices=("package-check", "preflight", "validate-existing",
                                         "job", "validate", "freeze"))
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stage", choices=("stage2", "stage2b"))
    parser.add_argument("--job-id")
    parser.add_argument("--preflight")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    require(args.workers == 4, "Operational R2 repair is frozen to four workers")
    if args.mode == "package-check":
        verify_package(args.manifest_sha256)
        print(json.dumps({"status": "PASS", "scope": "OPERATIONAL_R2_PACKAGE_ONLY",
                          "controller_interface_complete": True,
                          "scientific_training_steps": 0, "scientific_evaluations": 0}, indent=2))
    elif args.mode == "preflight":
        preflight(args.manifest_sha256, args.workers)
    elif args.mode == "validate-existing":
        validate_existing(args.manifest_sha256)
    elif args.mode == "job":
        require(args.stage and args.job_id and args.preflight, "R2 job arguments required")
        with execution_lock(args.stage + "-" + args.job_id):
            run_job(args.stage, args.job_id, args.manifest_sha256, args.preflight, args.workers)
    elif args.mode == "validate":
        require(args.stage is not None, "R2 validate stage required")
        if args.record:
            saved = record_validation(args.stage, args.manifest_sha256)
            print(json.dumps({"status": "PASS", "validator_report": saved}, indent=2))
        else:
            validate_stage(args.stage, args.manifest_sha256)
    else:
        require(args.stage is not None, "R2 freeze stage required")
        freeze(args.stage, args.manifest_sha256)


if __name__ == "__main__":
    main()
