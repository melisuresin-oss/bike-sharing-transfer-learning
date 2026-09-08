from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from research.remote.stage2b_a40 import (
    EXECUTION_ENVIRONMENT,
    FINAL_TARGETS,
    AUTHORIZATION,
    INCIDENT_RECORD,
    PACKAGE_MANIFEST,
    PACKAGE_PREFLIGHT,
    REMOTE_OUTPUT,
    REMOTE_PREFLIGHT,
    REQUIRED_A40_RUNTIME,
    ROOT,
    assert_output_isolation,
    build_job_rows,
    establish_cublas_workspace_config,
    load_and_verify_frozen_seal,
    sha256_file,
    split_roles,
    validate_job_rows,
    read_job_map,
    load_authorization,
    write_json_once_or_verify,
    write_or_verify_job_map,
)


def _gpu_name() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if names != ["NVIDIA A40"]:
        raise RuntimeError(f"Expected exactly one NVIDIA A40, observed {names}")
    return names[0]


def runtime_snapshot() -> dict[str, Any]:
    # Establish this before importing torch; the fixture below fails at the
    # same CuBLAS boundary that aborted the R1 attempt if it is missing.
    establish_cublas_workspace_config()
    import duckdb
    import numpy
    import pandas
    import psutil
    import torch

    runtime = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": torch.__version__, "numpy": numpy.__version__,
        "pandas": pandas.__version__, "duckdb": duckdb.__version__,
        "psutil": psutil.__version__, "gpu": _gpu_name(),
    }
    if runtime != REQUIRED_A40_RUNTIME:
        raise RuntimeError(f"Frozen A40 runtime mismatch: {runtime}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA device must be visible")
    if torch.cuda.get_device_name(0) != "NVIDIA A40":
        raise RuntimeError("CUDA device is not NVIDIA A40")
    provenance = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_full_version": platform.python_version(),
        "module_paths": {
            "torch": str(Path(torch.__file__).resolve()),
            "numpy": str(Path(numpy.__file__).resolve()),
            "pandas": str(Path(pandas.__file__).resolve()),
            "duckdb": str(Path(duckdb.__file__).resolve()),
            "psutil": str(Path(psutil.__file__).resolve()),
        },
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
    }
    forbidden = (ROOT / "tmp" / "neural_build" / "pydeps").resolve()
    for name, value in provenance["module_paths"].items():
        path = Path(value).resolve()
        if path == forbidden or forbidden in path.parents:
            raise RuntimeError(f"Active {name} was imported from forbidden vendored pydeps")
    if str(forbidden) in sys.path:
        raise RuntimeError("Vendored pydeps was prepended to sys.path")
    from research.development.stage2b_vanilla_execution import (
        configure_deterministic_device,
        cuda_determinism_fixture,
    )
    device, settings = configure_deterministic_device(17, "cuda")
    fixture = cuda_determinism_fixture(device)
    return {
        "runtime": runtime,
        "runtime_provenance": provenance,
        "deterministic_settings": settings,
        "cuda_determinism_fixture": fixture,
    }


def run_preflight(*, runtime: bool) -> dict[str, Any]:
    frozen = load_and_verify_frozen_seal()
    roles = split_roles()
    rows = read_job_map() if runtime else build_job_rows()
    counts = validate_job_rows(rows)
    job_map = write_or_verify_job_map()
    assert_output_isolation(require_empty=True)
    result: dict[str, Any] = {
        "status": "PASS",
        "gate": "STRICT_REMOTE_RUNTIME" if runtime else "PACKAGE_INTEGRITY_ONLY",
        "execution_environment": EXECUTION_ENVIRONMENT,
        "expected_source_fits": 48,
        "expected_evaluations": 48,
        "fine_tuning_fits": 0,
        "job_counts": counts,
        "job_map_sha256": sha256_file(job_map),
        "frozen_stage2b_manifest_sha256": sha256_file(ROOT / "research/results/stage2b_vanilla_fairness_v2_1/stage2b_fairness_manifest.json"),
        "final_target_city_ids_derived_from_split": list(roles["final"]),
        "final_target_labels_accessed": False,
        "final_evaluation_windows_accessed": False,
        "stage4_outputs_accessed": False,
        "final_label_firewall_passed": set(roles["final"]) == set(FINAL_TARGETS),
        "scientific_results_present": False,
        "training_started": False,
        "evaluation_started": False,
        "runtime_validation": "PASS" if runtime else "PENDING_UNIVERSITY_A40",
    }
    if runtime:
        load_authorization()
        result.update({
            "execution_package_manifest_sha256": sha256_file(PACKAGE_MANIFEST),
            "execution_authorization_sha256": sha256_file(AUTHORIZATION),
            "incident_record_sha256": sha256_file(INCIDENT_RECORD),
            "stage2b_implementation_sha256": sha256_file(
                ROOT / "research/development/stage2b_vanilla_execution.py"
            ),
            "remote_stage2b_prior_attempt": "ABORTED_FIRST_FORWARD_MISSING_CUBLAS_WORKSPACE_CONFIG",
        })
        result.update(runtime_snapshot())
        target = REMOTE_PREFLIGHT
    else:
        result["runtime"] = None
        result["runtime_provenance"] = None
        result["deterministic_settings"] = {
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cudnn_allow_tf32": False,
            "verification": "PENDING_STRICT_UNIVERSITY_A40_PREFLIGHT",
        }
        result["cuda_determinism_fixture"] = {
            "passed": None,
            "operation": "torch.matmul",
            "verification": "PENDING_STRICT_UNIVERSITY_A40_PREFLIGHT",
            "training_or_evaluation_performed": False,
        }
        target = PACKAGE_PREFLIGHT
    write_json_once_or_verify(target, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2B frozen execution preflight")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--package-only", action="store_true", help="verify package without importing ML dependencies")
    modes.add_argument("--runtime", action="store_true", help="strict UNIVERSITY_A40 runtime gate")
    args = parser.parse_args()
    print(json.dumps(run_preflight(runtime=bool(args.runtime)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
