from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.remote.stage2b_a40 import (
    AUTHORIZATION,
    EXECUTION_ENVIRONMENT,
    FINAL_TARGETS,
    FROZEN_MANIFEST,
    FROZEN_SEAL_SHA256,
    INCIDENT_RECORD,
    JOB_MAP,
    PACKAGE_MANIFEST,
    PACKAGE_PREFLIGHT,
    PACKAGE_ROOT,
    REMOTE_OUTPUT,
    REQUIRED_A40_RUNTIME,
    ROOT,
    assert_output_isolation,
    build_job_rows,
    load_and_verify_frozen_seal,
    read_json,
    sha256_file,
    split_roles,
    validate_job_rows,
    write_or_verify_job_map,
)


REPORT = PACKAGE_ROOT / "STAGE2B_EXECUTION_PACKAGE_SEAL.md"
HASH_LEDGER = PACKAGE_ROOT / "STAGE2B_EXECUTION_PACKAGE.sha256"
INCIDENT_REPORT = PACKAGE_ROOT / "STAGE2B_CUBLAS_DETERMINISM_INCIDENT_REPORT.md"

SOURCE_FILES = (
    "research/governance/stage2b_execution_package.py",
    "research/remote/stage2b_a40.py",
    "research/development/stage2b_vanilla_execution.py",
    "research/scripts/preflight_stage2b_a40.py",
    "research/scripts/run_stage2b_remote_a40.py",
    "research/scripts/validate_stage2b_remote_a40.py",
    "research/scripts/verify_stage2b_execution_package.py",
    "research/tests/test_stage2b_execution_package.py",
    "research/tests/test_stage2b_cache_builder.py",
    "research/tests/test_stage2b_determinism_r2.py",
    "STAGE2B_REMOTE_EXECUTION_GUIDE.md",
    "requirements_stage2b_remote.txt",
)
GOVERNING_FILES = (
    "RESEARCH_PROTOCOL_V2_1.md", "COMPUTATION_DAG_V2_1.md",
    "TRAINING_BUDGET_PROTOCOL.md", "EVALUATION_PROTOCOL_V2.md",
    "DEVELOPMENT_SELECTION_PROTOCOL.md", "FEATURE_PROTOCOL_V2_1.md",
    "MODEL_SPECIFICATION_V2_1.md", "NEURAL_ARCHITECTURE_SPEC.md",
    "NEURAL_REPRODUCIBILITY_PROTOCOL.md",
)
UPSTREAM_MANIFESTS = (
    "research/results/stage1_scale_loss_v2_1/stage1_decision_manifest.json",
    "research/results/stage2_graphgru_selection_v2_1_a40/stage2_decision_manifest.json",
    "research/results/stage3_finetuning_policy_v2_1/stage3_finetuning_policy_manifest.json",
    "research/results/pre_stage4_development_audit_v2_1/pre_stage4_audit_manifest.json",
    "research/results/preseal_method_audit_v2_1/preseal_audit_manifest.json",
    "research/results/stage4_adversarial_method_v2_1/stage4_adversarial_method_manifest.json",
    "research/results/stage2b_vanilla_fairness_v2_1/stage2b_fairness_manifest.json",
)
DATA_BINDINGS = (
    "processed/protocol_v2_1/development/development_panel.parquet",
    "processed/protocol_v2_1/manifests/split_manifest.json",
    "processed/protocol_v2_1/manifests/feature_manifest.json",
    "processed/protocol_v2_1/station_manifests/station_manifest.json",
    "processed/protocol_v2_1/PROTOCOL_DATASET_MANIFEST.json",
)


def _hashes(paths: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in paths:
        path = ROOT / value
        if not path.is_file():
            raise FileNotFoundError(path)
        result[value] = sha256_file(path)
    return result


def _write_text_once(path: Path, text: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"Append-only artifact differs from existing file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def expected_manifest() -> dict[str, Any]:
    frozen = load_and_verify_frozen_seal()
    roles = split_roles()
    rows = build_job_rows()
    counts = validate_job_rows(rows)
    preflight = read_json(PACKAGE_PREFLIGHT)
    if preflight.get("status") != "PASS" or preflight.get("gate") != "PACKAGE_INTEGRITY_ONLY":
        raise RuntimeError("Package-only preflight has not passed")
    return {
        "schema_version": "1.0", "protocol_version": "2.1", "stage": "2B",
        "package_revision": "R2_CUBLAS_DETERMINISM_FIX",
        "seal_type": "append_only_stage2b_remote_execution_package",
        "status": "FROZEN_VERIFIED_PACKAGE",
        "execution_environment": EXECUTION_ENVIRONMENT,
        "frozen_stage2b_manifest_sha256": FROZEN_SEAL_SHA256,
        "frozen_stage1_decision": frozen["frozen_stage1_decision"],
        "frozen_stage2_decision_unchanged": frozen["frozen_stage2_decision_unchanged"],
        "append_only": True,
        "supersedes_execution_package": {
            "path": "research/results/stage2b_execution_package_v2_1_r1/stage2b_execution_package_manifest.json",
            "sha256": "32d9bbe8f9df15bce968a7876df19f66b2951f8bee054a796fa2a0dbbc7f92c4",
            "authorization_sha256": "f3b04938d6c8e054c4e868f2c0537857f467edb88dfb4e68689df2b3d48ceff0",
            "reason": "first-forward deterministic CUDA failure caused by missing pre-import CuBLAS workspace configuration",
        },
        "execution_package_lineage": [
            {
                "revision": "R0",
                "manifest_sha256": "6cb1aaeadfe2648709a6746b3b54928cfaa43eaf2ad5e7a72beb1c7f56d57711",
                "authorization_sha256": "a00368c31724497699b9301676b7dccb3ee7012b84f904ab67afc84172727b53",
            },
            {
                "revision": "R1_CACHE_FILL_FIX",
                "manifest_sha256": "32d9bbe8f9df15bce968a7876df19f66b2951f8bee054a796fa2a0dbbc7f92c4",
                "authorization_sha256": "f3b04938d6c8e054c4e868f2c0537857f467edb88dfb4e68689df2b3d48ceff0",
            },
        ],
        "remote_attempt_incident": {
            "path": INCIDENT_RECORD.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(INCIDENT_RECORD),
            "status": "ABORTED_FIRST_FORWARD_MISSING_CUBLAS_WORKSPACE_CONFIG",
            "audit_report": INCIDENT_REPORT.relative_to(ROOT).as_posix(),
            "audit_report_sha256": sha256_file(INCIDENT_REPORT),
        },
        "scientific_decisions_changed": False,
        "candidate_space": frozen["candidate_grid"],
        "selection_rule": frozen["selection_rule"],
        "source_policy": frozen["source_policy"],
        "job_counts": counts,
        "workload": {"source_fits": 48, "zero_shot_evaluations": 48, "fine_tuning_fits": 0},
        "job_map": {"path": JOB_MAP.relative_to(ROOT).as_posix(), "sha256": sha256_file(JOB_MAP)},
        "remote_output_root": REMOTE_OUTPUT.relative_to(ROOT).as_posix(),
        "required_runtime": REQUIRED_A40_RUNTIME,
        "runtime_dependency_policy": "active_university_environment_only_no_vendored_pydeps_or_sys_path_injection",
        "deterministic_runtime_contract": {
            "cublas_workspace_config": ":4096:8",
            "set_automatically_before_torch_import": True,
            "reject_conflicting_preexisting_value": True,
            "deterministic_algorithms": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cudnn_allow_tf32": False,
            "strict_remote_fixture": "repeated 64x64 CUDA torch.matmul with exact equality",
        },
        "final_target_city_ids_derived_from_authoritative_split": list(roles["final"]),
        "firewall": {
            "final_target_labels_accessed": False,
            "final_evaluation_windows_accessed": False,
            "final_target_derived_statistics_used": False,
            "stage4_performance_accessed": False,
            "training_or_evaluation_triggered_by_seal": False,
            "scientific_results_present": False,
            "allowed_panel": "development/development_panel.parquet only",
        },
        "preflight": {"path": PACKAGE_PREFLIGHT.relative_to(ROOT).as_posix(), "sha256": sha256_file(PACKAGE_PREFLIGHT), "runtime_gate": "PENDING_UNIVERSITY_A40"},
        "source_integrity": _hashes(SOURCE_FILES),
        "governing_input_sha256": _hashes(GOVERNING_FILES),
        "upstream_sha256": _hashes(UPSTREAM_MANIFESTS),
        "data_binding_sha256": _hashes(DATA_BINDINGS),
        "unresolved_final_execution_blockers": frozen["unresolved_final_execution_blockers"],
        "does_not_authorize": ["local_stage2b_execution", "stage4_execution", "final_target_access", "final_execution"],
    }


def generate() -> dict[str, Any]:
    PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)
    write_or_verify_job_map()
    assert_output_isolation(require_empty=True)
    manifest = expected_manifest()
    if PACKAGE_MANIFEST.is_file() and read_json(PACKAGE_MANIFEST) != manifest:
        raise RuntimeError("Append-only Stage-2B package manifest differs")
    if not PACKAGE_MANIFEST.is_file():
        PACKAGE_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_hash = sha256_file(PACKAGE_MANIFEST)
    report = f"""# Stage 2B remote execution package seal

Status: **FROZEN_VERIFIED_PACKAGE — R2 CuBLAS DETERMINISM FIX**  
Scientific execution: **conditionally authorized only on UNIVERSITY_A40 after strict runtime preflight PASS**  
Training/evaluation performed during sealing: **none**

This R2 package supersedes R1 after the UNIVERSITY_A40 launch aborted in the first CUDA model forward, before any completed fit, optimizer update, evaluation, or scientific result. R2 establishes `CUBLAS_WORKSPACE_CONFIG=:4096:8` automatically before PyTorch import, rejects conflicting values, mirrors the successful ordinary Stage 2 deterministic flags, and adds a strict CUDA matrix-multiplication fixture. No scientific semantics, grid, seed, budget, or evaluation rule changed.

The package implements the frozen station-independent vanilla-GRU fairness comparison without changing Stages 1–3 or the Stage 2B scientific contract. The exact immutable job ledger contains 48 source-pretraining fits and 48 zero-shot pseudo-target evaluations, with no adaptation or fine-tuning fit.

## Frozen map and isolation

- Four configurations: H32/H64 × dropout 0.0/0.1.
- Four pseudo-target folds: Bilbao, Cardiff, Freiburg, Göteborg.
- Three seeds: 17, 29, 43.
- Every fold excludes its pseudo-target and uses the other seven development cities.
- Output is isolated at `research/results/stage2b_vanilla_fairness_v2_1_a40/`.
- Manifest SHA-256: `{manifest_hash}`.
- Job-map SHA-256: `{sha256_file(JOB_MAP)}`.
- Frozen Stage-2B fairness seal SHA-256: `{FROZEN_SEAL_SHA256}`.

## Firewall

Final targets are derived mechanically from the authoritative split as Mannheim (195), Innsbruck (199), Glasgow (237), and Split (617). Their labels, final evaluation windows, final-derived statistics, and prediction keys are inaccessible to the runner. Stage 4 outputs are outside the execution scope. Package creation produced no scientific artifact.

The package-only preflight is not a hardware authorization. On university compute, the strict preflight must verify Python 3.10, PyTorch 2.0.1+cu118, NumPy 1.26.4, pandas 2.2.3, DuckDB 1.5.5, psutil 5.9.8, CUDA, one NVIDIA A40, non-vendored import provenance, all sealed deterministic flags, and a successful repeated CUDA `torch.matmul` fixture before cache construction or training.

The unresolved 3-vs-5 final-seed contradiction and seed × temporal-bootstrap aggregation rule remain recorded and unresolved because they do not govern this development-only Stage 2B computation.
"""
    _write_text_once(REPORT, report)
    authorization = {
        "schema_version": "1.0", "stage": "2B",
        "package_revision": "R2_CUBLAS_DETERMINISM_FIX",
        "status": "AUTHORIZED_CONDITIONAL_UNIVERSITY_A40",
        "scope": "FROZEN_STAGE2B_48_SOURCE_FITS_AND_48_ZERO_SHOT_EVALUATIONS_ONLY",
        "execution_environment": EXECUTION_ENVIRONMENT,
        "requires_strict_remote_runtime_preflight_pass": True,
        "local_execution_authorized": False,
        "training_started": False, "evaluation_started": False,
        "scientific_results_present_at_authorization": False,
        "frozen_stage2b_manifest_sha256": FROZEN_SEAL_SHA256,
        "execution_package_manifest_sha256": manifest_hash,
        "job_map_sha256": sha256_file(JOB_MAP),
        "package_preflight_sha256": sha256_file(PACKAGE_PREFLIGHT),
        "final_label_firewall_passed": True,
        "stage4_or_final_execution_authorized": False,
        "supersedes_execution_package_manifest_sha256": "32d9bbe8f9df15bce968a7876df19f66b2951f8bee054a796fa2a0dbbc7f92c4",
        "remote_stage2b_prior_attempt": "ABORTED_FIRST_FORWARD_MISSING_CUBLAS_WORKSPACE_CONFIG",
        "incident_record_sha256": sha256_file(INCIDENT_RECORD),
        "strict_cuda_matmul_fixture_required": True,
    }
    if AUTHORIZATION.is_file() and read_json(AUTHORIZATION) != authorization:
        raise RuntimeError("Append-only Stage-2B authorization differs")
    if not AUTHORIZATION.is_file():
        AUTHORIZATION.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ledger_files = [INCIDENT_RECORD, INCIDENT_REPORT, JOB_MAP, PACKAGE_PREFLIGHT, PACKAGE_MANIFEST, REPORT, AUTHORIZATION]
    ledger = "".join(f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in ledger_files)
    _write_text_once(HASH_LEDGER, ledger)
    return {"manifest_sha256": manifest_hash, "authorization_sha256": sha256_file(AUTHORIZATION), "job_map_sha256": sha256_file(JOB_MAP), "hash_ledger_sha256": sha256_file(HASH_LEDGER)}
