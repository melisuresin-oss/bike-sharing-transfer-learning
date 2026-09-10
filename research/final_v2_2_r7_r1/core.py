from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "deployment/final_v2_2_r7_r1"
SEAL = "research/results/final_v2_2_r7_r1_execution_seal"
OUT = "research/results/final_v2_2_r7_r1_a40"
PACKAGE_MANIFEST = PACKAGE + "/package_manifest.json"
MEMBER_MANIFEST = PACKAGE + "/package_member_manifest.json"
CODE_MANIFEST = SEAL + "/executable_code_manifest.json"
IMPLEMENTATION_CONTRACT = SEAL + "/implementation_contract.json"
JOB_MANIFEST = "research/results/final_v2_2_execution_seal/final_job_manifest.json"
EVALUATION_MANIFEST = "research/results/final_v2_2_execution_seal/evaluation_manifest.json"
DATA_MANIFEST = "processed/protocol_v2_2/FINAL_DATASET_MANIFEST.json"
PREDICTION_MANIFEST = OUT + "/predictions/prediction_manifest.json"
PREDICTION_COMMITMENT = OUT + "/predictions/prediction_commitment.json"

SEEDS = (17, 29, 43, 71, 101)
TARGETS = (195, 199, 237, 617)
SOURCES = (129, 194, 438, 467, 476, 532, 619, 658)
DOMAIN_MAP = {129: 0, 194: 1, 438: 2, 467: 3, 476: 4, 532: 5, 619: 6, 658: 7}
BUDGETS = ("0", "1", "7", "30", "full")
NONZERO_BUDGETS = ("1", "7", "30", "full")
UPDATES = {"0": 0, "1": 100, "7": 300, "30": 600, "full": 1200}
BOUNDARIES = {"H0": 1661749200000000, "HF": 1681682400000000,
              "HT": 1683817200000000, "HE": 1689451200000000}
TIMEZONES = {129: "Europe/Berlin", 194: "Europe/Berlin", 438: "Europe/Berlin",
             467: "Europe/Berlin", 476: "Europe/London", 532: "Europe/Madrid",
             619: "Europe/Berlin", 658: "Europe/Stockholm", 195: "Europe/Berlin",
             199: "Europe/Vienna", 237: "Europe/London", 617: "Europe/Zagreb"}

FROZEN = {
    "FINAL_EVALUATION_PROTOCOL_V2_2_AMENDMENT.md": "c4a8fb1cc314cd93336269b8b1ee035771ce766c823e82476daac97e3e565f6c",
    "final_evaluation_protocol_v2_2.json": "16617ca602e2d70cc621005a838dfb84d308b5f33488bac8b92d38fccea1d054",
    "FINAL_GRL_SOURCE_DOMAIN_CARDINALITY_CLARIFICATION_V2_2.md": "9e96cf5262861ec83d722ec6db18f97c66526a6e493a4970062635a2e4e25843",
    "final_grl_source_domain_cardinality_v2_2.json": "9324c11efac316607c39dd67d436e2efe57dd3fcf3bf2b7d2002d78494438037",
    DATA_MANIFEST: "afffa1e539626839fd86f1f9a3f5e8aa7a7833b67dde43f839d9cf1ce527896a",
    JOB_MANIFEST: "b415398fb627a89a46253bfed0089cf1e4d24b1ef5bc4847d77fa674f9aae908",
    EVALUATION_MANIFEST: "523b571672bcc27395d940dce8c4362ab2d21143c9590f756b6427ae694bb22e",
    "research/results/stage1_scale_loss_v2_2_a40/stage1_decision_manifest.json": "8db24acabd373fa57aa6f35708566da400646fa2ea8474fd0e656a4955828aed",
    "research/results/stage2_graphgru_selection_v2_2_a40/stage2_decision_manifest.json": "9cca604ec3496a28f4440d89a9d6d1184311fc507c35f27b428a3ca768a6eeca",
    "research/results/stage2b_vanilla_fairness_v2_2_a40/stage2b_decision_manifest.json": "9d85d6df6a509013f1b5411002246aee81c254fbde078f639293976674b4f36d",
    "research/results/stage3_finetuning_policy_v2_2/stage3_finetuning_policy_manifest.json": "f5cb2653dd097eb1288f419884778ae01e9034e9e1d88006b33beac4cff069b5",
    "research/results/stage4_source_invariance_v2_2/scientific_contract.json": "94e586624d18ac9047c6b0a71922139ad00795da54e20ca05d8ef04008470e15",
    "research/results/stage4_v2_2_returned_20260908/stage4_source_invariance_v2_2_a40/stage4_decision_manifest.json": "b8ab61745465fcb3c2db3ce811effaba091111a57eefe89f70d5fa5df3e1aae0",
}

SOURCE_PREDICTORS = {city: f"processed/protocol_v2_2/development/city_{city}_predictors.npz" for city in SOURCES}
SOURCE_GRAPHS = {city: f"tmp/stage2_graph_grid_cache_v2_1/k_04/city_{city}/adjacency.npy" for city in SOURCES}

def utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def path(relative: str, *, output: bool = False) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative or ".." in Path(relative).parts:
        raise PermissionError("canonical package-relative path required")
    low = relative.lower()
    if any(token in low for token in ("final_labels", "sealed_final", "final_evaluation_labels", "phase_b_targets")):
        raise PermissionError("Phase-B target path is inaccessible to R7 Phase A")
    result = (ROOT / relative).resolve()
    result.relative_to(ROOT)
    if output and not relative.startswith(OUT + "/"):
        raise PermissionError("R7 writes are restricted to the R7 output namespace")
    return result

def sha256_path(value: Path) -> str:
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def sha(relative: str) -> str:
    return sha256_path(path(relative))

def read_json(relative: str) -> Any:
    return json.loads(path(relative).read_text(encoding="utf-8"))

def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()

def atomic_bytes(relative: str, payload: bytes) -> dict[str, Any]:
    destination = path(relative, output=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(destination.name + ".pending-" + os.urandom(8).hex())
    try:
        with pending.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic same-volume publication with CREATE_NEW/no-clobber semantics.
        os.link(pending, destination)
    finally:
        try:
            pending.unlink()
        except FileNotFoundError:
            pass
    return entry(relative)

def atomic_json(relative: str, value: Any) -> dict[str, Any]:
    return atomic_bytes(relative, json.dumps(value, indent=2, ensure_ascii=False,
                                             allow_nan=False).encode("utf-8") + b"\n")

def entry(relative: str) -> dict[str, Any]:
    value = path(relative)
    return {"path": relative, "bytes": value.stat().st_size, "sha256": sha256_path(value)}

def verify_frozen() -> None:
    for relative, expected in FROZEN.items():
        if sha(relative) != expected:
            raise RuntimeError("frozen authority hash mismatch: " + relative)

def seed_for(job: dict[str, Any], purpose: str) -> int:
    text = f"final-v2.2-r6|{job['job_id']}|{purpose}|{job['seed']}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")

def runtime_contract() -> dict[str, Any]:
    return {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "torch_deterministic_algorithms": True,
            "cuda_matmul_tf32": False, "cudnn_benchmark": False,
            "cudnn_deterministic": True, "cudnn_tf32": False,
            "max_workers": 4, "fresh_python_process_per_fit": True,
            "partial_checkpoint_resume": False, "early_stopping": False}

def configure_torch(seed: int, device_name: str):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    import random
    import numpy as np
    import torch
    torch.set_num_threads(2)
    random.seed(seed); np.random.seed(seed & 0xFFFFFFFF); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device

