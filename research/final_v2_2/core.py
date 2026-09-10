from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "deployment/final_v2_2_a40"
SEAL = "research/results/final_v2_2_execution_seal"
OUT = "research/results/final_v2_2_a40"

SEEDS = (17, 29, 43, 71, 101)
TARGETS = ((195, "Mannheim"), (199, "Innsbruck"), (237, "Glasgow"), (617, "Split"))
SOURCES = (129, 194, 438, 467, 476, 532, 619, 658)
BUDGETS = ("0", "1", "7", "30", "full")
NONZERO_BUDGETS = ("1", "7", "30", "full")
UPDATES = {"0": 0, "1": 100, "7": 300, "30": 600, "full": 1200}
BOUNDARIES = {"H0": 1661749200000000, "HF": 1681682400000000,
              "HT": 1683817200000000, "HE": 1689451200000000}
BOOTSTRAP_NAMESPACE = "FINAL_EVALUATION_V2_2|TEMPORAL_BLOCK_BOOTSTRAP|20260908"
BOOTSTRAP_SEEDS = {195: 12648532355080271797, 199: 2634071097790620903,
                   237: 14234234470032652229, 617: 6032700276159629083}

AUTHORITY = {
    "FINAL_EVALUATION_PROTOCOL_V2_2_AMENDMENT.md": "c4a8fb1cc314cd93336269b8b1ee035771ce766c823e82476daac97e3e565f6c",
    "final_evaluation_protocol_v2_2.json": "16617ca602e2d70cc621005a838dfb84d308b5f33488bac8b92d38fccea1d054",
    "research/results/stage1_scale_loss_v2_2_a40/stage1_decision_manifest.json": "8db24acabd373fa57aa6f35708566da400646fa2ea8474fd0e656a4955828aed",
    "research/results/stage2_graphgru_selection_v2_2_a40/stage2_decision_manifest.json": "9cca604ec3496a28f4440d89a9d6d1184311fc507c35f27b428a3ca768a6eeca",
    "research/results/stage2b_vanilla_fairness_v2_2_a40/stage2b_decision_manifest.json": "9d85d6df6a509013f1b5411002246aee81c254fbde078f639293976674b4f36d",
    "research/results/stage3_finetuning_policy_v2_2/stage3_finetuning_policy_manifest.json": "f5cb2653dd097eb1288f419884778ae01e9034e9e1d88006b33beac4cff069b5",
    "research/results/stage4_v2_2_returned_20260908/stage4_source_invariance_v2_2_a40/stage4_decision_manifest.json": "b8ab61745465fcb3c2db3ce811effaba091111a57eefe89f70d5fa5df3e1aae0",
    "research/results/stage4_source_invariance_v2_2/scientific_contract.json": "94e586624d18ac9047c6b0a71922139ad00795da54e20ca05d8ef04008470e15",
}

SOURCE_BINDINGS = (
    "feasibility_test/full_audit/source_manifest.json",
    "feasibility_test/full_audit/coverage/station_status_source_manifest.json",
    "processed/protocol_v2_2/PROTOCOL_DATASET_MANIFEST.json",
    "research/results/causal_history_audit_v2_2/graph_static_reconciliation.json",
)

def utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def repo_path(relative: str, *, output: bool = False) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise PermissionError("canonical repository-relative path required")
    p = (ROOT / relative).resolve()
    p.relative_to(ROOT)
    if output and not (relative.startswith(OUT + "/") or relative.startswith(SEAL + "/") or relative.startswith(PACKAGE + "/")):
        raise PermissionError("write outside isolated final namespace")
    return p

def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def sha(relative: str) -> str:
    return sha256_path(repo_path(relative))

def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()

def read_json(relative: str) -> Any:
    return json.loads(repo_path(relative).read_text(encoding="utf-8"))

def atomic_write_json(relative: str, value: Any) -> dict[str, Any]:
    p = repo_path(relative, output=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError("append-only destination exists: " + relative)
    pending = p.with_name(p.name + ".pending-" + os.urandom(8).hex())
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"
    with pending.open("xb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    pending.replace(p)
    return {"path": relative, "bytes": p.stat().st_size, "sha256": sha256_path(p)}

def verify_authority() -> None:
    for rel, expected in AUTHORITY.items():
        if sha(rel) != expected:
            raise RuntimeError("frozen authority hash mismatch: " + rel)

def deterministic_runtime() -> dict[str, Any]:
    return {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "torch_deterministic_algorithms": True,
            "cuda_matmul_tf32": False, "cudnn_benchmark": False,
            "cudnn_deterministic": True, "cudnn_tf32": False,
            "workers": 4, "fresh_python_process_per_fit": True}
