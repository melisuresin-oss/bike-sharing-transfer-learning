"""Capability-based final-label firewall. Merely importing this module opens nothing."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from . import core

FINAL_LABEL_ROLE = "V2_2_FINAL_RETROSPECTIVE_EVALUATION_LABELS"
PHASE_A = "PRE_LABEL_EXECUTION_AND_PREDICTION_COMMITMENT"
PHASE_B = "POST_COMMIT_LABEL_SCORING"

@dataclass(frozen=True)
class LabelCapability:
    authorization_sha256: str
    prediction_commitment_sha256: str
    label_binding_sha256: str

def reject_labelish_path(relative: str) -> None:
    low = relative.replace("\\", "/").lower()
    if any(x in low for x in ("final_labels", "sealed_final", "final_evaluation_labels")):
        raise PermissionError("final-label path unavailable in Phase A")

def phase_a_path(relative: str) -> Path:
    reject_labelish_path(relative)
    return core.repo_path(relative)

def bind_opaque_file(path: Path) -> dict:
    """Hash bytes without parsing or reporting any scientific content."""
    return {"role": FINAL_LABEL_ROLE, "path_name": path.name,
            "bytes": path.stat().st_size, "sha256": core.sha256_path(path),
            "content_parsed": False}

def authorize_phase_b(capability_path: Path, commitment_path: Path, label_binding_path: Path) -> LabelCapability:
    auth = json.loads(capability_path.read_text(encoding="utf-8"))
    commit_sha = core.sha256_path(commitment_path)
    binding_sha = core.sha256_path(label_binding_path)
    if auth.get("status") != "FINAL_LABEL_SCORING_AUTHORIZED":
        raise PermissionError("separate Phase B authorization absent")
    if auth.get("prediction_commitment_sha256") != commit_sha or auth.get("label_binding_sha256") != binding_sha:
        raise PermissionError("Phase B authorization binding mismatch")
    return LabelCapability(core.sha256_path(capability_path), commit_sha, binding_sha)
