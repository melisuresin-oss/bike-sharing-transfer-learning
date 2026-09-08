from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SEAL_ROOT = ROOT / "research" / "results" / "stage3_finetuning_policy_v2_1"

STAGE1_DECISION = (
    ROOT
    / "research"
    / "results"
    / "stage1_scale_loss_v2_1"
    / "stage1_decision_manifest.json"
)
STAGE2_DECISION = (
    ROOT
    / "research"
    / "results"
    / "stage2_graphgru_selection_v2_1_a40"
    / "stage2_decision_manifest.json"
)

FROZEN_UPSTREAM_SHA256 = {
    "research/results/stage1_scale_loss_v2_1/stage1_decision_manifest.json": (
        "f35a38fe936ca9389d5bf08ee08c5ec61920e588f6ef029044cf5f4fdf040d31"
    ),
    "research/results/stage2_graphgru_selection_v2_1_a40/"
    "stage2_decision_manifest.json": (
        "593dd284acc5d4e2da0e1d74ace84e85396894c05f3b738c01d64421d232e408"
    ),
}

AUTHORITATIVE_INPUT_SHA256 = {
    "RESEARCH_PROTOCOL_V2_1.md": (
        "6bbf5bc39292ef9e1bf8321b90006269217b43fd0df59da25687712fc418c35c"
    ),
    "COMPUTATION_DAG_V2_1.md": (
        "48f38bc97505d622316008b8ac155fb09e2b9a9efce6f36dad7e15b35d9f53cf"
    ),
    "TRAINING_BUDGET_PROTOCOL.md": (
        "479d87f18095611b1d488a7f1b932c160e688d11a8776c1b0c6d85b21af81940"
    ),
    "EVALUATION_PROTOCOL_V2.md": (
        "ebde12ca004508645e876e66d1899eb23f36410751ef0d102029c602874549ee"
    ),
    "DEVELOPMENT_SELECTION_PROTOCOL.md": (
        "14a1778ceb6437c6187837f1e34822164e4f32ae2e4d0f5c8f840c0576d5eb97"
    ),
    "MODEL_SPECIFICATION_V2_1.md": (
        "430fd12c6322a102687c2a546be394bb7ef344656c1a34edf05a86beadca88bf"
    ),
    "TEMPORAL_SPLIT_PROTOCOL_V2.md": (
        "f02d22ac5794b9f892180336e421cbe3cc8dc77e3c4d27177272adf146d11531"
    ),
    "NEURAL_REPRODUCIBILITY_PROTOCOL.md": (
        "35f5ecdca9dea4553455130ea3883e2d5bce07758a26c04958dd67865249bab9"
    ),
}


@dataclass(frozen=True)
class Stage3FineTuningPolicy:
    full_network_fine_tuning: bool = True
    reset_adamw_optimizer_state_before_adaptation: bool = True
    optimizer: str = "AdamW"
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    global_gradient_clip: float = 1.0
    early_stopping: bool = False
    update_budgets: tuple[int, ...] = (0, 100, 300, 600, 1200)
    zero_budget_is_fine_tuning_fit: bool = False
    adaptation_sampling: str = "with_replacement_within_elapsed_time_budget"
    causal_pre_budget_history_as_input: bool = True
    causal_pre_budget_history_as_extra_adaptation_target: bool = False


REGISTERED_POLICY = Stage3FineTuningPolicy()
REGISTERED_BACKBONE = {
    "config_id": "GGRU_K04_H032_D00",
    "model_type": "Graph-GRU",
    "graph_k": 4,
    "hidden_size": 32,
    "dropout": 0.0,
    "scale_loss_formulation": "LOG1P_TARGET_MAE",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_dict(policy: Stage3FineTuningPolicy = REGISTERED_POLICY) -> dict[str, Any]:
    value = asdict(policy)
    value["update_budgets"] = list(policy.update_budgets)
    return value


def validate_policy(policy: Stage3FineTuningPolicy) -> None:
    if policy != REGISTERED_POLICY:
        raise ValueError("Stage 3 policy differs from the frozen registered policy")
    if policy.update_budgets != (0, 100, 300, 600, 1200):
        raise ValueError("Unregistered Stage 3 update budget")
    if not policy.reset_adamw_optimizer_state_before_adaptation:
        raise ValueError("Stage 3 requires a reset AdamW state")
    if policy.early_stopping:
        raise ValueError("Stage 3 prohibits early stopping")
    if policy.zero_budget_is_fine_tuning_fit:
        raise ValueError("The zero-update reference is not a fine-tuning fit")


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_hash(relative_path: str, expected: str) -> None:
    path = ROOT / relative_path
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"Frozen input hash mismatch for {relative_path}: {actual} != {expected}"
        )


def verify_frozen_upstream() -> None:
    for relative_path, expected in FROZEN_UPSTREAM_SHA256.items():
        _assert_hash(relative_path, expected)

    stage1 = _read_json(STAGE1_DECISION)
    if stage1.get("stage") != 1:
        raise RuntimeError("Invalid Stage 1 decision stage")
    if stage1.get("SCALE_LOSS_FORMULATION") != "LOG1P_TARGET_MAE":
        raise RuntimeError("Frozen Stage 1 scale/loss decision changed")

    stage2 = _read_json(STAGE2_DECISION)
    if stage2.get("stage") != 2:
        raise RuntimeError("Invalid Stage 2 decision stage")
    if stage2.get("ORDINARY_GRAPH_GRU_CONFIG") != "GGRU_K04_H032_D00":
        raise RuntimeError("Frozen Stage 2 backbone decision changed")
    winning = stage2.get("winning_configuration", {})
    required = {
        "config_id": "GGRU_K04_H032_D00",
        "graph_k": 4,
        "hidden_size": 32,
        "dropout": 0.0,
        "loss_output_formulation": "LOG1P_TARGET_MAE",
    }
    if any(winning.get(key) != value for key, value in required.items()):
        raise RuntimeError("Frozen Stage 2 winning configuration changed")
    if stage2.get("sealed_final_evaluation_labels_accessed") is not False:
        raise RuntimeError("Stage 2 final-label firewall is not closed")


def verify_authoritative_inputs() -> None:
    for relative_path, expected in AUTHORITATIVE_INPUT_SHA256.items():
        _assert_hash(relative_path, expected)


def verify_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("seal_type") != "append_only_stage3_fine_tuning_policy":
        raise RuntimeError("Invalid Stage 3 seal type")
    if manifest.get("status") != "FROZEN_VERIFIED":
        raise RuntimeError("Stage 3 policy is not frozen and verified")
    if manifest.get("policy") != policy_dict():
        raise RuntimeError("Manifest policy differs from the registered policy")
    if manifest.get("selected_backbone") != REGISTERED_BACKBONE:
        raise RuntimeError("Manifest backbone differs from the frozen Stage 2 winner")

    firewall = manifest.get("firewall", {})
    required_false = (
        "final_target_labels_accessed",
        "training_triggered",
        "evaluation_triggered",
        "performance_decision_created",
        "stage3_execution_authorized",
        "stage4_started",
    )
    if any(firewall.get(key) is not False for key in required_false):
        raise RuntimeError("Stage 3 policy-seal firewall is not closed")
    if firewall.get("policy_seal_only") is not True:
        raise RuntimeError("Manifest authorizes more than the policy seal")


def verify_seal_bundle(seal_root: Path = SEAL_ROOT) -> dict[str, Any]:
    validate_policy(REGISTERED_POLICY)
    verify_frozen_upstream()
    verify_authoritative_inputs()

    manifest_path = seal_root / "stage3_finetuning_policy_manifest.json"
    document_path = seal_root / "STAGE3_FINE_TUNING_POLICY_SEAL.md"
    ledger_path = seal_root / "STAGE3_FINE_TUNING_POLICY_SEAL.sha256"
    manifest = _read_json(manifest_path)
    verify_manifest(manifest)

    for relative_path, expected in manifest.get("artifact_integrity", {}).items():
        _assert_hash(relative_path, expected)

    ledger: dict[str, str] = {}
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        digest, relative_path = line.split("  ", maxsplit=1)
        ledger[relative_path] = digest
    required_ledger = {
        str(manifest_path.relative_to(ROOT)).replace("\\", "/"): sha256_file(
            manifest_path
        ),
        str(document_path.relative_to(ROOT)).replace("\\", "/"): sha256_file(
            document_path
        ),
    }
    if ledger != required_ledger:
        raise RuntimeError("Stage 3 seal hash ledger mismatch")

    return {
        "status": "PASS",
        "seal_type": manifest["seal_type"],
        "selected_backbone": manifest["selected_backbone"]["config_id"],
        "scale_loss_formulation": manifest["selected_backbone"][
            "scale_loss_formulation"
        ],
        "update_budgets": manifest["policy"]["update_budgets"],
        "stage3_training_started": False,
        "final_target_labels_accessed": False,
        "downstream_execution_plan_authorized": False,
    }
