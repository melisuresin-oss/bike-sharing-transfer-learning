from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SEAL_ROOT = ROOT / "research" / "results" / "stage2b_vanilla_fairness_v2_1"

UPSTREAM_SHA256 = {
    "research/results/stage1_scale_loss_v2_1/stage1_decision_manifest.json": "f35a38fe936ca9389d5bf08ee08c5ec61920e588f6ef029044cf5f4fdf040d31",
    "research/results/stage2_graphgru_selection_v2_1_a40/stage2_decision_manifest.json": "593dd284acc5d4e2da0e1d74ace84e85396894c05f3b738c01d64421d232e408",
    "research/results/stage3_finetuning_policy_v2_1/stage3_finetuning_policy_manifest.json": "c63528bfe73881eafe02167f5cfddf37e51bca769fab6289850a66836642a6c1",
    "research/results/pre_stage4_development_audit_v2_1/pre_stage4_audit_manifest.json": "c1fb8fd309ab4168691dccf5a2cb4a9b2d493a98865c45b4765e30abdcab6873",
    "research/results/preseal_method_audit_v2_1/preseal_audit_manifest.json": "296bc0d0b16a9ace6ad057fb6df8baedd588a10032e4eb7a109c5f8f1d48d4b6",
}

GOVERNING_INPUTS = (
    "RESEARCH_PROTOCOL_V2_1.md",
    "COMPUTATION_DAG_V2_1.md",
    "DEVELOPMENT_SELECTION_PROTOCOL.md",
    "TRAINING_BUDGET_PROTOCOL.md",
    "EVALUATION_PROTOCOL_V2.md",
    "MODEL_SPECIFICATION_V2_1.md",
    "FEATURE_PROTOCOL_V2_1.md",
    "NEURAL_ARCHITECTURE_SPEC.md",
    "NEURAL_REPRODUCIBILITY_PROTOCOL.md",
    "processed/protocol_v2_1/development/development_panel.parquet",
    "processed/protocol_v2_1/manifests/split_manifest.json",
    "research/models/common.py",
    "research/models/vanilla_gru.py",
    "research/models/model_factory.py",
    "research/models/heads.py",
    "research/training/trainer.py",
    "research/data/window_dataset.py",
    "research/data/budget_sampler.py",
)

DEVELOPMENT_CITIES = {
    129: "Dortmund",
    194: "Heidelberg",
    438: "Marburg",
    467: "Gie\u00dfen",
    476: "Cardiff",
    532: "Bilbao",
    619: "Freiburg",
    658: "G\u00f6teborg",
}
PSEUDO_TARGETS = (532, 476, 619, 658)
SEEDS = (17, 29, 43)
HIDDEN_SIZES = (32, 64)
DROPOUTS = (0.0, 0.1)
SOURCE_POLICY = {
    "optimizer": "AdamW",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "betas": [0.9, 0.999],
    "epsilon": 1e-8,
    "source_updates": 12000,
    "batch_size": 16,
    "global_gradient_clip": 1.0,
    "checkpoint": "final_update",
    "early_stopping": False,
}
SELECTION_RULE = {
    "seed_aggregation": "arithmetic_mean_within_pseudo_target",
    "primary": "arithmetic_mean_of_four_pseudo_target_seed_mean_count_space_mae",
    "primary_quantization": "decimal_ROUND_HALF_UP_to_4_places",
    "tie_1": "lower_population_standard_deviation_across_four_pseudo_target_seed_means",
    "tie_2": "smaller_hidden_size",
    "tie_3": "lower_dropout",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_stage2_seed(label: str, fold_city_id: int, registered_seed: int) -> int:
    payload = f"stage2-v2.1|{label}|{fold_city_id}|{registered_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def quantize_primary(value: float) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def build_run_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hidden_size in HIDDEN_SIZES:
        for dropout in DROPOUTS:
            config_id = f"VGRU_H{hidden_size:03d}_D{int(round(dropout * 100)):02d}"
            for pseudo_target in PSEUDO_TARGETS:
                sources = tuple(city for city in DEVELOPMENT_CITIES if city != pseudo_target)
                for seed in SEEDS:
                    rows.append(
                        {
                            "status": "SEALED_NOT_AUTHORIZED_FOR_EXECUTION",
                            "job_id": f"vgru_h{hidden_size:03d}_d{int(round(dropout * 100)):02d}_target-{pseudo_target}_seed-{seed}",
                            "config_id": config_id,
                            "model_family": "genuine_vanilla_gru",
                            "hidden_size": hidden_size,
                            "dropout": dropout,
                            "pseudo_target_city_id": pseudo_target,
                            "pseudo_target": DEVELOPMENT_CITIES[pseudo_target],
                            "source_city_ids": ";".join(str(value) for value in sources),
                            "source_cities": ";".join(DEVELOPMENT_CITIES[value] for value in sources),
                            "seed": seed,
                            "initialization_seed": seed,
                            "source_city_schedule_seed": deterministic_stage2_seed(
                                "source-city-schedule", pseudo_target, seed
                            ),
                            "source_anchor_sampling_seed": deterministic_stage2_seed(
                                "source-anchor-sampling", pseudo_target, seed
                            ),
                            "source_updates": SOURCE_POLICY["source_updates"],
                            "batch_size": SOURCE_POLICY["batch_size"],
                            "optimizer": SOURCE_POLICY["optimizer"],
                            "learning_rate": SOURCE_POLICY["learning_rate"],
                            "weight_decay": SOURCE_POLICY["weight_decay"],
                            "global_gradient_clip": SOURCE_POLICY["global_gradient_clip"],
                            "early_stopping": SOURCE_POLICY["early_stopping"],
                            "scale_loss_formulation": "LOG1P_TARGET_MAE",
                            "source_pretraining_fit": True,
                            "zero_shot_evaluation": True,
                            "target_adaptation_fit": False,
                        }
                    )
    validate_run_plan(rows)
    return rows


def validate_run_plan(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 48 or len({row["job_id"] for row in rows}) != 48:
        raise RuntimeError("Stage 2B requires exactly 48 unique jobs")
    if {row["hidden_size"] for row in rows} != set(HIDDEN_SIZES):
        raise RuntimeError("Stage 2B hidden-size grid changed")
    if {row["dropout"] for row in rows} != set(DROPOUTS):
        raise RuntimeError("Stage 2B dropout grid changed")
    for row in rows:
        sources = {int(value) for value in row["source_city_ids"].split(";")}
        if len(sources) != 7 or row["pseudo_target_city_id"] in sources:
            raise RuntimeError("Stage 2B pseudo-target exclusion failed")
        if sources | {row["pseudo_target_city_id"]} != set(DEVELOPMENT_CITIES):
            raise RuntimeError("Stage 2B source membership changed")
        if row["target_adaptation_fit"] or not row["zero_shot_evaluation"]:
            raise RuntimeError("Stage 2B is source-only with zero-shot evaluation")


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_upstream() -> None:
    for relative, expected in UPSTREAM_SHA256.items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen upstream hash mismatch: {relative}")
    stage1 = _read_json(ROOT / next(path for path in UPSTREAM_SHA256 if "stage1_decision" in path))
    stage2 = _read_json(ROOT / next(path for path in UPSTREAM_SHA256 if "stage2_decision" in path))
    if stage1.get("SCALE_LOSS_FORMULATION") != "LOG1P_TARGET_MAE":
        raise RuntimeError("Stage 1 decision changed")
    if stage2.get("ORDINARY_GRAPH_GRU_CONFIG") != "GGRU_K04_H032_D00":
        raise RuntimeError("Stage 2 winner changed")


def render_document() -> str:
    return """# Stage 2B Vanilla-GRU Baseline-Fairness Seal

**Status:** `FROZEN_VERIFIED`  
**Authorization:** policy and 48-job enumeration are sealed; scientific execution is not authorized.

## Purpose and chronology

This append-only development amendment gives the genuine vanilla-GRU family an independent bounded selection opportunity. It was introduced after Stage 2 in response to the baseline-fairness audits and before any final-target label was opened. It does not reopen Stage 2: `GGRU_K04_H032_D00` remains the permanently frozen Graph-GRU winner, and Stage 1 remains `LOG1P_TARGET_MAE`.

## Frozen Stage 2B experiment

- Genuine vanilla GRU, never Graph-GRU with `k=0`.
- Hidden size in `{32,64}` and dropout in `{0.0,0.1}`: four configurations.
- Pseudo-targets Bilbao, Cardiff, Freiburg, and Goteborg; each is excluded from its seven-city source pool.
- Development seeds 17, 29, and 43.
- Frozen `LOG1P_TARGET_MAE` formulation.
- Frozen Stage 2 source policy: 12,000 updates, AdamW at `1e-3`, weight decay `1e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, batch size 16, global gradient clipping 1.0, final-update checkpoint, and no early stopping.
- Source pretraining only; one zero-shot pseudo-target evaluation per fit; no target adaptation or fine-tuning.

The sealed ledger contains `4 x 4 x 3 = 48` source fits and 48 zero-shot evaluations. No job has run.

## Frozen selection hierarchy

1. Average the three seed MAEs within each pseudo-target.
2. Compute the arithmetic mean of the four pseudo-target seed-mean count-space MAEs.
3. Compare that primary statistic after four-decimal decimal `ROUND_HALF_UP` quantization.
4. If tied, prefer lower population standard deviation across the four pseudo-target seed means.
5. If still tied, prefer smaller hidden size.
6. If still tied, prefer lower dropout.

This is the vanilla projection of Stage 2: `fold dispersion -> hidden -> k -> dropout` becomes `fold dispersion -> hidden -> dropout`.

## Fairness and reporting boundary

The fairness claim is limited to identical registered optimizer-update budgets and development-selection structure across the neural families. This seal does not claim equal parameter counts, equal wall-clock compute, or equal hardware cost. Later reporting must retain both the matched-capacity `vanilla H32/D0` versus `Graph-GRU K4/H32/D0` comparison and the best-development-selected vanilla-family versus frozen Graph-GRU comparison.

## Firewall and unresolved downstream issues

This seal accessed no final-target labels and triggered no training or evaluation. Stage 2B execution requires a separate execution plan/firewall authorization. The unresolved final three-versus-five seed contradiction and seed-by-temporal-bootstrap aggregation convention do not block this development seal and are not resolved here.
"""


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_seal_bundle(seal_root: Path = SEAL_ROOT) -> dict[str, Any]:
    verify_upstream()
    if seal_root.exists() and any(seal_root.iterdir()):
        raise FileExistsError(f"Stage 2B seal is append-only and already exists: {seal_root}")
    seal_root.mkdir(parents=True, exist_ok=True)
    plan = build_run_plan()
    plan_path = seal_root / "stage2b_run_plan.csv"
    document_path = seal_root / "STAGE2B_BASELINE_FAIRNESS_SEAL.md"
    _write_csv(plan_path, plan)
    document_path.write_text(render_document(), encoding="utf-8")
    governing_hashes = {relative: sha256_file(ROOT / relative) for relative in GOVERNING_INPUTS}
    artifacts = {
        document_path.relative_to(ROOT).as_posix(): sha256_file(document_path),
        plan_path.relative_to(ROOT).as_posix(): sha256_file(plan_path),
        "research/governance/stage2b_fairness_seal.py": sha256_file(
            ROOT / "research" / "governance" / "stage2b_fairness_seal.py"
        ),
        "research/scripts/verify_stage2b_fairness_seal.py": sha256_file(
            ROOT / "research" / "scripts" / "verify_stage2b_fairness_seal.py"
        ),
        "research/tests/test_stage2b_fairness_seal.py": sha256_file(
            ROOT / "research" / "tests" / "test_stage2b_fairness_seal.py"
        ),
    }
    manifest = {
        "schema_version": "1.0",
        "protocol_version": "2.1",
        "stage": "2B",
        "seal_type": "append_only_stage2b_vanilla_baseline_fairness",
        "status": "FROZEN_VERIFIED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "append_only": True,
        "introduced_after_stage2": True,
        "motivation": "baseline_fairness_audit_before_final_target_label_access",
        "frozen_stage1_decision": "LOG1P_TARGET_MAE",
        "frozen_stage2_decision_unchanged": "GGRU_K04_H032_D00",
        "candidate_grid": {
            "model_family": "genuine_vanilla_gru_not_graph_gru_k0",
            "hidden_size": list(HIDDEN_SIZES),
            "dropout": list(DROPOUTS),
            "pseudo_target_city_ids": list(PSEUDO_TARGETS),
            "seeds": list(SEEDS),
        },
        "source_policy": SOURCE_POLICY,
        "selection_rule": SELECTION_RULE,
        "workload": {
            "configurations": 4,
            "pseudo_target_folds": 4,
            "seeds": 3,
            "source_fits": 48,
            "zero_shot_evaluations": 48,
            "target_adaptation_fits": 0,
        },
        "fairness_criterion": {
            "matched": ["registered_optimizer_update_budget", "development_selection_structure"],
            "not_claimed_matched": ["parameter_count", "wall_clock_compute", "hardware_cost"],
        },
        "upstream_sha256": UPSTREAM_SHA256,
        "governing_input_sha256": governing_hashes,
        "artifact_integrity": artifacts,
        "firewall": {
            "seal_only": True,
            "stage2b_execution_authorized": False,
            "stage2b_training_triggered": False,
            "prediction_evaluation_triggered": False,
            "target_adaptation_triggered": False,
            "final_target_labels_accessed": False,
            "stage2_winner_reopened": False,
        },
        "unresolved_final_execution_blockers": [
            "final_seed_count_3_vs_5",
            "seed_x_temporal_bootstrap_aggregation",
        ],
    }
    manifest_path = seal_root / "stage2b_fairness_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ledger_paths = (manifest_path, document_path, plan_path)
    (seal_root / "STAGE2B_BASELINE_FAIRNESS_SEAL.sha256").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}\n"
            for path in ledger_paths
        ),
        encoding="utf-8",
    )
    return manifest


def verify_seal_bundle(seal_root: Path = SEAL_ROOT) -> dict[str, Any]:
    verify_upstream()
    manifest_path = seal_root / "stage2b_fairness_manifest.json"
    document_path = seal_root / "STAGE2B_BASELINE_FAIRNESS_SEAL.md"
    plan_path = seal_root / "stage2b_run_plan.csv"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "FROZEN_VERIFIED":
        raise RuntimeError("Stage 2B seal is not frozen and verified")
    if manifest.get("selection_rule") != SELECTION_RULE:
        raise RuntimeError("Stage 2B selection rule changed")
    firewall = manifest.get("firewall", {})
    if firewall.get("seal_only") is not True or any(
        firewall.get(key) is not False
        for key in (
            "stage2b_execution_authorized",
            "stage2b_training_triggered",
            "prediction_evaluation_triggered",
            "target_adaptation_triggered",
            "final_target_labels_accessed",
            "stage2_winner_reopened",
        )
    ):
        raise RuntimeError("Stage 2B seal firewall is not closed")
    for relative, expected in manifest["governing_input_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"Governing input changed: {relative}")
    for relative, expected in manifest["artifact_integrity"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"Stage 2B artifact changed: {relative}")
    with plan_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 48 or len({row["job_id"] for row in rows}) != 48:
        raise RuntimeError("Sealed Stage 2B plan is incomplete")
    for row in rows:
        if row["status"] != "SEALED_NOT_AUTHORIZED_FOR_EXECUTION":
            raise RuntimeError("A Stage 2B job was authorized")
        if row["target_adaptation_fit"] != "False":
            raise RuntimeError("Stage 2B plan contains target adaptation")
        sources = {int(value) for value in row["source_city_ids"].split(";")}
        if int(row["pseudo_target_city_id"]) in sources or len(sources) != 7:
            raise RuntimeError("Sealed Stage 2B fold exclusion failed")
    ledger = {}
    for line in (seal_root / "STAGE2B_BASELINE_FAIRNESS_SEAL.sha256").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        ledger[relative] = digest
    expected_ledger = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in (manifest_path, document_path, plan_path)
    }
    if ledger != expected_ledger:
        raise RuntimeError("Stage 2B hash ledger mismatch")
    return {
        "status": "PASS",
        "seal_type": manifest["seal_type"],
        "jobs_verified": 48,
        "stage2b_execution_authorized": False,
        "scientific_training_triggered": False,
        "final_target_labels_accessed": False,
        "frozen_stage2_decision": "GGRU_K04_H032_D00",
    }


def main() -> int:
    print(json.dumps(create_seal_bundle(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
