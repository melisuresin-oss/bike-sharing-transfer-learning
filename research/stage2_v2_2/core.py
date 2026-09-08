"""Shared frozen definitions for Stage-2/Stage-2B V2.2.

This module contains scientific definitions and deterministic aggregation only.
It does not launch training or open retrospective development labels.
"""
from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import random
import statistics
from typing import Any

from research.models.common import NeuralModelConfig
from research.training.trainer import OptimizerConfig
from research.v2_2.contract import DEVELOPMENT, PSEUDO_TARGETS as CONTRACT_TARGETS, canonical_bytes


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_VERSION = "2.2"
ENVIRONMENT = "UNIVERSITY_A40"
DATA_MANIFEST = "processed/protocol_v2_2/PROTOCOL_DATASET_MANIFEST.json"
DATA_MANIFEST_SHA256 = "140dd8c53cf51f48be5cf66b70d162c9f2cc6a9fa0b1b7039496633e385a6ab6"
IMPLEMENTATION_MANIFEST = "research/results/causal_history_implementation_v2_2/implementation_manifest.json"
IMPLEMENTATION_MANIFEST_SHA256 = "a1fe0ee486c55b1348e92d91edd42289dd49ce982129cdaa544691eafd605a9d"
STAGE1_DECISION = "research/results/stage1_scale_loss_v2_2_a40/stage1_decision_manifest.json"
STAGE1_DECISION_SHA256 = "8db24acabd373fa57aa6f35708566da400646fa2ea8474fd0e656a4955828aed"
FIT_MANIFEST = "processed/protocol_v2_2/fit_snapshots/fit_snapshot_manifest.json"
FIT_MANIFEST_SHA256 = "467aefd170c789c6533a7f2f1ebc0f3c343944f8e2781c9e2e5e0927d200077d"
LABEL_MANIFEST = "processed/protocol_v2_2/development_labels/retrospective_label_manifest.json"
LABEL_MANIFEST_SHA256 = "1868d6a95641e43ef3ad779f8a4f779aaf94fc3cb938b3db13b77e2eb0dd3469"
BASE_CACHE = "tmp/stage1_scale_loss_cache_v2_2/"
BASE_CACHE_MANIFEST = BASE_CACHE + "cache_manifest.json"
BASE_CACHE_MANIFEST_SHA256 = "836e7d483a76a2253b5ec63180e23683d22ba1160d170e290d83e68d5d79756d"
GRAPH_RECONCILIATION = "research/results/causal_history_audit_v2_2/graph_static_reconciliation.json"
GRAPH_RECONCILIATION_SHA256 = "e425ce873da946a4973fb03a57ffd6ecf92027da82a4220f56b6231343ff9758"

STAGE2_ROOT = "research/results/stage2_graphgru_selection_v2_2/"
STAGE2B_ROOT = "research/results/stage2b_vanilla_fairness_v2_2/"
STAGE2_OUT = "research/results/stage2_graphgru_selection_v2_2_a40/"
STAGE2B_OUT = "research/results/stage2b_vanilla_fairness_v2_2_a40/"
JOINT_OUT = "research/results/stage2_stage2b_v2_2_a40/"
STAGE2_CONTRACT = STAGE2_ROOT + "scientific_contract.json"
STAGE2B_CONTRACT = STAGE2B_ROOT + "scientific_contract.json"
STAGE2_RULE = STAGE2_ROOT + "selection_rule.json"
STAGE2B_RULE = STAGE2B_ROOT + "selection_rule.json"
STAGE2_JOB_MAP = STAGE2_ROOT + "job_map.json"
STAGE2B_JOB_MAP = STAGE2B_ROOT + "job_map.json"

PSEUDO_TARGETS = (532, 476, 619, 658)
SEEDS = (17, 29, 43)
GRAPH_K = (4, 8, 16)
HIDDEN_SIZES = (32, 64)
DROPOUTS = (0.0, 0.1)
SOURCE_UPDATES = 12_000
BATCH_SIZE = 16
OUTPUT_MODE = "log1p_target"
GRADIENT_CLIP = 1.0
HISTORICAL_STAGE2_WINNER = "GGRU_K04_H032_D00"
HISTORICAL_STAGE2B_WINNER = "VGRU_H032_D00"

AUTHORITY_SHA256 = {
    "RESEARCH_PROTOCOL_V2_2_AMENDMENT.md": "02a44e3aa8bf1d656d2f9de833b45072d564052e2ab3c1426bda2a6456c558d0",
    "RESEARCH_PROTOCOL_V2_1.md": "6bbf5bc39292ef9e1bf8321b90006269217b43fd0df59da25687712fc418c35c",
    "COMPUTATION_DAG_V2_1.md": "48f38bc97505d622316008b8ac155fb09e2b9a9efce6f36dad7e15b35d9f53cf",
    "DEVELOPMENT_SELECTION_PROTOCOL.md": "14a1778ceb6437c6187837f1e34822164e4f32ae2e4d0f5c8f840c0576d5eb97",
    "TRAINING_BUDGET_PROTOCOL.md": "479d87f18095611b1d488a7f1b932c160e688d11a8776c1b0c6d85b21af81940",
    "EVALUATION_PROTOCOL_V2.md": "ebde12ca004508645e876e66d1899eb23f36410751ef0d102029c602874549ee",
    "MODEL_SPECIFICATION_V2_1.md": "430fd12c6322a102687c2a546be394bb7ef344656c1a34edf05a86beadca88bf",
    "NEURAL_ARCHITECTURE_SPEC.md": "ddc6875307518940f8cf98d5480b90984d984d996bf6aefa0cdffd0179542f33",
    "NEURAL_REPRODUCIBILITY_PROTOCOL.md": "35f5ecdca9dea4553455130ea3883e2d5bce07758a26c04958dd67865249bab9",
    "research/results/causal_history_spec_v2_2/v2_2_specification.json": "c85c8fdbcab26e7239bfb4528b570b720e7d31ea5933934d9258b63ad90f9277",
    "research/results/causal_history_spec_v2_2/causal_history_spec_seal.json": "170d2c3c8b3371dd6c6c3d0606733d6d9a6b93e83449808dd7ccb98a904a86fb",
    "research/results/causal_history_spec_v2_2/fixed_cohort_static_manifest.json": "24f25a4d18dafe888dffc74acc279e59e51ef5c68c03eaa9e8148c632ae4a046",
    "research/results/stage2_graphgru_selection_v2_1_a40/stage2_selection_rule.json": "658f7a9a89ccbb6c7184fae39684301a04569e4632dbb7fa56268f252546ccc5",
    "research/results/stage2b_vanilla_fairness_v2_1/stage2b_fairness_manifest.json": "6b3f7e4936e201c91023145a73e36b4a122e8a97305ef408bc14a6c727c6259f",
}

STAGE2_SELECTION_RULE = {
    "stage": 2,
    "candidate_family": "ordinary Graph-GRU",
    "metric": "MAE in original count space",
    "seed_aggregation": "arithmetic mean over seeds 17, 29, and 43 within each pseudo-target fold",
    "pseudo_target_aggregation": "unweighted arithmetic mean over the four seed-mean fold MAEs",
    "regime_aggregation": "none; Stage 2 contains only parameter zero-shot",
    "primary_order": "lower equal-fold count-space MAE",
    "tie_definition": "primary statistics equal after ROUND_HALF_UP quantization to four decimal places",
    "tie_break_hierarchy": [
        "lower population standard deviation across the four seed-mean fold MAEs",
        "smaller hidden size",
        "smaller graph k",
        "lower dropout",
    ],
    "selection_exclusions": ["RMSE", "WAPE", "station-macro MAE", "pooled station-hours"],
}

STAGE2B_SELECTION_RULE = {
    "seed_aggregation": "arithmetic_mean_within_pseudo_target",
    "primary": "arithmetic_mean_of_four_pseudo_target_seed_mean_count_space_mae",
    "primary_quantization": "decimal_ROUND_HALF_UP_to_4_places",
    "tie_1": "lower_population_standard_deviation_across_four_pseudo_target_seed_means",
    "tie_2": "smaller_hidden_size",
    "tie_3": "lower_dropout",
}


def repository_path(relative: str) -> Path:
    if not isinstance(relative, str) or "\\" in relative or ":" in relative or ".." in Path(relative).parts:
        raise PermissionError("Repository-relative path required")
    result = (ROOT / relative).resolve()
    lowered = result.relative_to(ROOT).as_posix().lower()
    if any(token in lowered for token in (
        "final_labels", "sealed_final_evaluation_labels", "final_adaptation", "final_features"
    )):
        raise PermissionError("Final-target firewall")
    return result


def sha256_file(relative: str) -> str:
    digest = hashlib.sha256()
    with repository_path(relative).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(relative: str) -> Any:
    return json.loads(repository_path(relative).read_text(encoding="utf-8"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(
        name="AdamW", learning_rate=1e-3, betas=(0.9, 0.999), epsilon=1e-8,
        weight_decay=1e-4,
    )


def graph_configurations() -> tuple[dict[str, Any], ...]:
    rows = []
    for graph_k in GRAPH_K:
        for hidden_size in HIDDEN_SIZES:
            for dropout in DROPOUTS:
                rows.append({
                    "config_id": f"GGRU_K{graph_k:02d}_H{hidden_size:03d}_D{int(round(dropout * 100)):02d}",
                    "model_family": "ordinary_graph_gru",
                    "graph_k": graph_k,
                    "hidden_size": hidden_size,
                    "dropout": dropout,
                })
    return tuple(rows)


def vanilla_configurations() -> tuple[dict[str, Any], ...]:
    rows = []
    for hidden_size in HIDDEN_SIZES:
        for dropout in DROPOUTS:
            rows.append({
                "config_id": f"VGRU_H{hidden_size:03d}_D{int(round(dropout * 100)):02d}",
                "model_family": "genuine_vanilla_gru",
                "graph_k": None,
                "hidden_size": hidden_size,
                "dropout": dropout,
            })
    return tuple(rows)


def model_config(configuration: dict[str, Any]) -> NeuralModelConfig:
    family = configuration["model_family"]
    if family not in ("ordinary_graph_gru", "genuine_vanilla_gru"):
        raise ValueError("Unregistered model family")
    return NeuralModelConfig(
        model_type="graph_gru" if family == "ordinary_graph_gru" else "vanilla_gru",
        input_size=2,
        hidden_size=int(configuration["hidden_size"]),
        context_size=10,
        dropout=float(configuration["dropout"]),
        output_mode=OUTPUT_MODE,
        recurrent_layers=1,
    )


def deterministic_seed(label: str, fold_city_id: int, registered_seed: int) -> int:
    """Carry forward the registered candidate-independent V2.1 RNG namespace."""
    payload = f"stage2-v2.1|{label}|{fold_city_id}|{registered_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def city_sequence(city_ids: tuple[int, ...], seed: int) -> list[int]:
    sequence = [city_ids[index % len(city_ids)] for index in range(SOURCE_UPDATES)]
    random.Random(seed).shuffle(sequence)
    counts = [sequence.count(city_id) for city_id in city_ids]
    if sum(counts) != SOURCE_UPDATES or max(counts) - min(counts) > 1:
        raise RuntimeError("Frozen equal-city source exposure failed")
    return sequence


def validate_authorities() -> dict[str, Any]:
    for relative, expected in AUTHORITY_SHA256.items():
        if sha256_file(relative) != expected:
            raise RuntimeError("Authoritative design source changed: " + relative)
    if sha256_file(DATA_MANIFEST) != DATA_MANIFEST_SHA256:
        raise RuntimeError("V2.2 development-data manifest changed")
    if sha256_file(IMPLEMENTATION_MANIFEST) != IMPLEMENTATION_MANIFEST_SHA256:
        raise RuntimeError("V2.2 implementation manifest changed")
    if sha256_file(STAGE1_DECISION) != STAGE1_DECISION_SHA256:
        raise RuntimeError("Frozen Stage-1 V2.2 decision changed")
    if sha256_file(FIT_MANIFEST) != FIT_MANIFEST_SHA256:
        raise RuntimeError("V2.2 fitting-snapshot manifest changed")
    if sha256_file(LABEL_MANIFEST) != LABEL_MANIFEST_SHA256:
        raise RuntimeError("V2.2 retrospective development-label manifest changed")
    if sha256_file(BASE_CACHE_MANIFEST) != BASE_CACHE_MANIFEST_SHA256:
        raise RuntimeError("Frozen V2.2 causal fitting cache changed")
    if sha256_file(GRAPH_RECONCILIATION) != GRAPH_RECONCILIATION_SHA256:
        raise RuntimeError("Reconciled static graph authority changed")

    stage1 = read_json(STAGE1_DECISION)
    if not (
        stage1.get("status") == "FROZEN_VALIDATED"
        and stage1.get("selected_candidate") == "LOG1P"
        and stage1.get("selected_mode") == OUTPUT_MODE
        and stage1.get("development_data_manifest_sha256") == DATA_MANIFEST_SHA256
        and stage1.get("implementation_manifest_sha256") == IMPLEMENTATION_MANIFEST_SHA256
        and stage1.get("local_cpu_results_used") is False
        and stage1.get("final_target_labels_accessed") is False
    ):
        raise RuntimeError("Stage-1 V2.2 decision does not authorize LOG1P-dependent development selection")
    if read_json("research/results/stage2_graphgru_selection_v2_1_a40/stage2_selection_rule.json") != STAGE2_SELECTION_RULE:
        raise RuntimeError("Stage-2 historical selection contract conflicts with the reconstructed rule")
    stage2b = read_json("research/results/stage2b_vanilla_fairness_v2_1/stage2b_fairness_manifest.json")
    expected_source = {
        "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4,
        "betas": [0.9, 0.999], "epsilon": 1e-8, "source_updates": SOURCE_UPDATES,
        "batch_size": BATCH_SIZE, "global_gradient_clip": GRADIENT_CLIP,
        "checkpoint": "final_update", "early_stopping": False,
    }
    if stage2b.get("selection_rule") != STAGE2B_SELECTION_RULE or stage2b.get("source_policy") != expected_source:
        raise RuntimeError("Stage-2B historical selection/source contract conflicts with reconstruction")
    if set(PSEUDO_TARGETS) != set(CONTRACT_TARGETS) or set(DEVELOPMENT) != {129, 194, 438, 467, 476, 532, 619, 658}:
        raise RuntimeError("V2.2 development identities changed")
    return {
        "status": "PASS",
        "contradictions": [],
        "stage1_decision_sha256": STAGE1_DECISION_SHA256,
        "development_data_manifest_sha256": DATA_MANIFEST_SHA256,
        "implementation_manifest_sha256": IMPLEMENTATION_MANIFEST_SHA256,
        "stage2_v2_1_selection_rule_sha256": AUTHORITY_SHA256[
            "research/results/stage2_graphgru_selection_v2_1_a40/stage2_selection_rule.json"
        ],
        "stage2b_v2_1_fairness_manifest_sha256": AUTHORITY_SHA256[
            "research/results/stage2b_vanilla_fairness_v2_1/stage2b_fairness_manifest.json"
        ],
        "final_target_labels_accessed": False,
    }


def validate_job_map(stage: str, value: dict[str, Any]) -> None:
    expected_configs = graph_configurations() if stage == "stage2" else vanilla_configurations()
    expected_count = 144 if stage == "stage2" else 48
    jobs = value["jobs"]
    if value.get("protocol_version") != PROTOCOL_VERSION or value.get("status") != "IMMUTABLE_SEALED":
        raise RuntimeError("Unsealed V2.2 job map")
    if len(jobs) != expected_count or len({job["job_id"] for job in jobs}) != expected_count:
        raise RuntimeError(f"{stage} immutable job count/uniqueness failed")
    observed = {(job["config_id"], job["pseudo_target_city_id"], job["seed"]) for job in jobs}
    expected = {
        (config["config_id"], target, seed)
        for config in expected_configs for target in PSEUDO_TARGETS for seed in SEEDS
    }
    if observed != expected:
        raise RuntimeError(f"{stage} factorial grid is incomplete or contains extras")
    outputs: set[str] = set()
    for job in jobs:
        sources = tuple(job["source_city_ids"])
        target = int(job["pseudo_target_city_id"])
        if sources != tuple(city for city in DEVELOPMENT if city != target):
            raise RuntimeError("Pseudo-target exclusion/source order failed")
        if job["source_updates"] != SOURCE_UPDATES or job["batch_size_city_hours"] != BATCH_SIZE:
            raise RuntimeError("Frozen source exposure changed")
        if job["output_mode"] != OUTPUT_MODE or job["target_adaptation_fits"] != 0:
            raise RuntimeError("Stage-2/2B is source-only LOG1P zero-shot selection")
        if job["seed"] not in SEEDS or job["source_city_schedule_seed"] != deterministic_seed(
            "source-city-schedule", target, job["seed"]
        ) or job["source_anchor_sampling_seed"] != deterministic_seed(
            "source-anchor-sampling", target, job["seed"]
        ):
            raise RuntimeError("Deterministic seed mapping changed")
        if any("final" in str(item).lower() for item in job.values()):
            raise RuntimeError("Final-target token entered an immutable development job")
        for key in ("expected_completion", "attempt_namespace"):
            if job[key] in outputs:
                raise RuntimeError("Writable output collision")
            outputs.add(job[key])


def quantized_primary(value: float) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def aggregate(stage: str, completed: list[dict[str, Any]]) -> dict[str, Any]:
    configurations = graph_configurations() if stage == "stage2" else vanilla_configurations()
    expected = 144 if stage == "stage2" else 48
    if len(completed) != expected:
        raise RuntimeError(f"{stage} requires exactly {expected} completed records")
    fold_rows: list[dict[str, Any]] = []
    for configuration in configurations:
        for target in PSEUDO_TARGETS:
            rows = [row for row in completed if row["config_id"] == configuration["config_id"]
                    and row["pseudo_target_city_id"] == target]
            if sorted(row["seed"] for row in rows) != list(SEEDS):
                raise RuntimeError("Fold does not contain exactly the three registered seeds")
            maes = [float(row["metrics"]["mae"]) for row in rows]
            fold_rows.append({
                **configuration,
                "pseudo_target_city_id": target,
                "seeds": list(SEEDS),
                "mae_seed_mean": statistics.fmean(maes),
                "mae_seed_population_sd": statistics.pstdev(maes),
                "observed_station_hours": int(rows[0]["metrics"]["n"]),
            })
    config_rows: list[dict[str, Any]] = []
    for configuration in configurations:
        folds = [row for row in fold_rows if row["config_id"] == configuration["config_id"]]
        maes = [float(row["mae_seed_mean"]) for row in folds]
        config_rows.append({
            **configuration,
            "equal_fold_count_space_mae": statistics.fmean(maes),
            "primary_rounded_4dp": str(quantized_primary(statistics.fmean(maes))),
            "fold_mae_population_sd": statistics.pstdev(maes),
            "fold_mae_min": min(maes),
            "fold_mae_max": max(maes),
            "fold_mae_range": max(maes) - min(maes),
        })
    def rank(row: dict[str, Any]) -> tuple[Any, ...]:
        base = (
            quantized_primary(float(row["equal_fold_count_space_mae"])),
            float(row["fold_mae_population_sd"]),
            int(row["hidden_size"]),
        )
        if stage == "stage2":
            return base + (int(row["graph_k"]), float(row["dropout"]))
        return base + (float(row["dropout"]),)
    config_rows.sort(key=rank)
    for index, row in enumerate(config_rows, 1):
        row["rank"] = index
    winner, runner_up = config_rows[:2]
    return {
        "stage": 2 if stage == "stage2" else "2B",
        "selection_rule": STAGE2_SELECTION_RULE if stage == "stage2" else STAGE2B_SELECTION_RULE,
        "fold_rows": fold_rows,
        "configuration_rows": config_rows,
        "winner": winner,
        "runner_up": runner_up,
        "margin_to_runner_up": float(runner_up["equal_fold_count_space_mae"])
        - float(winner["equal_fold_count_space_mae"]),
        "tie_break_invoked": quantized_primary(float(winner["equal_fold_count_space_mae"]))
        == quantized_primary(float(runner_up["equal_fold_count_space_mae"])),
        "historical_v2_1_winner": HISTORICAL_STAGE2_WINNER if stage == "stage2" else HISTORICAL_STAGE2B_WINNER,
        "winner_matches_historical_v2_1": winner["config_id"]
        == (HISTORICAL_STAGE2_WINNER if stage == "stage2" else HISTORICAL_STAGE2B_WINNER),
    }


def contract_payload(stage: str, code_sha256: dict[str, str]) -> dict[str, Any]:
    is_graph = stage == "stage2"
    configurations = graph_configurations() if is_graph else vanilla_configurations()
    return {
        "schema_version": "1.0",
        "protocol_version": PROTOCOL_VERSION,
        "artifact_role": "FROZEN_SCIENTIFIC_DEFINITION",
        "stage": 2 if is_graph else "2B",
        "status": "FROZEN_VERIFIED",
        "model_family": "ordinary_graph_gru" if is_graph else "genuine_vanilla_gru",
        "candidate_configurations": list(configurations),
        "pseudo_target_city_ids": list(PSEUDO_TARGETS),
        "development_city_ids": list(DEVELOPMENT),
        "seeds": list(SEEDS),
        "target_transform": {"candidate": "LOG1P", "output_mode": OUTPUT_MODE,
                             "loss": "masked_MAE_in_log1p_target_space"},
        "source_policy": {
            "source_updates": SOURCE_UPDATES,
            "batch_size_city_hours": BATCH_SIZE,
            "optimizer": asdict(optimizer_config()),
            "gradient_clip_global_norm": GRADIENT_CLIP,
            "checkpoint": "final_update",
            "early_stopping": False,
            "source_city_exposure": "deterministically balanced; counts differ by at most one",
            "source_anchor_sampling": "uniform eligible city-hour anchors with replacement",
        },
        "evaluation": {
            "regime": "zero_shot",
            "window": "[HD,HF)",
            "primary_metric": "count_space_MAE",
            "prediction_commitment": "full origin/station grid persisted before retrospective development labels open",
            "target_adaptation_fits": 0,
        },
        "selection_rule": STAGE2_SELECTION_RULE if is_graph else STAGE2B_SELECTION_RULE,
        "expected_source_fits": 144 if is_graph else 48,
        "expected_zero_shot_evaluations": 144 if is_graph else 48,
        "expected_adaptation_fits": 0,
        "stage1_decision_sha256": STAGE1_DECISION_SHA256,
        "development_data_manifest_sha256": DATA_MANIFEST_SHA256,
        "implementation_manifest_sha256": IMPLEMENTATION_MANIFEST_SHA256,
        "fit_snapshot_manifest_sha256": FIT_MANIFEST_SHA256,
        "base_cache_manifest_sha256": BASE_CACHE_MANIFEST_SHA256,
        "graph_static_reconciliation_sha256": GRAPH_RECONCILIATION_SHA256,
        "authoritative_design_source_sha256": AUTHORITY_SHA256,
        "execution_code_sha256": code_sha256,
        "v2_1_fitted_checkpoints_reused": False,
        "historical_v2_1_winner_context_only": HISTORICAL_STAGE2_WINNER if is_graph else HISTORICAL_STAGE2B_WINNER,
        "contradictions": [],
        "stage3_started": False,
        "stage4_started_or_implemented": False,
        "final_target_labels_accessed": False,
    }
