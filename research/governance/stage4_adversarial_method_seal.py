from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SEAL_ROOT = ROOT / "research" / "results" / "stage4_adversarial_method_v2_1"

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
    "MODEL_SPECIFICATION_V2.md",
    "MODEL_SPECIFICATION_V2_1.md",
    "FEATURE_PROTOCOL_V2_1.md",
    "NEURAL_ARCHITECTURE_SPEC.md",
    "NEURAL_REPRODUCIBILITY_PROTOCOL.md",
    "processed/protocol_v2_1/development/development_panel.parquet",
    "processed/protocol_v2_1/manifests/split_manifest.json",
    "research/models/common.py",
    "research/models/graph_gru.py",
    "research/models/heads.py",
    "research/training/trainer.py",
    "research/data/graph_dataset.py",
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
LAMBDA_MAX_VALUES = (0.01, 0.10, 0.50)
SCHEDULES = ("constant", "linear")
SOURCE_UPDATES = 12000
WARMUP_STEPS = 3000

BACKBONE = {
    "config_id": "GGRU_K04_H032_D00",
    "model_type": "Graph-GRU",
    "graph_k": 4,
    "hidden_size": 32,
    "dropout": 0.0,
    "scale_loss_formulation": "LOG1P_TARGET_MAE",
}
METHOD_SPECIFICATION = {
    "scientific_name": "GRL-based multi-source domain-generalized pretraining",
    "alternate_name": "adversarial source-domain-invariant pretraining",
    "canonical_source_target_dann_claim": False,
    "forecast_objective": "LOG1P_TARGET_MAE",
    "domain_objective": "ordinary_unweighted_seven_class_cross_entropy",
    "domain_ce_forward_weight": 1.0,
    "separate_alpha_coefficient": False,
    "grl_forward": "identity",
    "grl_encoder_backward_multiplier": "-lambda_t",
    "discriminator_parameter_gradient": "ordinary_unscaled_cross_entropy_gradient",
    "forecast_head_domain_gradient": "none",
    "pooling": {
        "representation": "final_per_station_graph_gru_hidden_state",
        "formula": "sum_n(M_target[n]*h[n])/sum_n(M_target[n])",
        "mask": "current_hour_M_target_valid_station_coverage_mask_only",
        "demand_value_supplied_to_domain_classifier": False,
        "empty_mask_behavior": "raise_error_no_denominator_clamp",
        "M_hist_used": False,
    },
    "classifier": {
        "layers": ["Linear(32,64)", "ReLU", "torch.nn.Dropout(0.10)", "Linear(64,7)"],
        "dropout_location": "after_hidden_relu",
        "dropout_train_active": True,
        "dropout_eval_disabled": True,
        "recurrent_or_interlayer_gru_dropout": False,
        "pseudo_target_is_class": False,
    },
    "optimizer": {
        "type": "single_joint_AdamW",
        "owns": ["graph_gru_encoder", "forecast_head", "domain_discriminator"],
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "separate_discriminator_optimizer": False,
        "global_gradient_norm_clip": 1.0,
        "clip_scope": "all_joint_optimizer_parameters",
    },
    "source_balancing": {
        "sampler": "EqualCitySampler",
        "updates": SOURCE_UPDATES,
        "maximum_exposure_count_difference": 1,
        "domain_ce_class_weighting": "none",
    },
    "adaptation": {
        "target_history": "7_days",
        "updates": 300,
        "full_network": True,
        "fresh_reset_adamw": True,
        "domain_objective": False,
        "discriminator_present": False,
        "early_stopping": False,
        "governing_policy": "frozen_stage3_fine_tuning_policy",
    },
    "inference": {"discriminator_present": False},
}
SELECTION_RULE = {
    "evaluation": "post_7_day_300_update_adaptation_count_space_mae",
    "seed_aggregation": "arithmetic_mean_within_pseudo_target",
    "primary": "equal_pseudo_target_mean_mae",
    "tie_1": "lower_fold_dispersion",
    "tie_2": "smaller_lambda_max",
    "tie_3": "constant_schedule",
    "diagnostics_used_for_selection_or_stopping": False,
}
REQUIRED_DIAGNOSTICS = (
    "forecast_loss",
    "domain_cross_entropy",
    "domain_accuracy",
    "lambda_t",
    "cumulative_exposure_count_for_each_source_city",
)
FUTURE_SOFTWARE_GATES = (
    "grl_forward_identity",
    "grl_exact_negative_scaled_backward_gradient",
    "lambda_zero_gives_zero_adversarial_encoder_gradient",
    "discriminator_ce_gradient_not_lambda_scaled",
    "forecast_head_receives_no_domain_gradient",
    "exact_M_target_pooling",
    "nonempty_pooling_mask_assertion",
    "explicit_discriminator_dropout_active_in_train",
    "discriminator_deterministic_in_eval",
    "exact_warmup_boundaries",
    "joint_optimizer_each_parameter_exactly_once",
    "global_clipping_covers_all_optimizer_parameters",
    "EqualCitySampler_source_balance",
    "pseudo_target_absent_from_source_domain_classes",
    "discriminator_stripping_leaves_forecasting_model_executable",
    "fresh_reset_adaptation_optimizer",
    "stage3_7_day_300_update_policy",
    "checkpoint_round_trip",
    "deterministic_reproducibility",
    "final_label_firewall",
    "small_nonscientific_synthetic_grl_adversarial_optimization_test",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_stage2_seed(label: str, fold_city_id: int, registered_seed: int) -> int:
    payload = f"stage2-v2.1|{label}|{fold_city_id}|{registered_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def specified_lambda(schedule: str, lambda_max: float, update_index: int) -> float:
    if schedule not in SCHEDULES or lambda_max not in LAMBDA_MAX_VALUES:
        raise ValueError("Unregistered Stage 4 lambda schedule")
    if update_index < 0 or update_index >= SOURCE_UPDATES:
        raise ValueError("Source update index must be in [0,11999]")
    if schedule == "constant":
        return lambda_max
    return lambda_max * min(1.0, update_index / (WARMUP_STEPS - 1))


def build_workload() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lambda_max in LAMBDA_MAX_VALUES:
        for schedule in SCHEDULES:
            lambda_code = int(round(lambda_max * 100))
            config_id = f"ADG_L{lambda_code:02d}_{schedule.upper()}"
            for pseudo_target in PSEUDO_TARGETS:
                sources = tuple(city for city in DEVELOPMENT_CITIES if city != pseudo_target)
                class_map = ";".join(f"{index}:{city}" for index, city in enumerate(sources))
                for seed in SEEDS:
                    source_fit_id = f"{config_id.lower()}_target-{pseudo_target}_seed-{seed}_source"
                    rows.append(
                        {
                            "status": "METHOD_SEALED_IMPLEMENTATION_NOT_AUTHORIZED",
                            "config_id": config_id,
                            "lambda_max": lambda_max,
                            "schedule": schedule,
                            "pseudo_target_city_id": pseudo_target,
                            "pseudo_target": DEVELOPMENT_CITIES[pseudo_target],
                            "source_city_ids": ";".join(str(value) for value in sources),
                            "source_domain_class_map": class_map,
                            "seed": seed,
                            "source_city_schedule_seed": deterministic_stage2_seed(
                                "source-city-schedule", pseudo_target, seed
                            ),
                            "source_anchor_sampling_seed": deterministic_stage2_seed(
                                "source-anchor-sampling", pseudo_target, seed
                            ),
                            "source_fit_id": source_fit_id,
                            "source_updates": SOURCE_UPDATES,
                            "adaptation_fit_id": source_fit_id.replace("_source", "_adapt-7d-300"),
                            "adaptation_history": "7_days",
                            "adaptation_updates": 300,
                            "post_adaptation_evaluation": True,
                            "zero_shot_selection_evaluation": False,
                        }
                    )
    validate_workload(rows)
    return rows


def validate_workload(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 72 or len({row["source_fit_id"] for row in rows}) != 72:
        raise RuntimeError("Stage 4 method workload requires 72 unique source fits")
    if len({row["adaptation_fit_id"] for row in rows}) != 72:
        raise RuntimeError("Stage 4 method workload requires 72 unique adaptations")
    for row in rows:
        sources = {int(value) for value in row["source_city_ids"].split(";")}
        classes = {int(item.split(":")[1]) for item in row["source_domain_class_map"].split(";")}
        if len(sources) != 7 or sources != classes or row["pseudo_target_city_id"] in sources:
            raise RuntimeError("Stage 4 source-domain membership is invalid")
        if row["zero_shot_selection_evaluation"] or not row["post_adaptation_evaluation"]:
            raise RuntimeError("Stage 4 evaluation workload changed")


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_upstream() -> None:
    for relative, expected in UPSTREAM_SHA256.items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen upstream hash mismatch: {relative}")
    stage1 = _read_json(ROOT / next(path for path in UPSTREAM_SHA256 if "stage1_decision" in path))
    stage2 = _read_json(ROOT / next(path for path in UPSTREAM_SHA256 if "stage2_decision" in path))
    stage3 = _read_json(ROOT / next(path for path in UPSTREAM_SHA256 if "stage3_finetuning" in path))
    if stage1.get("SCALE_LOSS_FORMULATION") != "LOG1P_TARGET_MAE":
        raise RuntimeError("Stage 1 decision changed")
    if stage2.get("ORDINARY_GRAPH_GRU_CONFIG") != "GGRU_K04_H032_D00":
        raise RuntimeError("Stage 2 decision changed")
    if stage3.get("policy", {}).get("update_budgets") != [0, 100, 300, 600, 1200]:
        raise RuntimeError("Stage 3 update policy changed")


def render_document() -> str:
    return f"""# Adversarial Domain-Generalization Executable-Method Seal

**Status:** `FROZEN_VERIFIED`  
**Authorization:** executable method specification only; implementation and scientific execution are not authorized.

## Scientific identity and frozen context

The method is **GRL-based multi-source domain-generalized pretraining**, also called **adversarial source-domain-invariant pretraining**. It is not canonical source-target DANN: the unseen pseudo-target is excluded from source pretraining and is never a discriminator class.

The backbone is frozen `GGRU_K04_H032_D00` (`k=4`, hidden 32, backbone dropout 0.0) under `LOG1P_TARGET_MAE`. The development grid is `lambda_max` in `{{0.01,0.10,0.50}}` crossed with `{{constant,linear}}`, giving six configurations over Bilbao, Cardiff, Freiburg, and Goteborg folds with seeds 17, 29, and 43.

## Forecast and adversarial objective

Forecast loss remains `LOG1P_TARGET_MAE`. The discriminator uses ordinary unweighted seven-class cross-entropy with unit forward weight and no separate alpha. GRL is identity in the forward pass. In backward propagation it multiplies only the domain gradient entering the shared Graph-GRU representation by `-lambda_t`. Discriminator parameters receive ordinary unscaled CE gradients; the forecasting head receives no domain-loss gradient.

## Domain representation and classifier

For each city-hour graph, pool final station hidden states as `z=sum_n(M_target[n]*h[n])/sum_n(M_target[n])`. `M_target` is only the current-hour valid-station/coverage mask; no target demand value is supplied, `M_hist` is not used, and an empty mask raises an error rather than clamping the denominator.

The classifier is `Linear(32,64) -> ReLU -> explicit torch.nn.Dropout(0.10) -> Linear(64,7)`. Dropout is invoked after ReLU, active in train mode, and disabled in eval mode. It is not recurrent/inter-layer GRU dropout.

## Lambda schedules

There are 12,000 zero-indexed source optimizer updates `j=0,...,11999`. Constant uses `lambda(j)=lambda_max`. Linear uses `lambda(j)=lambda_max*min(1,j/2999)`, so `lambda(0)=0`, `lambda(2998)<lambda_max`, and `lambda(2999)=lambda(3000)=lambda_max` exactly. Ganin's sigmoid schedule is prohibited. The constant arm is not labeled degenerate and carries no assumed outcome.

## Optimizer, balancing, and diagnostics

One AdamW optimizer jointly owns the Graph-GRU encoder, forecast head, and discriminator: LR `1e-3`, weight decay `1e-4`, betas `(0.9,0.999)`, epsilon `1e-8`. Global gradient-norm clipping 1.0 covers all owned parameters. There is no separate discriminator optimizer or LR.

Source fitting must use `EqualCitySampler`; 12,000 updates across seven sources must differ by at most one exposure. CE is unweighted and has no inverse-frequency weights. Each future fit logs forecast loss, domain CE, domain accuracy, `lambda_t`, and cumulative exposure per source. Chance accuracy `1/7={1/7:.12g}` and uniform CE `ln(7)={math.log(7):.12g}` are descriptive only and cannot select, stop, alter lambda, or invalidate a run. The random-encoder diagnostic is omitted.

## Adaptation, inference, and workload

After source pretraining, discard the discriminator and retain Graph-GRU plus forecast head. Reset AdamW and apply the unchanged Stage 3 policy: full-network target-only adaptation on seven days for exactly 300 updates, no domain loss/discriminator, and no early stopping. The discriminator is absent at inference.

The sealed expected workload is `6 x 4 x 3 = 72` adversarial source fits, 72 dependent seven-day/300-update adaptations, and 72 post-adaptation evaluations: 144 parameter-changing fits. Stage 4 selection has zero zero-shot evaluations. This is an enumeration, not execution authorization.

## Future software-validation gate

Scientific execution is prohibited until every requirement in the machine-readable `future_software_gates` list passes, including a small non-scientific synthetic GRL/adversarial optimization test. In particular, the actual path must prove GRL gradients, exact pooling, discriminator dropout, schedule boundaries, optimizer ownership/clipping, EqualCitySampler balance, pseudo-target exclusion, discriminator stripping, fresh adaptation state, checkpoint round-trip, reproducibility, and the final-label firewall.

## Unresolved final-execution blockers

The final three-versus-five seed contradiction and exact seed-by-temporal-bootstrap aggregation convention remain unresolved. The frozen mapping `0/1/7/30/full -> 0/100/300/600/1200`, graph batch size 16, audited effective passes, no early stopping, zero-shot zero-day contract, Stage 2 reporting, and evaluation hierarchy remain unchanged. These blockers do not prevent this development-only method seal but do prevent later final execution/evaluation authorization.
"""


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_seal_bundle(seal_root: Path = SEAL_ROOT) -> dict[str, Any]:
    verify_upstream()
    if seal_root.exists() and any(seal_root.iterdir()):
        raise FileExistsError(f"Stage 4 method seal is append-only and already exists: {seal_root}")
    seal_root.mkdir(parents=True, exist_ok=True)
    workload = build_workload()
    workload_path = seal_root / "stage4_expected_workload.csv"
    document_path = seal_root / "STAGE4_ADVERSARIAL_METHOD_SEAL.md"
    _write_csv(workload_path, workload)
    document_path.write_text(render_document(), encoding="utf-8")
    artifacts = {
        document_path.relative_to(ROOT).as_posix(): sha256_file(document_path),
        workload_path.relative_to(ROOT).as_posix(): sha256_file(workload_path),
        "research/governance/stage4_adversarial_method_seal.py": sha256_file(
            ROOT / "research" / "governance" / "stage4_adversarial_method_seal.py"
        ),
        "research/scripts/verify_stage4_adversarial_method_seal.py": sha256_file(
            ROOT / "research" / "scripts" / "verify_stage4_adversarial_method_seal.py"
        ),
        "research/tests/test_stage4_adversarial_method_seal.py": sha256_file(
            ROOT / "research" / "tests" / "test_stage4_adversarial_method_seal.py"
        ),
    }
    manifest = {
        "schema_version": "1.0",
        "protocol_version": "2.1",
        "stage": 4,
        "seal_type": "append_only_adversarial_domain_generalization_executable_method",
        "status": "FROZEN_VERIFIED",
        "append_only": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backbone": BACKBONE,
        "candidate_grid": {
            "lambda_max": list(LAMBDA_MAX_VALUES),
            "schedule": list(SCHEDULES),
            "configurations": 6,
            "pseudo_target_city_ids": list(PSEUDO_TARGETS),
            "seeds": list(SEEDS),
        },
        "method": METHOD_SPECIFICATION,
        "lambda_schedule": {
            "source_updates": SOURCE_UPDATES,
            "zero_indexed_updates": [0, SOURCE_UPDATES - 1],
            "warmup_steps": WARMUP_STEPS,
            "constant": "lambda_max",
            "linear": "lambda_max * min(1, j/2999)",
            "ganin_sigmoid": False,
            "constant_known_degenerate_claim": False,
        },
        "selection_rule": SELECTION_RULE,
        "required_diagnostics": list(REQUIRED_DIAGNOSTICS),
        "diagnostic_references": {
            "chance_accuracy": 1 / 7,
            "uniform_prediction_cross_entropy": math.log(7),
            "descriptive_only": True,
            "may_affect_selection_stopping_lambda_or_validity": False,
        },
        "random_encoder_diagnostic": "OMIT",
        "workload": {
            "source_pretraining_fits": 72,
            "adaptation_fits": 72,
            "parameter_changing_fits": 144,
            "post_adaptation_evaluations": 72,
            "zero_shot_selection_evaluations": 0,
        },
        "future_software_gates": list(FUTURE_SOFTWARE_GATES),
        "upstream_sha256": UPSTREAM_SHA256,
        "governing_input_sha256": {
            relative: sha256_file(ROOT / relative) for relative in GOVERNING_INPUTS
        },
        "artifact_integrity": artifacts,
        "firewall": {
            "method_seal_only": True,
            "stage4_implementation_authorized": False,
            "stage4_scientific_execution_authorized": False,
            "scientific_training_triggered": False,
            "prediction_evaluation_triggered": False,
            "final_target_labels_accessed": False,
            "stage1_stage2_stage3_changed": False,
        },
        "recorded_unchanged": {
            "target_history_to_updates": {"0": 0, "1": 100, "7": 300, "30": 600, "full": 1200},
            "batch_size_semantics": "16_city_hour_graphs",
            "effective_adaptation_passes_approx": {"1_day": 66.67, "7_days": 28.57, "30_days": 13.3, "full": 4.8},
            "early_stopping": False,
            "zero_day": "distinct_zero_shot_target_fitting_methods_NA_target_only_transfer_gain_undefined",
            "stage2_winner": "GGRU_K04_H032_D00_ranked_first_in_all_four_folds_small_leading_gaps",
            "evaluation": "count_MAE_primary_secondary_RMSE_WAPE_station_macro_per_city_then_equal_city_2000_paired_7day_blocks_no_iid_cross_city",
        },
        "unresolved_final_execution_blockers": [
            "final_seed_count_3_vs_5",
            "seed_x_temporal_bootstrap_aggregation",
        ],
    }
    manifest_path = seal_root / "stage4_adversarial_method_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ledger_paths = (manifest_path, document_path, workload_path)
    (seal_root / "STAGE4_ADVERSARIAL_METHOD_SEAL.sha256").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}\n"
            for path in ledger_paths
        ),
        encoding="utf-8",
    )
    return manifest


def verify_seal_bundle(seal_root: Path = SEAL_ROOT) -> dict[str, Any]:
    verify_upstream()
    manifest_path = seal_root / "stage4_adversarial_method_manifest.json"
    document_path = seal_root / "STAGE4_ADVERSARIAL_METHOD_SEAL.md"
    workload_path = seal_root / "stage4_expected_workload.csv"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "FROZEN_VERIFIED" or manifest.get("method") != METHOD_SPECIFICATION:
        raise RuntimeError("Stage 4 executable method changed")
    if manifest.get("future_software_gates") != list(FUTURE_SOFTWARE_GATES):
        raise RuntimeError("Stage 4 future software gate changed")
    firewall = manifest.get("firewall", {})
    if firewall.get("method_seal_only") is not True or any(
        firewall.get(key) is not False
        for key in (
            "stage4_implementation_authorized",
            "stage4_scientific_execution_authorized",
            "scientific_training_triggered",
            "prediction_evaluation_triggered",
            "final_target_labels_accessed",
            "stage1_stage2_stage3_changed",
        )
    ):
        raise RuntimeError("Stage 4 method firewall is not closed")
    for relative, expected in manifest["governing_input_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"Governing input changed: {relative}")
    for relative, expected in manifest["artifact_integrity"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"Stage 4 seal artifact changed: {relative}")
    with workload_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 72 or len({row["source_fit_id"] for row in rows}) != 72:
        raise RuntimeError("Sealed Stage 4 workload is incomplete")
    for row in rows:
        if row["status"] != "METHOD_SEALED_IMPLEMENTATION_NOT_AUTHORIZED":
            raise RuntimeError("A Stage 4 workload row was authorized")
        sources = {int(value) for value in row["source_city_ids"].split(";")}
        if int(row["pseudo_target_city_id"]) in sources or len(sources) != 7:
            raise RuntimeError("Stage 4 pseudo-target exclusion failed")
        if row["zero_shot_selection_evaluation"] != "False":
            raise RuntimeError("Stage 4 selection gained an unauthorized zero-shot evaluation")
    ledger = {}
    for line in (seal_root / "STAGE4_ADVERSARIAL_METHOD_SEAL.sha256").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        ledger[relative] = digest
    expected_ledger = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in (manifest_path, document_path, workload_path)
    }
    if ledger != expected_ledger:
        raise RuntimeError("Stage 4 seal hash ledger mismatch")
    return {
        "status": "PASS",
        "seal_type": manifest["seal_type"],
        "configurations_verified": 6,
        "source_fits_enumerated": 72,
        "adaptation_fits_enumerated": 72,
        "stage4_implementation_authorized": False,
        "stage4_scientific_execution_authorized": False,
        "scientific_training_triggered": False,
        "final_target_labels_accessed": False,
    }


def main() -> int:
    print(json.dumps(create_seal_bundle(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
