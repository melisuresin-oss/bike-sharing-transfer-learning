"""Verify authorities and create deterministic append-only V2.2 contracts/job maps."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.stage2_v2_2 import core as c
from research.v2_2.contract import Contract, DEVELOPMENT, HOUR


REPORT = "deployment/stage2_v2_2_a40/SCIENTIFIC_DEFINITION_VERIFICATION.json"
REPORT_MD = "deployment/stage2_v2_2_a40/SCIENTIFIC_DEFINITION_REPORT.md"


def write_once(relative: str, value: Any) -> dict[str, Any]:
    destination = c.repository_path(relative)
    text = (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
            if not isinstance(value, str) else value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != text:
            raise RuntimeError("Append-only scientific definition differs: " + relative)
    else:
        destination.write_text(text, encoding="utf-8")
    return {"path": relative, "sha256": c.sha256_file(relative), "bytes": destination.stat().st_size}


def code_hashes() -> dict[str, str]:
    paths = (
        "research/stage2_v2_2/core.py",
        "research/stage2_v2_2/a40.py",
        "research/models/common.py",
        "research/models/heads.py",
        "research/models/graph_gru.py",
        "research/models/vanilla_gru.py",
        "research/training/losses.py",
        "research/training/trainer.py",
        "research/training/reproducibility.py",
        "research/evaluation/metrics.py",
        "research/development_data_v2_2/registry.py",
        "research/v2_2/contract.py",
        "research/v2_2/artifacts.py",
        "research/v2_2/snapshots.py",
    )
    return {relative: c.sha256_file(relative) for relative in paths}


def prediction_grid(target: int, cache: dict[str, Any]) -> tuple[str, int]:
    city = cache["cities"][str(target)]
    origins = np.load(c.repository_path(city["arrays"]["origin_us"]["path"]), mmap_mode="r", allow_pickle=False)
    stations = np.load(c.repository_path(city["arrays"]["station_ids"]["path"]), mmap_mode="r", allow_pickle=False)
    contract = Contract()
    first = int((contract.boundaries["HD"] - contract.boundaries["H0"]) // HOUR)
    last = int((contract.boundaries["HF"] - contract.boundaries["H0"]) // HOUR)
    selected = origins[first:last]
    digest = c.canonical_hash({
        "city_id": target,
        "origin_us": list(map(int, selected)),
        "station_ids": list(map(int, stations)),
    })
    return digest, len(selected) * len(stations)


def jobs(stage: str, contract_entry: dict[str, Any]) -> dict[str, Any]:
    configurations = c.graph_configurations() if stage == "stage2" else c.vanilla_configurations()
    fit = c.read_json(c.FIT_MANIFEST)
    cache = c.read_json(c.BASE_CACHE_MANIFEST)
    graphs = c.read_json(c.GRAPH_RECONCILIATION)
    rows: list[dict[str, Any]] = []
    output_root = c.STAGE2_OUT if stage == "stage2" else c.STAGE2B_OUT
    for configuration in configurations:
        for target in c.PSEUDO_TARGETS:
            source_cities = tuple(city for city in DEVELOPMENT if city != target)
            source_snapshot = next(item for item in fit["source_folds"] if item["target_city"] == target)
            source_info = c.read_json(source_snapshot["path"])
            if tuple(source_info["request"]["city_ids"]) != source_cities:
                raise RuntimeError("Source snapshot roster differs from frozen pseudo-target exclusion")
            grid_hash, grid_count = prediction_grid(target, cache)
            graph_bindings = []
            if stage == "stage2":
                graph_bindings = [
                    next(item for item in graphs["graphs"]
                         if item["city_id"] == city and item["k"] == configuration["graph_k"])
                    for city in sorted(set(source_cities + (target,)))
                ]
            for seed in c.SEEDS:
                job_id = f"{configuration['config_id'].lower()}_target-{target}_seed-{seed}"
                rows.append({
                    "schema_version": "1.0",
                    "protocol_version": "2.2",
                    "stage": 2 if stage == "stage2" else "2B",
                    "job_id": job_id,
                    "config_id": configuration["config_id"],
                    "configuration": configuration,
                    "pseudo_target_city_id": target,
                    "source_city_ids": list(source_cities),
                    "seed": seed,
                    "initialization_seed": seed,
                    "source_city_schedule_seed": c.deterministic_seed("source-city-schedule", target, seed),
                    "source_anchor_sampling_seed": c.deterministic_seed("source-anchor-sampling", target, seed),
                    "source_snapshot": source_snapshot,
                    "source_training_key_seal_sha256": source_info["training_key_seal_sha256"],
                    "source_partition_bindings": source_info["source_partitions"],
                    "graph_bindings": graph_bindings,
                    "target_transform": "LOG1P",
                    "output_mode": c.OUTPUT_MODE,
                    "source_updates": c.SOURCE_UPDATES,
                    "batch_size_city_hours": c.BATCH_SIZE,
                    "optimizer": {
                        "name": "AdamW", "learning_rate": 1e-3, "betas": [0.9, 0.999],
                        "epsilon": 1e-8, "weight_decay": 1e-4,
                    },
                    "global_gradient_clip": c.GRADIENT_CLIP,
                    "early_stopping": False,
                    "source_pretraining_fit": True,
                    "zero_shot_evaluation": True,
                    "target_adaptation_fits": 0,
                    "prediction_grid_key_sha256": grid_hash,
                    "prediction_grid_key_count": grid_count,
                    "prediction_commitment_before_development_labels": True,
                    "expected_completion": output_root + "completed/" + job_id + ".json",
                    "attempt_namespace": output_root + "attempts/" + job_id + "/",
                })
    result = {
        "schema_version": "1.0",
        "protocol_version": "2.2",
        "artifact_role": "IMMUTABLE_SCIENTIFIC_JOB_MAP",
        "stage": 2 if stage == "stage2" else "2B",
        "status": "IMMUTABLE_SEALED",
        "scientific_contract": contract_entry,
        "jobs": rows,
        "job_count": len(rows),
        "source_fits": len(rows),
        "zero_shot_evaluations": len(rows),
        "target_adaptation_fits": 0,
        "final_target_labels_accessed": False,
    }
    c.validate_job_map(stage, result)
    return result


authority = c.validate_authorities()
stage2_rule = write_once(c.STAGE2_RULE, c.STAGE2_SELECTION_RULE)
stage2b_rule = write_once(c.STAGE2B_RULE, c.STAGE2B_SELECTION_RULE)
execution_code = code_hashes()
stage2_contract_value = {
    **c.contract_payload("stage2", execution_code),
    "selection_rule_artifact": stage2_rule,
}
stage2b_contract_value = {
    **c.contract_payload("stage2b", execution_code),
    "selection_rule_artifact": stage2b_rule,
    "fairness_matching": {
        "matched_to_graph_gru": [
            "source snapshots", "source city schedules", "source anchor schedules",
            "source update count", "batch size in city-hours", "optimizer", "loss/output mode",
            "gradient clipping", "seeds", "evaluation window and observed keys",
        ],
        "not_claimed_matched": ["parameter count across different hidden widths", "wall-clock runtime"],
    },
}
stage2_contract = write_once(c.STAGE2_CONTRACT, stage2_contract_value)
stage2b_contract = write_once(c.STAGE2B_CONTRACT, stage2b_contract_value)
stage2_map = write_once(c.STAGE2_JOB_MAP, jobs("stage2", stage2_contract))
stage2b_map = write_once(c.STAGE2B_JOB_MAP, jobs("stage2b", stage2b_contract))

verification = {
    "schema_version": "1.0",
    "protocol_version": "2.2",
    "status": "PASS",
    "scope": "SCIENTIFIC_DEFINITION_AND_IMMUTABLE_JOB_MAP_ONLY",
    "authority_verification": authority,
    "stage2_scientific_contract": stage2_contract,
    "stage2b_scientific_contract": stage2b_contract,
    "stage2_job_map": stage2_map,
    "stage2b_job_map": stage2b_map,
    "expected_counts": {"stage2": 144, "stage2b": 48, "total": 192},
    "scientific_training_steps": 0,
    "scientific_evaluations": 0,
    "final_target_labels_accessed": False,
    "stage3_started": False,
    "stage4_started_or_implemented": False,
    "contradictions": [],
}
report = write_once(REPORT, verification)
markdown = f"""# Stage 2 and Stage 2B V2.2 scientific-definition verification

Status: **PASS**. No governing contradiction was found.

- Stage 1 binding: `{c.STAGE1_DECISION_SHA256}`; selected formulation `LOG1P / log1p_target`.
- Stage 2: 12 Graph-GRU configurations x 4 pseudo-targets x 3 seeds = 144 source fits and 144 zero-shot evaluations.
- Stage 2B: 4 genuine Vanilla-GRU configurations x 4 pseudo-targets x 3 seeds = 48 source fits and 48 zero-shot evaluations.
- Both use 12,000 source updates, 16 city-hour anchors per update, AdamW 1e-3, weight decay 1e-4, global clipping 1.0, final-update checkpoints, and no early stopping or adaptation.
- Candidate-independent data-order seeds retain the registered `stage2-v2.1` namespace as historical deterministic design provenance.
- V2.1 fitted checkpoints, predictions, metrics, and winners are excluded from V2.2 selection.
- Final-target labels are inaccessible. Stage 3 and Stage 4 remain unstarted.

Artifacts:

- Stage-2 contract: `{stage2_contract['path']}` (`{stage2_contract['sha256']}`)
- Stage-2B contract: `{stage2b_contract['path']}` (`{stage2b_contract['sha256']}`)
- Stage-2 job map: `{stage2_map['path']}` (`{stage2_map['sha256']}`)
- Stage-2B job map: `{stage2b_map['path']}` (`{stage2b_map['sha256']}`)
- Machine verification: `{report['path']}` (`{report['sha256']}`)
"""
write_once(REPORT_MD, markdown)
print(json.dumps(verification, indent=2))
