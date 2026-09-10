"""PB1 R6 scorer: frozen mathematics with one append-only selection fix."""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.final_v2_2_r7_r1_pb1 import scorer as frozen

from . import context, core

METRICS = frozen.METRICS
EvaluationData = frozen.EvaluationData


def _zero_reference_selection(macro_lookup: dict) -> tuple[list[dict], str | None]:
    """Apply the registered rule to candidates whose macro MAE is defined."""
    statuses = []
    estimable = []
    for method in core.REGISTERED_ZERO_REFERENCES:
        value = macro_lookup[(method, "0")]["point"]["mae"]
        available = value is not None and math.isfinite(float(value))
        statuses.append({
            "method": method,
            "equal_city_macro_mae_estimable": available,
        })
        if available:
            estimable.append((float(value), method))
    selected = min(estimable, key=lambda item: (item[0], item[1]))[1] \
        if estimable else None
    return statuses, selected


def _comparison_specs(selected_reference: str | None) -> list[tuple]:
    specs = []
    for budget in ("1", "7", "30", "full"):
        specs.extend([
            ("graph_contribution", budget, "target_only_graph", "target_only_vanilla"),
            ("ordinary_transfer", budget, "ordinary_source_adapted", "target_only_graph"),
            ("adaptation", budget, "ordinary_source_adapted", "ordinary_source_graphgru"),
            ("pooled_vs_sequential", budget, "pooled_graph", "ordinary_source_adapted"),
            ("grl_incremental", budget, "grl_l50_constant_adapted", "ordinary_source_adapted"),
            ("grl_source_invariance", budget, "grl_l50_constant_source_graphgru", "ordinary_source_graphgru"),
        ])
    specs.extend([
        ("ordinary_zero_transfer", "0", "ordinary_source_graphgru", selected_reference),
        ("grl_zero_transfer", "0", "grl_l50_constant_source_graphgru", selected_reference),
        ("grl_zero_incremental", "0", "grl_l50_constant_source_graphgru", "ordinary_source_graphgru"),
    ])
    return specs


def _undefined_reference_result() -> tuple[dict, np.ndarray]:
    replicates = np.full((2000, len(METRICS)), np.nan, dtype=np.float64)
    result = {
        "point_gain": {metric: None for metric in METRICS},
        "intervals": {
            metric: {"lower": None, "upper": None,
                     "reason": "deterministic_zero_reference_unavailable"}
            for metric in METRICS
        },
        "valid_n_by_seed": [0] * len(core.FINAL_SEEDS),
        "training_seeds": list(core.FINAL_SEEDS),
        "gain_definition": "error(reference)-error(method)",
        "positive_favors_method": True,
        "fixed_intersection": True,
        "availability_status": "UNAVAILABLE_NO_ESTIMABLE_DETERMINISTIC_ZERO_REFERENCE",
    }
    return result, replicates


def _build_comparisons(group_cache: dict, weights: dict,
                       selected_reference: str | None,
                       replicate_arrays: dict) -> tuple[list, list, list]:
    comparison_specs = _comparison_specs(selected_reference)
    comparisons, macro_comparisons = [], []
    for family, budget, method, reference in comparison_specs:
        city_reps = []
        for city in core.TARGETS:
            if reference is None:
                result, reps = _undefined_reference_result()
            else:
                result, reps = frozen._paired(
                    city, group_cache[(city, method, budget)],
                    group_cache[(city, reference, budget)], weights[city])
            replicate_key = f"comparison_{len(comparisons):03d}"
            result.update({
                "comparison_id": f"{family}|{city}|{budget}",
                "family": family,
                "city_id": city,
                "budget": budget,
                "method": method,
                "reference": reference,
                "replicate_key": replicate_key,
            })
            comparisons.append(result)
            replicate_arrays[replicate_key] = reps
            city_reps.append(reps)
        stacked = np.stack(city_reps)
        macro_reps = np.mean(stacked, axis=0)
        macro_reps[np.any(~np.isfinite(stacked), axis=0)] = np.nan
        points = {
            metric: None if any(row["point_gain"][metric] is None
                                for row in comparisons[-4:])
            else float(np.mean([row["point_gain"][metric]
                                for row in comparisons[-4:]]))
            for metric in METRICS
        }
        replicate_key = f"comparison_macro_{len(macro_comparisons):03d}"
        macro = {
            "comparison_id": f"{family}|equal_city|{budget}",
            "family": family,
            "budget": budget,
            "method": method,
            "reference": reference,
            "cities": list(core.TARGETS),
            "weights": [0.25] * 4,
            "point_gain": points,
            "intervals": frozen._intervals(macro_reps),
            "replicate_key": replicate_key,
            "gain_definition": "error(reference)-error(method)",
        }
        if reference is None:
            macro["availability_status"] = \
                "UNAVAILABLE_NO_ESTIMABLE_DETERMINISTIC_ZERO_REFERENCE"
        macro_comparisons.append(macro)
        replicate_arrays[replicate_key] = macro_reps
    return comparison_specs, comparisons, macro_comparisons


def compute_outputs(stage_root: Path, materialized_targets: Path):
    """Copy of frozen compute_outputs with only zero-reference control flow changed."""
    labels = frozen._load_labels(materialized_targets)
    plans, evaluations = frozen._load_evaluations(stage_root, labels)
    pass_rows = []
    groups: dict[tuple[int, str, str], dict[int | None, EvaluationData]] = defaultdict(dict)
    for plan, item in zip(plans, evaluations):
        stats = frozen._sufficient(
            item.label, item.prediction,
            item.label_valid & item.prediction_available,
            item.station, item.hour_index)
        pass_rows.append({
            "evaluation_id": item.evaluation_id,
            "city_id": item.city_id,
            "method": item.method,
            "budget": item.budget,
            "seed": item.seed,
            "metrics": frozen._metric_dict(frozen._point(stats)),
            "valid_n": int(stats["valid"].sum()),
            "eligible_stations": int(len(stats["eligible"])),
        })
        for reporting_budget in plan["reuse_labels"]:
            key = (item.city_id, item.method, reporting_budget)
            if item.seed in groups[key]:
                raise ValueError("duplicate reporting-cell seed")
            groups[key][item.seed] = item
    reporting_count = sum(len(plan["reuse_labels"]) for plan in plans)
    if reporting_count != 660 or len(pass_rows) != 468:
        raise ValueError("468-pass/660-cell contract mismatch")
    weights = {city: frozen._draw_weights(city) for city in core.TARGETS}
    cell_rows, replicate_arrays = [], {}
    group_cache = {}
    ordered = sorted(
        groups,
        key=lambda value: (value[0], value[1], core.BUDGETS.index(value[2])))
    for index, key in enumerate(ordered):
        city, method, budget = key
        result, reps = frozen._group_result(city, groups[key], weights[city])
        replicate_key = f"cell_{index:03d}"
        result.update({
            "cell_id": f"{city}|{method}|{budget}",
            "city_id": city,
            "method": method,
            "budget": budget,
            "replicate_key": replicate_key,
        })
        cell_rows.append(result)
        replicate_arrays[replicate_key] = reps
        group_cache[key] = groups[key]
    if len(cell_rows) != 180:
        raise ValueError("expected 180 seed-aggregated city cells")
    macro_rows = []
    method_budgets = sorted(
        {(row["method"], row["budget"]) for row in cell_rows},
        key=lambda value: (value[0], core.BUDGETS.index(value[1])))
    for method, budget in method_budgets:
        city_rows = [row for row in cell_rows
                     if row["method"] == method and row["budget"] == budget]
        if {row["city_id"] for row in city_rows} != set(core.TARGETS):
            raise ValueError("equal-city macro missing target")
        point = {
            metric: None if any(row["point"][metric] is None for row in city_rows)
            else float(np.mean([row["point"][metric] for row in city_rows]))
            for metric in METRICS
        }
        city_replicates = np.stack([
            replicate_arrays[row["replicate_key"]] for row in city_rows])
        macro_replicates = np.mean(city_replicates, axis=0)
        macro_replicates[np.any(~np.isfinite(city_replicates), axis=0)] = np.nan
        replicate_key = f"macro_cell_{len(macro_rows):03d}"
        macro_rows.append({
            "macro_id": f"equal_city|{method}|{budget}",
            "method": method,
            "budget": budget,
            "cities": list(core.TARGETS),
            "weights": [0.25] * 4,
            "point": point,
            "intervals": frozen._intervals(macro_replicates),
            "replicate_key": replicate_key,
        })
        replicate_arrays[replicate_key] = macro_replicates
    macro_lookup = {(row["method"], row["budget"]): row for row in macro_rows}
    candidate_status, selected_reference = _zero_reference_selection(macro_lookup)
    comparison_specs, comparisons, macro_comparisons = _build_comparisons(
        group_cache, weights, selected_reference, replicate_arrays)
    metrics_payload = {
        "schema_version": core.METRICS_SCHEMA,
        "status": "PB1_FROZEN_COUNT_SPACE_METRICS",
        "evaluation_passes": pass_rows,
        "evaluation_pass_count": 468,
        "registered_reporting_cell_count": 660,
        "aggregated_city_cells": cell_rows,
        "aggregated_city_cell_count": 180,
        "equal_city_macro_cells": macro_rows,
        "equal_city_macro_cell_count": len(macro_rows),
        "registered_parameter_zero_candidates": candidate_status,
        "selected_parameter_zero_reference": selected_reference,
        "unknown_or_unavailable_filled_with_zero": False,
        "seed_aggregation": "metric per seed, arithmetic mean; population SD denominator 5",
    }
    bootstrap_payload = {
        "schema_version": core.BOOTSTRAP_SCHEMA,
        "status": "PB1_FROZEN_2000_REPLICATE_TEMPORAL_BOOTSTRAP",
        "replicates": 2000,
        "block_lengths": list(frozen.frozen.BLOCK_LENGTHS),
        "namespace": frozen.frozen.BOOTSTRAP_NAMESPACE,
        "city_draw_hashes": {
            str(city): hashlib.sha256(
                repr(frozen.frozen.block_draws(city)).encode("ascii")).hexdigest()
            for city in core.TARGETS
        },
        "city_cells": [
            {key: value for key, value in row.items()
             if key not in ("point", "seed_population_sd", "valid_n_by_seed",
                            "eligible_stations_by_seed", "training_seeds")}
            for row in cell_rows
        ],
        "equal_city_macro_cells": [
            {key: value for key, value in row.items() if key != "point"}
            for row in macro_rows
        ],
        "undefined_replicates_redrawn": False,
        "training_seed_randomness_inside_bootstrap": False,
    }
    comparisons_payload = {
        "schema_version": core.COMPARISONS_SCHEMA,
        "status": "PB1_FROZEN_PAIRED_COMPARISONS",
        "registered_parameter_zero_candidates": candidate_status,
        "selected_parameter_zero_reference": selected_reference,
        "city_comparisons": comparisons,
        "equal_city_macro_comparisons": macro_comparisons,
        "gain_definition": "error(reference)-error(method)",
        "positive_favors_method": True,
        "comparison_families": [list(item) for item in comparison_specs],
    }
    return metrics_payload, bootstrap_payload, comparisons_payload, replicate_arrays


def score(authorization_path: Path, stage_root: Path, raw_data_root: Path,
          target_authority_path: Path, r4_package_manifest: Path,
          r4_package_sha256: str, recovery_package_manifest: Path,
          recovery_package_sha256: str, scoring_package_manifest: Path,
          scoring_package_sha256: str) -> dict:
    bound = context.validate_scoring_context(
        authorization_path, stage_root, raw_data_root, target_authority_path,
        r4_package_manifest, r4_package_sha256, recovery_package_manifest,
        recovery_package_sha256, scoring_package_manifest,
        scoring_package_sha256)
    output_root = bound["stage_root"] / core.PB1_OUT
    paths = {
        "metrics": output_root / "metrics.json",
        "bootstrap": output_root / "bootstrap_summary.json",
        "comparisons": output_root / "paired_comparisons.json",
        "replicates": output_root / "bootstrap_replicates.npz",
        "provenance": output_root / "scoring_fix_provenance.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("repeat/conflicting R6 scoring publication")
    metrics, bootstrap, comparisons, replicates = compute_outputs(
        bound["stage_root"], bound["target_path"])
    published = {
        "metrics": core.atomic_json(paths["metrics"], metrics),
        "bootstrap": core.atomic_json(paths["bootstrap"], bootstrap),
        "comparisons": core.atomic_json(paths["comparisons"], comparisons),
        "replicates": core.atomic_bytes(
            paths["replicates"], core.deterministic_npz(replicates)),
    }
    provenance = {
        "schema_version": core.SCORING_PROVENANCE_SCHEMA,
        "status": "PB1_R6_SCORING_FIX_APPLIED",
        "scoring_revision": core.SCORING_REVISION,
        "authorization": core.entry(core.no_symlink_path(authorization_path),
                                    relative_to=bound["stage_root"]),
        "original_r4_package": bound["r4_package"],
        "r5_recovery_package": bound["recovery_package"],
        "r6_scoring_fix_package": bound["scoring_package"],
        "prediction_commitment": bound["prediction_commitment"],
        "prediction_manifest": bound["prediction_manifest"],
        "materialized_targets": core.entry(bound["target_path"],
                                           relative_to=bound["stage_root"]),
        "r5_recovery_provenance": bound["recovery_provenance"],
        "registered_zero_reference_candidates":
            metrics["registered_parameter_zero_candidates"],
        "selected_deterministic_reference":
            metrics["selected_parameter_zero_reference"],
        "scoring_outputs": {
            key: core.entry(paths[key], relative_to=bound["stage_root"])
            for key in ("metrics", "bootstrap", "comparisons", "replicates")
        },
        "selection_rule": "lowest estimable equal-city macro count-space MAE; exact tie lexicographic method ID",
        "scientific_protocol_changed": False,
        "predictions_changed": False,
        "models_retrained": False,
        "labels_changed": False,
        "masks_changed": False,
        "recovery_rerun": False,
        "raw_sources_reopened": False,
        "created_utc": core.utc(),
    }
    published["provenance"] = core.atomic_json(paths["provenance"], provenance)
    return {
        "status": "PASS_PB1_R6_SCORING_FIX",
        "outputs": published,
        "evaluation_passes": 468,
        "registered_reporting_cells": 660,
        "training_started": False,
        "recovery_rerun": False,
        "raw_sources_reopened": False,
        "predictions_written": False,
        "checkpoints_written": False,
    }

