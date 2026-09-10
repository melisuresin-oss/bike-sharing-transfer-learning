"""Deterministic evaluation-manifest scorer for already committed predictions."""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from research.final_v2_2_r7_r1 import metrics_bootstrap as frozen

from . import authorization, core

METRICS = ("mae", "rmse", "wape", "station_macro_mae")
DETERMINISTIC = {"persistence", "seasonal_naive", "HA_SOURCE", "HA_TARGET"}


@dataclass(frozen=True)
class EvaluationData:
    evaluation_id: str
    city_id: int
    method: str
    budget: str
    seed: int | None
    station: np.ndarray
    hour_index: np.ndarray
    label: np.ndarray
    label_valid: np.ndarray
    prediction: np.ndarray
    prediction_available: np.ndarray


def _json_float(value: float) -> float | None:
    return None if not math.isfinite(float(value)) else float(value)


def _metric_dict(values) -> dict:
    return {name: _json_float(values[index]) for index, name in enumerate(METRICS)}


def _load_labels(path: Path) -> dict[int, dict[str, np.ndarray]]:
    path = core.no_symlink_path(path)
    with np.load(path, allow_pickle=False) as data:
        expected = {"city_id", "station_id", "forecast_origin_us", "opaque_label_join_key", "count", "label_valid"}
        if set(data.files) != expected:
            raise ValueError("materialized-target schema mismatch")
        arrays = {name: np.array(data[name]) for name in expected}
    if arrays["city_id"].dtype != np.int64 or arrays["station_id"].dtype != np.int64 or arrays["forecast_origin_us"].dtype != np.int64:
        raise ValueError("materialized-target key dtype mismatch")
    if arrays["opaque_label_join_key"].dtype != np.uint64 or arrays["count"].dtype != np.int64 or arrays["label_valid"].dtype != np.bool_:
        raise ValueError("materialized-target value dtype mismatch")
    result = {}
    for city_id in core.TARGETS:
        mask = arrays["city_id"] == city_id
        if not mask.any():
            raise ValueError("missing final target city")
        city = {name: value[mask] for name, value in arrays.items()}
        hours = np.unique(city["forecast_origin_us"])
        if len(hours) != core.EVAL_HOURS or int(hours[0]) != core.HT or int(hours[-1]) != core.HE - core.HOUR:
            raise ValueError("final target interval mismatch")
        if np.any(city["count"][city["label_valid"]] < 0) or np.any(city["count"][~city["label_valid"]] != -1):
            raise ValueError("unknown labels were not kept distinct from zero")
        result[city_id] = city
    if set(map(int, np.unique(arrays["city_id"]))) != set(core.TARGETS):
        raise ValueError("unexpected final target city")
    return result


def _load_evaluations(stage_root: Path, labels: dict[int, dict[str, np.ndarray]]) -> tuple[list[dict], list[EvaluationData]]:
    evaluation_manifest = core.read_json(stage_root / core.R7_PATHS["evaluation_manifest"][0])
    prediction_manifest = core.read_json(stage_root / core.PHASE_A_OUT / "predictions/prediction_manifest.json")
    passes = evaluation_manifest.get("passes", [])
    items = prediction_manifest.get("items", [])
    if len(passes) != 468 or len(items) != 468:
        raise ValueError("expected exactly 468 evaluation passes and predictions")
    expected_ids = [item["evaluation_id"] for item in passes]
    actual_ids = [item["evaluation_item_id"] for item in items]
    if len(set(expected_ids)) != 468 or expected_ids != actual_ids:
        raise ValueError("missing, duplicate, extra, or noncanonical prediction")
    result = []
    for plan, item in zip(passes, items):
        path = core.verify_entry(stage_root, item["payload"])
        with np.load(path, allow_pickle=False) as data:
            expected = {"city_id", "station_id", "forecast_origin_us", "prediction", "availability", "opaque_label_join_key"}
            if set(data.files) != expected:
                raise ValueError("malformed prediction payload")
            arrays = {name: np.array(data[name]) for name in expected}
        if arrays["prediction"].dtype != np.float32 or arrays["availability"].dtype != np.bool_ or arrays["opaque_label_join_key"].dtype != np.uint64:
            raise ValueError("malformed prediction dtype")
        city_id = int(plan["target_city_id"]); target = labels[city_id]
        for key in ("city_id", "station_id", "forecast_origin_us", "opaque_label_join_key"):
            if not np.array_equal(arrays[key], target[key]):
                raise ValueError("prediction/materialized-target key mismatch")
        if np.any(~np.isfinite(arrays["prediction"][arrays["availability"]])) or np.any(arrays["prediction"][arrays["availability"]] < 0):
            raise ValueError("malformed prediction values")
        hour_index = ((target["forecast_origin_us"] - core.HT) // core.HOUR).astype(np.int64)
        if np.any(hour_index < 0) or np.any(hour_index >= core.EVAL_HOURS):
            raise ValueError("prediction outside final interval")
        result.append(EvaluationData(plan["evaluation_id"], city_id, plan["method"], plan["budget"], plan["seed"],
                                     target["station_id"], hour_index, target["count"].astype(np.float64),
                                     target["label_valid"], arrays["prediction"].astype(np.float64), arrays["availability"]))
    return passes, result


def _eligible_stations(station: np.ndarray, hour: np.ndarray, valid: np.ndarray) -> np.ndarray:
    pairs = np.rec.fromarrays((station[valid], hour[valid]), names=("station", "hour"))
    if not len(pairs):
        return np.array([], dtype=np.int64)
    unique = np.unique(pairs)
    stations, counts = np.unique(unique.station, return_counts=True)
    return stations[counts >= 24].astype(np.int64)


def _sufficient(label: np.ndarray, prediction: np.ndarray, valid: np.ndarray,
                station: np.ndarray, hour: np.ndarray, eligible: np.ndarray | None = None):
    valid = valid & np.isfinite(label) & np.isfinite(prediction)
    eligible = _eligible_stations(station, hour, valid) if eligible is None else np.asarray(eligible, dtype=np.int64)
    error = np.abs(label - prediction)
    block = np.minimum(hour // 168, 9).astype(np.int64)
    sums = np.zeros((10, 4), dtype=np.float64)
    for index in range(10):
        mask = valid & (block == index)
        sums[index] = (mask.sum(), error[mask].sum(), np.square(label[mask] - prediction[mask]).sum(), np.abs(label[mask]).sum())
    station_abs = np.zeros((10, len(eligible)), dtype=np.float64)
    station_n = np.zeros((10, len(eligible)), dtype=np.float64)
    station_lookup = {int(value): index for index, value in enumerate(eligible)}
    for block_id in range(10):
        mask = valid & (block == block_id)
        for sid in np.unique(station[mask]):
            if int(sid) not in station_lookup:
                continue
            out_index = station_lookup[int(sid)]; selected = mask & (station == sid)
            station_abs[block_id, out_index] = error[selected].sum()
            station_n[block_id, out_index] = selected.sum()
    return {"valid": valid, "eligible": eligible, "block": sums, "station_abs": station_abs, "station_n": station_n}


def _point(stats) -> np.ndarray:
    total = stats["block"].sum(axis=0); n, absolute, squared, denominator = total
    values = np.full(4, np.nan, dtype=np.float64)
    if n:
        values[0] = absolute / n; values[1] = math.sqrt(squared / n)
    if denominator:
        values[2] = absolute / denominator
    if len(stats["eligible"]):
        station_abs = stats["station_abs"].sum(axis=0); station_n = stats["station_n"].sum(axis=0)
        if np.all(station_n > 0):
            values[3] = np.mean(station_abs / station_n)
    return values


def _draw_weights(city_id: int) -> np.ndarray:
    draws = np.asarray(frozen.block_draws(city_id), dtype=np.int8)
    weights = np.zeros((2000, 10), dtype=np.float64)
    for block_id in range(10):
        weights[:, block_id] = (draws == block_id).sum(axis=1)
    return weights


def _replicates(stats, weights: np.ndarray) -> np.ndarray:
    totals = weights @ stats["block"]
    n, absolute, squared, denominator = (totals[:, index] for index in range(4))
    values = np.full((2000, 4), np.nan, dtype=np.float64)
    np.divide(absolute, n, out=values[:, 0], where=n > 0)
    np.sqrt(np.divide(squared, n, out=np.full(2000, np.nan), where=n > 0), out=values[:, 1])
    np.divide(absolute, denominator, out=values[:, 2], where=denominator > 0)
    if len(stats["eligible"]):
        station_abs = weights @ stats["station_abs"]; station_n = weights @ stats["station_n"]
        okay = np.all(station_n > 0, axis=1)
        ratios = np.divide(station_abs, station_n, out=np.zeros_like(station_abs), where=station_n > 0)
        values[okay, 3] = ratios[okay].mean(axis=1)
    return values


def _intervals(replicates: np.ndarray) -> dict:
    result = {}
    for index, metric in enumerate(METRICS):
        values = replicates[:, index]
        if len(values) != 2000 or np.any(~np.isfinite(values)):
            result[metric] = {"lower": None, "upper": None, "reason": "undefined_replicate"}
        else:
            result[metric] = {"lower": float(np.quantile(values, .025, method="linear")),
                              "upper": float(np.quantile(values, .975, method="linear")), "reason": None}
    return result


def _group_result(city_id: int, evaluations: dict[int | None, EvaluationData], weights: np.ndarray):
    expected = set(core.FINAL_SEEDS) if None not in evaluations else {None}
    if set(evaluations) != expected:
        raise ValueError("group must contain exactly five learned seeds or deterministic None")
    points, reps, counts, eligible_counts = [], [], [], []
    for seed in (core.FINAL_SEEDS if expected != {None} else (None,)):
        item = evaluations[seed]
        valid = item.label_valid & item.prediction_available
        stats = _sufficient(item.label, item.prediction, valid, item.station, item.hour_index)
        points.append(_point(stats)); reps.append(_replicates(stats, weights))
        counts.append(int(stats["valid"].sum())); eligible_counts.append(int(len(stats["eligible"])))
    point_array = np.stack(points); replicate_array = np.stack(reps)
    if expected == {None}:
        point = point_array[0]; dispersion = np.full(4, np.nan); combined = replicate_array[0]
    else:
        point = np.mean(point_array, axis=0); dispersion = np.std(point_array, axis=0, ddof=0)
        combined = np.mean(replicate_array, axis=0)
        combined[np.any(~np.isfinite(replicate_array), axis=0)] = np.nan
    return {"point": _metric_dict(point), "seed_population_sd": _metric_dict(dispersion),
            "valid_n_by_seed": counts, "eligible_stations_by_seed": eligible_counts,
            "training_seeds": list(core.FINAL_SEEDS) if expected != {None} else [None],
            "intervals": _intervals(combined)}, combined


def _paired(city_id: int, method: dict[int | None, EvaluationData], reference: dict[int | None, EvaluationData], weights: np.ndarray):
    learned = set(core.FINAL_SEEDS)
    for group in (method, reference):
        if set(group) not in ({None}, learned):
            raise ValueError("invalid paired seed universe")
    seeds = core.FINAL_SEEDS if set(method) == learned or set(reference) == learned else (None,)
    gains, replicates, counts = [], [], []
    for seed in seeds:
        left = method[seed if set(method) == learned else None]
        right = reference[seed if set(reference) == learned else None]
        if not (np.array_equal(left.station, right.station) and np.array_equal(left.hour_index, right.hour_index) and np.array_equal(left.label, right.label)):
            raise ValueError("paired key/label mismatch")
        valid = left.label_valid & right.label_valid & left.prediction_available & right.prediction_available
        eligible = _eligible_stations(left.station, left.hour_index, valid)
        ls = _sufficient(left.label, left.prediction, valid, left.station, left.hour_index, eligible)
        rs = _sufficient(right.label, right.prediction, valid, right.station, right.hour_index, eligible)
        gains.append(_point(rs) - _point(ls)); replicates.append(_replicates(rs, weights) - _replicates(ls, weights))
        counts.append(int(valid.sum()))
    gain_array = np.stack(gains); rep_array = np.stack(replicates)
    point = np.mean(gain_array, axis=0); combined = np.mean(rep_array, axis=0)
    combined[np.any(~np.isfinite(rep_array), axis=0)] = np.nan
    return {"point_gain": _metric_dict(point), "intervals": _intervals(combined),
            "valid_n_by_seed": counts, "training_seeds": list(seeds),
            "gain_definition": "error(reference)-error(method)", "positive_favors_method": True,
            "fixed_intersection": True}, combined


def compute_outputs(stage_root: Path, materialized_targets: Path):
    labels = _load_labels(materialized_targets)
    plans, evaluations = _load_evaluations(stage_root, labels)
    pass_rows = []
    groups: dict[tuple[int, str, str], dict[int | None, EvaluationData]] = defaultdict(dict)
    by_id = {item.evaluation_id: item for item in evaluations}
    for plan, item in zip(plans, evaluations):
        stats = _sufficient(item.label, item.prediction, item.label_valid & item.prediction_available,
                            item.station, item.hour_index)
        pass_rows.append({"evaluation_id": item.evaluation_id, "city_id": item.city_id, "method": item.method,
                          "budget": item.budget, "seed": item.seed, "metrics": _metric_dict(_point(stats)),
                          "valid_n": int(stats["valid"].sum()), "eligible_stations": int(len(stats["eligible"]))})
        for reporting_budget in plan["reuse_labels"]:
            key = (item.city_id, item.method, reporting_budget)
            if item.seed in groups[key]:
                raise ValueError("duplicate reporting-cell seed")
            groups[key][item.seed] = item
    reporting_count = sum(len(plan["reuse_labels"]) for plan in plans)
    if reporting_count != 660 or len(pass_rows) != 468:
        raise ValueError("468-pass/660-cell contract mismatch")
    weights = {city: _draw_weights(city) for city in core.TARGETS}
    cell_rows, replicate_arrays = [], {}
    group_cache = {}
    for index, key in enumerate(sorted(groups, key=lambda value: (value[0], value[1], core.BUDGETS.index(value[2])))):
        city, method, budget = key; result, reps = _group_result(city, groups[key], weights[city])
        replicate_key = f"cell_{index:03d}"
        result.update({"cell_id": f"{city}|{method}|{budget}", "city_id": city, "method": method,
                       "budget": budget, "replicate_key": replicate_key})
        cell_rows.append(result); replicate_arrays[replicate_key] = reps
        group_cache[key] = groups[key]
    if len(cell_rows) != 180:
        raise ValueError("expected 180 seed-aggregated city cells")
    macro_rows = []
    for method_budget in sorted({(row["method"], row["budget"]) for row in cell_rows}, key=lambda x: (x[0], core.BUDGETS.index(x[1]))):
        method, budget = method_budget
        city_rows = [row for row in cell_rows if row["method"] == method and row["budget"] == budget]
        if {row["city_id"] for row in city_rows} != set(core.TARGETS):
            raise ValueError("equal-city macro missing target")
        point = {metric: (None if any(row["point"][metric] is None for row in city_rows) else float(np.mean([row["point"][metric] for row in city_rows]))) for metric in METRICS}
        city_replicates = np.stack([replicate_arrays[row["replicate_key"]] for row in city_rows])
        macro_replicates = np.mean(city_replicates, axis=0)
        macro_replicates[np.any(~np.isfinite(city_replicates), axis=0)] = np.nan
        replicate_key = f"macro_cell_{len(macro_rows):03d}"
        macro_rows.append({"macro_id": f"equal_city|{method}|{budget}", "method": method, "budget": budget,
                           "cities": list(core.TARGETS), "weights": [0.25] * 4, "point": point,
                           "intervals": _intervals(macro_replicates), "replicate_key": replicate_key})
        replicate_arrays[replicate_key] = macro_replicates
    macro_lookup = {(row["method"], row["budget"]): row for row in macro_rows}
    zero_candidates = ("HA_SOURCE", "persistence", "seasonal_naive")
    zero_values = [(macro_lookup[(method, "0")]["point"]["mae"], method) for method in zero_candidates]
    if any(value is None for value, _ in zero_values):
        raise ValueError("parameter-zero deterministic reference is not estimable")
    selected_reference = min(zero_values, key=lambda item: (item[0], item[1]))[1]

    comparison_specs = []
    for budget in ("1", "7", "30", "full"):
        comparison_specs.extend([
            ("graph_contribution", budget, "target_only_graph", "target_only_vanilla"),
            ("ordinary_transfer", budget, "ordinary_source_adapted", "target_only_graph"),
            ("adaptation", budget, "ordinary_source_adapted", "ordinary_source_graphgru"),
            ("pooled_vs_sequential", budget, "pooled_graph", "ordinary_source_adapted"),
            ("grl_incremental", budget, "grl_l50_constant_adapted", "ordinary_source_adapted"),
            ("grl_source_invariance", budget, "grl_l50_constant_source_graphgru", "ordinary_source_graphgru"),
        ])
    comparison_specs.extend([
        ("ordinary_zero_transfer", "0", "ordinary_source_graphgru", selected_reference),
        ("grl_zero_transfer", "0", "grl_l50_constant_source_graphgru", selected_reference),
        ("grl_zero_incremental", "0", "grl_l50_constant_source_graphgru", "ordinary_source_graphgru"),
    ])
    comparisons, macro_comparisons = [], []
    for family, budget, method, reference in comparison_specs:
        city_reps = []
        for city in core.TARGETS:
            result, reps = _paired(city, group_cache[(city, method, budget)], group_cache[(city, reference, budget)], weights[city])
            replicate_key = f"comparison_{len(comparisons):03d}"
            result.update({"comparison_id": f"{family}|{city}|{budget}", "family": family, "city_id": city,
                           "budget": budget, "method": method, "reference": reference,
                           "replicate_key": replicate_key})
            comparisons.append(result); replicate_arrays[replicate_key] = reps; city_reps.append(reps)
        macro_reps = np.mean(np.stack(city_reps), axis=0)
        macro_reps[np.any(~np.isfinite(np.stack(city_reps)), axis=0)] = np.nan
        points = {metric: None if any(row["point_gain"][metric] is None for row in comparisons[-4:]) else float(np.mean([row["point_gain"][metric] for row in comparisons[-4:]])) for metric in METRICS}
        replicate_key = f"comparison_macro_{len(macro_comparisons):03d}"
        macro_comparisons.append({"comparison_id": f"{family}|equal_city|{budget}", "family": family,
                                  "budget": budget, "method": method, "reference": reference,
                                  "cities": list(core.TARGETS), "weights": [0.25] * 4,
                                  "point_gain": points, "intervals": _intervals(macro_reps),
                                  "replicate_key": replicate_key, "gain_definition": "error(reference)-error(method)"})
        replicate_arrays[replicate_key] = macro_reps
    metrics_payload = {"schema_version": core.METRICS_SCHEMA, "status": "PB1_FROZEN_COUNT_SPACE_METRICS",
                       "evaluation_passes": pass_rows, "evaluation_pass_count": 468,
                       "registered_reporting_cell_count": 660, "aggregated_city_cells": cell_rows,
                       "aggregated_city_cell_count": 180, "equal_city_macro_cells": macro_rows,
                       "equal_city_macro_cell_count": len(macro_rows),
                       "selected_parameter_zero_reference": selected_reference,
                       "unknown_or_unavailable_filled_with_zero": False,
                       "seed_aggregation": "metric per seed, arithmetic mean; population SD denominator 5"}
    bootstrap_payload = {"schema_version": core.BOOTSTRAP_SCHEMA, "status": "PB1_FROZEN_2000_REPLICATE_TEMPORAL_BOOTSTRAP",
                         "replicates": 2000, "block_lengths": list(frozen.BLOCK_LENGTHS),
                         "namespace": frozen.BOOTSTRAP_NAMESPACE,
                         "city_draw_hashes": {str(city): hashlib.sha256(repr(frozen.block_draws(city)).encode("ascii")).hexdigest() for city in core.TARGETS},
                         "city_cells": [{key: value for key, value in row.items() if key not in ("point", "seed_population_sd", "valid_n_by_seed", "eligible_stations_by_seed", "training_seeds")} for row in cell_rows],
                         "equal_city_macro_cells": [{key: value for key, value in row.items() if key != "point"} for row in macro_rows],
                         "undefined_replicates_redrawn": False, "training_seed_randomness_inside_bootstrap": False}
    comparisons_payload = {"schema_version": core.COMPARISONS_SCHEMA, "status": "PB1_FROZEN_PAIRED_COMPARISONS",
                           "selected_parameter_zero_reference": selected_reference,
                           "city_comparisons": comparisons, "equal_city_macro_comparisons": macro_comparisons,
                           "gain_definition": "error(reference)-error(method)", "positive_favors_method": True,
                           "comparison_families": [list(item) for item in comparison_specs]}
    return metrics_payload, bootstrap_payload, comparisons_payload, replicate_arrays


def score(authorization_path: Path, stage_root: Path, raw_data_root: Path,
          target_authority_path: Path, pb1_package_manifest: Path, pb1_package_sha: str) -> dict:
    capability = authorization.validate(authorization_path, stage_root, raw_data_root,
                                        target_authority_path, pb1_package_manifest, pb1_package_sha)
    output_root = Path(capability.stage_root_canonical_path) / core.PB1_OUT
    provenance = core.read_json(output_root / "target_materialization_provenance.json")
    if provenance.get("schema_version") != core.MATERIALIZATION_SCHEMA or provenance.get("authorization_sha256") != capability.authorization_sha256:
        raise PermissionError("materialization provenance mismatch")
    target_path = core.no_symlink_path(Path(provenance["materialized_targets"]["path"]))
    if core.sha256_file(target_path) != provenance["materialized_targets"]["sha256"]:
        raise PermissionError("materialized targets changed")
    paths = {"metrics": output_root / "metrics.json", "bootstrap": output_root / "bootstrap_summary.json",
             "comparisons": output_root / "paired_comparisons.json", "replicates": output_root / "bootstrap_replicates.npz"}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("repeat/conflicting score publication")
    metrics, bootstrap, comparisons, replicates = compute_outputs(Path(capability.stage_root_canonical_path), target_path)
    published = {
        "metrics": core.atomic_json(paths["metrics"], metrics),
        "bootstrap": core.atomic_json(paths["bootstrap"], bootstrap),
        "comparisons": core.atomic_json(paths["comparisons"], comparisons),
        "replicates": core.atomic_bytes(paths["replicates"], core.deterministic_npz(replicates)),
    }
    return {"status": "PASS_PB1_SCORING", "outputs": published, "evaluation_passes": 468,
            "registered_reporting_cells": 660, "training_started": False,
            "predictions_written": False, "checkpoints_written": False}
