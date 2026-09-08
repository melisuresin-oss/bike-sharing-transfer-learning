"""Read-only V2.2 governance verifier. No application imports or data construction.

Run with python -B. JSON is emitted to stdout; this program never writes files.
A PASS certifies specification consistency/integrity, not implementation tests.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = "research/results/causal_history_spec_v2_2/"
SEAL = BASE + "causal_history_spec_seal.json"
SPEC = BASE + "v2_2_specification.json"
COHORT = BASE + "fixed_cohort_static_manifest.json"
REGISTRY = BASE + "supersession_registry.json"
AMENDMENT = "RESEARCH_PROTOCOL_V2_2_AMENDMENT.md"
SELF = "research/scripts/verify_causal_history_spec_v2_2.py"
COUNTS = {129: 76, 194: 47, 195: 78, 199: 45, 237: 93, 438: 43,
          467: 28, 476: 73, 532: 43, 617: 60, 619: 87, 658: 126}
BOUNDARIES = {"H0": "2022-08-29T05:00:00Z", "HD": "2023-02-15T22:00:00Z",
              "HF": "2023-04-16T22:00:00Z", "HT": "2023-05-11T15:00:00Z",
              "HE": "2023-07-15T20:00:00Z"}
INVARIANTS = ["future_append", "status_prefix_equivalence", "no_future_status_certificate",
              "exact_12h_boundary", "train_inference_parity", "fit_cutoff_invariance",
              "target_predictor_separation", "final_label_firewall", "zero_denominator",
              "exact_50_percent", "three_value_states", "duplicate_missing_determinism",
              "dst_elapsed_utc", "fixed_cohort", "static_hash_binding", "idealized_count_feed",
              "no_retrospective_y_reads", "sealed_training_keys"]
BLOCKERS = ["final_seed_count_3_vs_5", "seed_x_temporal_bootstrap_aggregation",
            "prediction_commitment_and_final_execution_machinery"]
NODES = ["implementation", "invariants", "panel_and_fit_snapshots", "availability_baselines",
         "stage1", "stage2", "stage2b", "joint_closure", "stage3_rebind", "stage4_rebind",
         "stage4_implementation_gates", "stage4_development"]
EDGES = [["implementation", "invariants"], ["invariants", "panel_and_fit_snapshots"],
         ["panel_and_fit_snapshots", "availability_baselines"], ["availability_baselines", "stage1"],
         ["stage1", "stage2"], ["stage1", "stage2b"], ["stage2", "joint_closure"],
         ["stage2b", "joint_closure"], ["joint_closure", "stage3_rebind"],
         ["stage3_rebind", "stage4_rebind"], ["stage4_rebind", "stage4_implementation_gates"],
         ["stage4_implementation_gates", "stage4_development"]]


def safe_path(relative: str) -> Path:
    """Reject escapes and any label/scientific-data path before opening it."""
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("Expected a repository-relative POSIX path")
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT)
    lowered = relative.lower()
    if any(token in lowered for token in ("final_labels", "sealed_final", "final_evaluation_labels")):
        raise ValueError("Final-label access is prohibited")
    if path.suffix == ".parquet":
        permitted = (relative.startswith("processed/protocol_v2_1/station_manifests/")
                     or relative.startswith("feasibility_test/full_audit/data/cities/")
                     or relative.startswith("feasibility_test/full_audit/data/stations/"))
        if not permitted:
            raise ValueError("Only static metadata Parquet hashes are permitted")
    elif path.suffix.lower() not in (".json", ".md", ".py", ".sql", ".csv", ".sha256"):
        raise ValueError("Non-governance file type is prohibited")
    return path


def digest(relative: str) -> str:
    h = hashlib.sha256()
    with safe_path(relative).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(relative: str):
    return json.loads(safe_path(relative).read_text(encoding="utf-8"))


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def logical_hash(rows) -> str:
    # The already-registered station identity/coordinate serialization.
    h = hashlib.sha256()
    for row in rows:
        fields = ["<NULL>" if v is None else format(v, ".17g") if isinstance(v, float)
                  else str(v) for v in row]
        h.update(("\x1f".join(fields) + "\n").encode("utf-8"))
    return h.hexdigest()


def verify() -> dict:
    checks = []

    def require(condition, name):
        if not condition:
            raise ValueError(name)
        checks.append(name)

    seal = read_json(SEAL)
    require(seal["protocol_version"] == "2.2" and seal["status"] == "SPECIFICATION_SEALED",
            "seal_identity")
    require(set(seal["artifact_sha256"]) == {SPEC, COHORT, REGISTRY, AMENDMENT, SELF},
            "exact_sealed_artifact_set")
    for path, expected in seal["artifact_sha256"].items():
        require(digest(path) == expected, "sealed_hash:" + path)
    for path, expected in seal["governing_source_sha256"].items():
        require(digest(path) == expected, "unchanged_source_hash:" + path)
    spec, cohort, registry = read_json(SPEC), read_json(COHORT), read_json(REGISTRY)
    require(all(v["protocol_version"] == "2.2" for v in (spec, cohort, registry)), "versions")
    require(spec["authorization_scope"] == "GOVERNANCE_SPECIFICATION_WRITES_ONLY", "scope")
    require(all(v is False for v in spec["execution_authorizations"].values()), "no_execution_authorized")
    require(spec["implementation_blockers"] == [], "no_implementation_contradiction")
    require(spec["count_feed"] == {
        "policy": "IDEALIZED_COUNT_FEED_BENCHMARK", "Q": "1[e <= a]",
        "active_delay_microseconds": 0, "measured_ingestion": False,
        "raw_prefix_replay_validated": False, "export_duration_strict_upper_bound_hours": 8,
        "eight_hour_bound_is_active_rule": False}, "count_feed_contract")
    require(spec["coverage"] == {
        "evidence": "status_event_timestamp <= a", "previous": "max(u <= s)",
        "next": "min(u >= e AND u <= a)", "bracket_microseconds": 43200000000,
        "elapsed_bounds_inclusive": True, "missing_event_observed": False,
        "lifetime": "prefix_nonempty AND min(prefix) <= s <= max(prefix)",
        "denominator": "sum(L) over frozen city roster",
        "numerator": "sum(L*B) over frozen city roster",
        "city_gate": "D > 0 AND 2*N >= D", "O": "L*B*G",
        "full_history_extrema_allowed": False}, "coverage_contract")
    require(spec["features"]["lags"] == list(range(1, 25))
            and spec["features"]["weekly_lag"] == 168
            and spec["features"]["mask"] == "O_i(h;t)*Q_i(h;t)"
            and spec["features"]["states"] == {"positive": ["log1p(d)", 1], "zero": [0, 1], "unknown": [0, 0]}
            and spec["features"]["shared_function_required"] is True
            and spec["features"]["retrospective_Y_reads_allowed"] is False
            and spec["features"]["positive_count_exception"] is False, "feature_contract")
    require(spec["fit_policy"]["certification"] == "O_i(h;C)*Q_i(h;C) == 1"
            and spec["fit_policy"]["historical_predictor_cutoff"] == "t, not C"
            and all(spec["fit_policy"][k] is True for k in (
                "neural_labels_certified_by_C", "same_HA_and_adaptation_membership",
                "sealed_training_keys_required", "no_window_extension", "no_retrospective_backfill")),
            "fit_cutoff_contract")
    require(spec["time_rules"] == {"timezone": "UTC", "precision": "microsecond",
            "hour_interval": "[s,e)", "exact_12h_accepted": True,
            "12h_plus_1_microsecond_rejected": True, "cutoff_equality_accepted": True,
            "post_cutoff_evidence_prohibited": True, "minute_rounded_differences_allowed": False},
            "exact_time_contract")
    design = spec["carried_forward_design"]
    require(design["boundaries"] == BOUNDARIES and design["development_seeds"] == [17, 29, 43],
            "boundaries_and_seeds")
    require(design["stage2_grid"] == {"k": [4, 8, 16], "hidden_size": [32, 64], "dropout": [0, 0.1]}
            and design["stage2b_grid"] == {"hidden_size": [32, 64], "dropout": [0, 0.1]}, "grids")
    require(design["stage1"]["source_fits"] == 24 and design["stage1"]["adaptation_fits"] == 24
            and len(design["stage1"]["formulations"]) == 2, "stage1_bounded_design")
    legacy3 = read_json("research/results/stage3_finetuning_policy_v2_1/stage3_finetuning_policy_manifest.json")
    require(design["stage3_policy"] == legacy3["policy"], "unchanged_stage3_numerical_policy")
    split = read_json("processed/protocol_v2_1/manifests/split_manifest.json")
    require(design["city_roles"] == split["city_roles"] and design["budgets"] == split["budgets"],
            "unchanged_roles_and_elapsed_windows")
    require(spec["active_v2_2_winners"] == {"stage1": None, "stage2": None, "stage2b": None},
            "no_inherited_winners")
    require([x["id"] for x in spec["mandatory_invariants"]] == INVARIANTS
            and all(x["status"] == "REQUIRED_NOT_EXECUTED" and x["requirement"]
                    for x in spec["mandatory_invariants"]), "eighteen_unexecuted_implementation_gates")
    require(spec["rerun_dag"] == {"nodes": NODES, "edges": EDGES,
            "parallel_after_stage1": ["stage2", "stage2b"]}, "exact_rerun_DAG")
    require(spec["final_execution_blockers"] == BLOCKERS, "final_blockers_preserved")
    require(spec["descriptive_gate"]["selection_authority"] is False
            and spec["descriptive_gate"]["persistence_fallback_allowed"] is False
            and len(spec["descriptive_gate"]["required_reports"]) == 7, "descriptive_only_gate")
    require(spec["cross_loading"]["reject_v2_1_scientific_artifacts"] is True
            and spec["cross_loading"]["implicit_fallback_allowed"] is False
            and spec["cross_loading"]["sealed_upstream_hashes_required"] is True, "cross_version_rejection")
    roles = [x["role"] for x in spec["artifact_authority"]]
    require(roles == ["amendment", "cohort_static", "panel_data", "causal_history_audit", "fit_snapshots",
                      "baselines", "stage1", "stage2", "stage2b", "joint_closure", "stage3", "stage4"],
            "artifact_authority_complete")
    require(all(x["status"] == "REQUIRED_NOT_CREATED" for x in spec["artifact_authority"][2:]),
            "no_fabricated_downstream_artifacts")
    require(cohort["policy"] == "FIXED_RETROSPECTIVELY_SELECTED_RESEARCH_COHORT"
            and cohort["eligibility_recomputation_changes_membership"] is False, "cohort_policy")
    rows = cohort["stations"]
    require(len(rows) == 799 and len({r["station_id"] for r in rows}) == 799, "799_unique_station_identities")
    require(canonical_hash(rows) == cohort["static_rows_sha256"], "full_static_metadata_binding")
    original = read_json("processed/protocol_v2_1/station_manifests/station_manifest.json")
    originals = {x["city_id"]: x for x in original["cities"]}
    for city, expected_n in COUNTS.items():
        group = [r for r in rows if r["city_id"] == city]
        require(len(group) == expected_n and [r["node_index"] for r in group] == list(range(expected_n)),
                "fixed_city_order:" + str(city))
        require(logical_hash([(r["station_id"],) for r in group]) == originals[city]["roster_hash_sha256"],
                "original_identity_hash:" + str(city))
        require(logical_hash([(r["station_id"], r["latitude"], r["longitude"]) for r in group])
                == originals[city]["coordinate_hash_sha256"], "original_coordinate_hash:" + str(city))
        require(all(math.isfinite(r["latitude"]) and math.isfinite(r["longitude"])
                    and isinstance(r["timezone"], str) and r["timezone"] for r in group),
                "static_fields:" + str(city))
    require(set(registry["categories"]) == {"predictor_panel_and_caches", "stage1", "stage2", "stage2b",
            "baselines_and_HA", "stage3_executable_binding", "stage4_executable_binding"}, "supersession_scope")
    require(all(x["status"] == "EXECUTION_PROVENANCE_RETAINED_SUPERSEDED_FOR_CAUSAL_INFERENCE"
                for x in registry["categories"].values()) and registry["rewrite_originals"] is False,
            "neutral_append_only_supersession")
    require(registry["historical_hash_record"]["path"] in seal["governing_source_sha256"],
            "historical_data_hashes_bound_without_label_access")
    return {"status": "PASS", "verification_scope": "SPECIFICATION_AND_INTEGRITY_ONLY",
            "protocol_version": "2.2", "checks_passed": len(checks), "checks": checks,
            "seal_sha256": digest(SEAL), "artifact_sha256": seal["artifact_sha256"],
            "implementation_tests_executed": False, "scientific_computation_occurred": False,
            "final_target_evaluation_labels_accessed": False, "files_written_by_verifier": [],
            "implementation_blockers": [], "final_execution_blockers": BLOCKERS}


if __name__ == "__main__":
    try:
        result = verify()
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False))
        sys.exit(1)
    print(json.dumps(result, indent=2, ensure_ascii=False))
