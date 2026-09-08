"""Read-only implementation verification. No test, data, model or evaluation imports.

Hashes saved implementation evidence and the pre-implementation preservation
inventory. Final evaluation label files are excluded before any file read.
Emits JSON only to stdout; no artifacts, caches or scientific results are written.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
BASE = "research/results/causal_history_implementation_v2_2/"
SPEC_BASE = "research/results/causal_history_spec_v2_2/"
PINS = {
    SPEC_BASE + "v2_2_specification.json": "c85c8fdbcab26e7239bfb4528b570b720e7d31ea5933934d9258b63ad90f9277",
    SPEC_BASE + "causal_history_spec_seal.json": "170d2c3c8b3371dd6c6c3d0606733d6d9a6b93e83449808dd7ccb98a904a86fb",
    SPEC_BASE + "fixed_cohort_static_manifest.json": "24f25a4d18dafe888dffc74acc279e59e51ef5c68c03eaa9e8148c632ae4a046",
    "RESEARCH_PROTOCOL_V2_2_AMENDMENT.md": "02a44e3aa8bf1d656d2f9de833b45072d564052e2ab3c1426bda2a6456c558d0",
    SPEC_BASE + "supersession_registry.json": "9b9bbbae7ab478b3bb83f791142344e94aebe455ca6693723f515d7ffb459c06",
    "research/scripts/verify_causal_history_spec_v2_2.py": "f4e2277fd37240f4f090f8664c2d8ad60ba64c490d0e231ea9808fa6b6d041ed",
}
EXPECTED_IMPLEMENTATION = {
    "research/v2_2/" + n for n in
    ("__init__.py", "contract.py", "history.py", "artifacts.py", "snapshots.py", "labels.py", "README.md")
} | {"research/scripts/verify_causal_history_implementation_v2_2.py"}
EXPECTED_TESTS = {
    "research/tests/test_causal_history_v2_2.py",
    "research/tests/test_causal_history_v2_2_integration.py",
    "research/scripts/test_causal_history_implementation_v2_2.py",
}
EXPECTED_REPORTS = {BASE + n for n in ("test_report.json", "preservation_before.json", "preservation_after.json")}


def safe_path(relative):
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise PermissionError("Only repository-relative POSIX paths are allowed")
    parts = PurePosixPath(relative).parts
    if not parts or PurePosixPath(relative).is_absolute() or ".." in parts:
        raise PermissionError("Path escape")
    path = (ROOT / relative).resolve()
    resolved = path.relative_to(ROOT).as_posix().lower()
    segments = resolved.split("/")
    if any(p in {"final_labels", "final_target_evaluation_labels", "evaluation_labels", "sealed_final_labels"}
           for p in segments):
        raise PermissionError("Final evaluation labels cannot be accessed")
    if path.suffix.lower() in {".parquet", ".csv", ".npz", ".npy", ".arrow", ".pkl", ".pt"} and any(
            p in path.name.lower() for p in ("final_label", "evaluation_label", "sealed_final")):
        raise PermissionError("Final evaluation label payload cannot be accessed")
    return path


def digest(relative):
    h = hashlib.sha256()
    with safe_path(relative).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(relative):
    return json.loads(safe_path(relative).read_text(encoding="utf-8"))


def main():
    checks = []
    def check(name, condition):
        checks.append({"check": name, "status": "PASS" if condition else "FAIL"})
        if not condition:
            raise ValueError(name)
    output = {"purpose": "READ_ONLY_IMPLEMENTATION_VERIFICATION", "status": "FAIL",
              "checks": checks, "tests_rerun": False, "raw_status_read": False,
              "scientific_model_calls": 0, "scientific_panel_builds": 0,
              "scientific_fit_snapshots_built": 0,
              "final_target_evaluation_labels_accessed": False, "files_written": []}
    try:
        for rel, expected in PINS.items():
            check("sealed_binding:" + rel, digest(rel) == expected)
        manifest_path = BASE + "implementation_manifest.json"
        manifest = read_json(manifest_path)
        output["implementation_manifest_sha256"] = digest(manifest_path)
        output["verifier_sha256"] = digest("research/scripts/verify_causal_history_implementation_v2_2.py")
        check("manifest_version", manifest["protocol_version"] == "2.2" and manifest["schema_version"] == "1.0")
        check("manifest_authority", manifest["specification_sha256"] == PINS[SPEC_BASE + "v2_2_specification.json"]
              and manifest["specification_seal_sha256"] == PINS[SPEC_BASE + "causal_history_spec_seal.json"]
              and manifest["cohort_static_sha256"] == PINS[SPEC_BASE + "fixed_cohort_static_manifest.json"])
        check("manifest_purpose", manifest["purpose"] == "IMPLEMENTATION_AND_NON_SCIENTIFIC_TEST_ONLY")
        check("manifest_feature_version", manifest["feature_function_version"] == "v2_2.asof_history.1")
        check("active_count_rule", manifest["active_count_feed"] == {
            "policy": "IDEALIZED_COUNT_FEED_BENCHMARK", "Q": "1[e <= a]", "active_reconstruction_delay_hours": 0})
        check("exact_precision", manifest["timestamp_precision"] == "INTEGER_UTC_MICROSECONDS"
              and manifest["station_bracket"] == "0 <= s-p <= 12h AND 0 <= n-e <= 12h; evidence <= cutoff")
        for field, expected_files in (("implementation_files_sha256", EXPECTED_IMPLEMENTATION),
                                      ("test_files_sha256", EXPECTED_TESTS),
                                      ("evidence_files_sha256", EXPECTED_REPORTS)):
            mapping = manifest[field]
            check("complete_file_set:" + field, set(mapping) == expected_files)
            for rel, expected in mapping.items():
                check("file_binding:" + rel, digest(rel) == expected)
        report = read_json(BASE + "test_report.json")
        check("test_report_purpose", report["purpose"] == "NON_SCIENTIFIC_TEST_ONLY" and report["status"] == "PASS")
        for key in ("specification_sha256", "specification_seal_sha256", "cohort_static_sha256"):
            check("test_authority:" + key, report[key] == manifest[key])
        check("test_totals", report["unit_tests"] == 25 and report["integration_tests"] == 3
              and len(report["tests"]) == 28 and len({r["test"] for r in report["tests"]}) == 28
              and all(r["status"] == "PASS" for r in report["tests"]))
        expected_invariants = [{"number": n, "status": "PASS"} for n in range(1, 19)]
        check("all_18_reported", report["all_18_invariants"] == expected_invariants
              and manifest["all_18_invariants"] == expected_invariants)
        for n in range(1, 19):
            check("invariant_test_present:" + str(n), any(f".test_{n:02d}_" in row["test"] for row in report["tests"]))
        for fragment in ("test_01b_", "test_07b_"):
            check("additional_adversarial_case:" + fragment, any(fragment in row["test"] for row in report["tests"]))
        provenance = report["integration_provenance"]
        check("integration_provenance_binding", provenance == manifest["integration_provenance"])
        check("development_only_fixture", provenance["purpose"] == "NON_SCIENTIFIC_TEST_ONLY"
              and provenance["city_ids"] == [129] and provenance["station_count"] == 76
              and provenance["snapshot_fixture_candidate_hours"] == 1
              and provenance["snapshot_fixture_rows"] == 50 and len(provenance["cutoffs_us"]) == 2
              and provenance["count_values"].startswith("SYNTHETIC:")
              and not provenance["scientific_panel_accessed"]
              and not provenance["final_target_evaluation_labels_accessed"] and provenance["model_calls"] == 0)
        check("fixture_source_manifest", digest(provenance["source_manifest"]) == provenance["source_manifest_sha256"])
        check("runtime_versions_recorded", set(report["runtime_versions"]) == {"numpy", "duckdb", "tzdata"})
        check("no_scientific_execution_reported", all(report[k] == 0 for k in
              ("scientific_model_calls", "scientific_panel_builds", "scientific_fit_snapshots_built"))
              and report["files_written_by_test_runner"] == []
              and not report["final_target_evaluation_labels_accessed"]
              and manifest["scientific_execution_performed"] is False
              and manifest["full_panel_build_authorized"] is False)
        check("firewall_report", manifest["final_label_firewall"] == "PASS_NO_FINAL_TARGET_EVALUATION_LABEL_ACCESS")
        for forbidden in ("processed/protocol_v2_1/final_labels/x.parquet", "../escape.json",
                          "processed/protocol_v2_2/final_labels/x.json", "x/final_evaluation_labels.parquet"):
            try:
                safe_path(forbidden)
            except (PermissionError, ValueError):
                check("verifier_path_rejection:" + forbidden, True)
            else:
                check("verifier_path_rejection:" + forbidden, False)
        for rel in sorted(EXPECTED_IMPLEMENTATION):
            if not rel.startswith("research/v2_2/") or not rel.endswith(".py"):
                continue
            tree = ast.parse(safe_path(rel).read_text(encoding="utf-8"))
            imports = [name for node in ast.walk(tree) for name in
                       ([x.name for x in node.names] if isinstance(node, ast.Import)
                        else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])]
            check("no_scientific_imports:" + rel, not any(
                module.split(".")[0] in {"torch", "duckdb", "pandas", "pickle"}
                or module.startswith(("research.data", "research.models", "research.training", "research.evaluation"))
                for module in imports))
            if rel.endswith("/history.py"):
                forbidden_fields = {"y", "Y", "m_target", "target_12h", "target_24h", "coverage_observed"}
                tokens = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
                tokens |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
                tokens |= {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
                check("no_retrospective_predictor_fields", not (tokens & forbidden_fields)
                      and not any(m in {"labels", "snapshots"} for m in imports))
        before = read_json(BASE + "preservation_before.json")
        after = read_json(BASE + "preservation_after.json")
        check("preservation_report_binding", after["before_inventory_sha256"] == digest(BASE + "preservation_before.json")
              and after["status"] == "PASS" and after["mismatches"] == []
              and after["files_compared"] == len(before["files"]) == 1182)
        preserved_bytes = 0
        for rel, recorded in before["files"].items():
            path = safe_path(rel)
            if path.stat().st_size != recorded["bytes"] or digest(rel) != recorded["sha256"]:
                raise ValueError("Historical preservation failed: " + rel)
            preserved_bytes += recorded["bytes"]
        check("all_1182_historical_files_unchanged", True)
        output["historical_files_verified"] = len(before["files"])
        output["historical_bytes_verified"] = preserved_bytes
        output["preservation_exclusions"] = before["excluded"]
        seal = read_json(SPEC_BASE + "causal_history_spec_seal.json")
        for group in ("artifact_sha256", "governing_source_sha256"):
            for rel, expected in seal[group].items():
                # The sealed source inventory also covers ETL files outside the
                # research preservation inventory. Verify those against the seal.
                actual = before["files"][rel]["sha256"] if rel in before["files"] else digest(rel)
                check("sealed_governing_source:" + rel, actual == expected)
        cohort = read_json(SPEC_BASE + "fixed_cohort_static_manifest.json")
        check("fixed_799_cohort", len(cohort["stations"]) == 799
              and len({r["station_id"] for r in cohort["stations"]}) == 799
              and manifest["fixed_station_count"] == 799)
        spec = read_json(SPEC_BASE + "v2_2_specification.json")
        future = [r["path"] for r in spec["artifact_authority"] if r["status"] == "REQUIRED_NOT_CREATED"]
        check("future_scientific_authorities_recorded", manifest["scientific_artifacts_required_absent"] == future)
        for rel in future:
            check("scientific_authority_absent:" + rel, not safe_path(rel).exists())
        for rel in sorted({str(PurePosixPath(p).parent) for p in future} | {"processed/protocol_v2_2"}):
            check("scientific_output_directory_absent:" + rel, not safe_path(rel).exists())
        output["status"] = "PASS"
        output["verdict"] = "A. V2_2_IMPLEMENTATION_VERIFIED_READY_FOR_PANEL_BUILD_AUTHORIZATION"
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        if not checks or checks[-1]["status"] != "FAIL":
            checks.append({"check": "verification_completed_without_error", "status": "FAIL"})
        output["error"] = str(exc)
        output["verdict"] = "B. V2_2_IMPLEMENTATION_BLOCKED_BY_LISTED_FAILURES"
    output["check_count"] = len(checks)
    print(json.dumps(output, indent=2, ensure_ascii=True))
    return 0 if output["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
