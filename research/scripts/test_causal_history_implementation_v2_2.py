"""Run only V2.2 non-scientific tests; emit report to stdout, write no files."""
from __future__ import annotations

import io
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
LOCAL_DEPS = ROOT / "tmp/neural_build/pydeps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.dont_write_bytecode = True


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append({"test": test.id(), "status": "PASS"})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.records.append({"test": test.id(), "status": "FAIL", "detail": self._exc_info_to_string(err, test)})

    def addError(self, test, err):
        super().addError(test, err)
        self.records.append({"test": test.id(), "status": "ERROR", "detail": self._exc_info_to_string(err, test)})

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.records.append({"test": test.id(), "status": "SKIP", "reason": reason})

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err:
            self.records.append({"test": test.id(), "status": "FAIL", "detail": str(subtest)})


def main():
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordingResult)
    unit = runner.run(unittest.defaultTestLoader.loadTestsFromName("research.tests.test_causal_history_v2_2"))
    records = list(unit.records)
    integration_count = 0
    provenance = {}
    passed = unit.wasSuccessful() and not unit.skipped
    if passed:
        integration = runner.run(unittest.defaultTestLoader.loadTestsFromName(
            "research.tests.test_causal_history_v2_2_integration"))
        records.extend(integration.records)
        integration_count = integration.testsRun
        passed = integration.wasSuccessful() and not integration.skipped
        from research.tests.test_causal_history_v2_2_integration import DevelopmentStatusFixtureTests
        provenance = DevelopmentStatusFixtureTests.provenance
    from research.v2_2.contract import SPEC_SHA256, SEAL_SHA256, COHORT_SHA256
    report = {"purpose": "NON_SCIENTIFIC_TEST_ONLY", "status": "PASS" if passed else "FAIL",
              "specification_sha256": SPEC_SHA256, "specification_seal_sha256": SEAL_SHA256,
              "cohort_static_sha256": COHORT_SHA256, "unit_tests": unit.testsRun,
              "integration_tests": integration_count, "tests": records,
              "all_18_invariants": [{"number": n, "status": "PASS" if any(
                  f".test_{n:02d}_" in x["test"] and x["status"] == "PASS" for x in records)
                  and not any(f".test_{n:02d}" in x["test"] and x["status"] != "PASS" for x in records)
                  else "NOT_PASSED"} for n in range(1,19)],
              "integration_provenance": provenance, "python_version": platform.python_version(),
              "runtime_versions": {name: version(name) for name in ("numpy", "duckdb", "tzdata")},
              "log": stream.getvalue(), "scientific_model_calls": 0,
              "scientific_panel_builds": 0, "scientific_fit_snapshots_built": 0,
              "final_target_evaluation_labels_accessed": False, "files_written_by_test_runner": []}
    if any(r['status'] != 'PASS' for r in report['all_18_invariants']):
        report['status'] = 'FAIL'
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == "__main__":
    raise SystemExit(main())
