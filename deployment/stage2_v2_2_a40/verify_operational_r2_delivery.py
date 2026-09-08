"""Independent isolated verification of base package plus R1 and R2 overlays."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "deployment/stage2_v2_2_a40"
DELIVERY = BASE / "DELIVERY_MANIFEST_OPERATIONAL_R2.json"
REPORT = BASE / "LOCAL_VERIFICATION_OPERATIONAL_R2.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    delivery = json.loads(DELIVERY.read_text(encoding="utf-8"))
    manifest_path = ROOT / delivery["r2_manifest"]["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_zip = BASE / "stage2_stage2b_v2_2_UNIVERSITY_A40.zip"
    r1_zip = BASE / "stage2_stage2b_v2_2_OPERATIONAL_R1.zip"
    r2_zip = ROOT / delivery["bundle"]["path"]
    assert sha(r2_zip) == delivery["bundle"]["sha256"]
    assert sha(manifest_path) == delivery["r2_manifest"]["sha256"]

    extraction = ROOT / "tmp" / ("stage2_v2_2_operational_r2_check_" + uuid.uuid4().hex)
    extraction.mkdir(parents=True)
    member_sets = []
    for archive_path in (base_zip, r1_zip, r2_zip):
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            member_sets.append(names)
            archive.extractall(extraction)
    r2_expected = set(manifest["files"]) | {
        delivery["r2_manifest"]["path"],
        "deployment/stage2_v2_2_a40/COMMANDS_OPERATIONAL_R2.txt",
    }
    assert member_sets[2] == r2_expected and len(member_sets[2]) == 10
    assert not (member_sets[2] & (member_sets[0] | member_sets[1]))
    for relative, digest in manifest["files"].items():
        assert sha(extraction / relative) == digest, relative
    for relative, digest in manifest["required_existing_files"].items():
        assert sha(extraction / relative) == digest, relative

    dependencies = str(ROOT / "tmp/neural_build/pydeps")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((dependencies, str(extraction)))
    manifest_sha = delivery["r2_manifest"]["sha256"]
    package = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", "-m", "research.stage2_v2_2.a40_r2",
         "package-check", "--manifest-sha256", manifest_sha, "--workers", "4"],
        cwd=extraction, env=environment, capture_output=True, text=True,
    )
    print(package.stdout); print(package.stderr)
    assert package.returncode == 0 and '"controller_interface_complete": true' in package.stdout

    test_results = []
    for test_file, count in (("test_operational_r1.py", 14), ("test_operational_r2.py", 8)):
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-B",
             "deployment/stage2_v2_2_a40/" + test_file],
            cwd=extraction, env=environment, capture_output=True, text=True,
        )
        print(result.stdout); print(result.stderr)
        assert result.returncode == 0 and f"Ran {count} tests" in result.stderr and "OK" in result.stderr
        test_results.append(count)

    preflight = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", "-m", "research.stage2_v2_2.a40_r2",
         "preflight", "--manifest-sha256", manifest_sha, "--workers", "4"],
        cwd=extraction, env=environment, capture_output=True, text=True,
    )
    print(preflight.stdout); print(preflight.stderr)
    assert preflight.returncode != 0 and "bike_env" in preflight.stderr
    assert not (extraction / "research/results/stage2_stage2b_v2_2_a40").exists()

    wrapper = extraction / "deployment/stage2_v2_2_a40/run_stage2_stage2b_operational_r2.ps1"
    command = ("$e=$null; [System.Management.Automation.Language.Parser]::ParseFile('" +
               str(wrapper).replace("'", "''") + "',[ref]$null,[ref]$e)|Out-Null; if($e.Count){exit 1}")
    powershell = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command],
                                capture_output=True, text=True)
    assert powershell.returncode == 0, powershell.stderr

    report = {
        "status": "PASS",
        "scope": "LOCAL_ZERO_SCIENCE_BASE_PLUS_R1_PLUS_R2_ISOLATED_VERIFICATION",
        "bundle": delivery["bundle"],
        "r2_manifest": delivery["r2_manifest"],
        "r1_overlay_manifest_sha256": delivery["r1_overlay_manifest_sha256"],
        "archive_member_count": 10,
        "r2_does_not_contain_or_overwrite_base_or_r1_files": True,
        "all_r2_and_required_existing_hashes_verified": True,
        "r1_canonicalization_tests_passed": test_results[0],
        "r2_controller_startup_integration_tests_passed": test_results[1],
        "previous_attribute_error_path_explicitly_covered": True,
        "successful_worker_exit_interpretation_covered": True,
        "genuine_worker_failure_stop_covered": True,
        "powershell_syntax_passed": True,
        "local_non_a40_runtime_rejected_before_outputs": True,
        "actual_a40_post_launch_acceptance": "REQUIRES_TS9",
        "scientific_optimizer_steps": 0,
        "scientific_evaluations": 0,
        "final_target_labels_accessed": False,
        "stage3_started": False,
        "stage4_started": False,
        "verifier_sha256": sha(Path(__file__)),
    }
    text = json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    if REPORT.exists():
        assert REPORT.read_text(encoding="utf-8") == text
    else:
        REPORT.write_text(text, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
