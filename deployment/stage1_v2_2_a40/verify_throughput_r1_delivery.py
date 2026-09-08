"""Independent, non-scientific verification of the throughput-R1 ZIP."""
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
BASE = ROOT / "deployment/stage1_v2_2_a40"
DELIVERY = BASE / "DELIVERY_MANIFEST_THROUGHPUT_R1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


delivery = json.loads(DELIVERY.read_text(encoding="utf-8"))
archive = ROOT / delivery["bundle"]["path"]
manifest_path = ROOT / delivery["bundle_manifest"]["path"]
commands_path = ROOT / delivery["commands"]["path"]
assert sha256(archive) == delivery["bundle"]["sha256"]
assert sha256(manifest_path) == delivery["bundle_manifest"]["sha256"]
assert sha256(commands_path) == delivery["commands"]["sha256"]

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = set(manifest["files"]) | {
    delivery["bundle_manifest"]["path"],
    delivery["commands"]["path"],
}
assert len(expected) == delivery["zip_member_count"]

extract = ROOT / "tmp" / ("stage1_v2_2_throughput_r1_check_" + uuid.uuid4().hex)
extract.mkdir(parents=True)
with zipfile.ZipFile(archive) as bundle:
    names = bundle.namelist()
    assert len(names) == len(expected) and set(names) == expected
    for info in bundle.infolist():
        (extract / info.filename).resolve().relative_to(extract.resolve())
        lowered = info.filename.lower()
        assert not any(token in lowered for token in (
            "final_label", "final_adaptation", "sealed_final", "__pycache__"
        ))
        assert not lowered.endswith(".pt")
    bundle.extractall(extract)

for relative, expected_hash in manifest["files"].items():
    assert sha256(extract / relative) == expected_hash, relative
assert sha256(extract / delivery["bundle_manifest"]["path"]) == delivery["bundle_manifest"]["sha256"]
assert sha256(extract / delivery["commands"]["path"]) == delivery["commands"]["sha256"]

dependencies = str(ROOT / "tmp/neural_build/pydeps")
environment = dict(os.environ)
environment["PYTHONPATH"] = dependencies
package_code = r"""
import sys
from pathlib import Path
sys.path[:0] = [sys.argv[1], str(Path.cwd())]
from research.stage1_v2_2 import a40_parallel as p
assert p.ROOT == Path.cwd()
assert p.DEFAULT_WORKERS == 4 and p.MAX_WORKERS == 4
p.verify_throughput_package(sys.argv[2])
try:
    p.throughput_preflight(sys.argv[2])
except RuntimeError as error:
    assert "bike_env" in str(error), str(error)
else:
    raise AssertionError("Local CPU incorrectly passed the A40 preflight")
assert not (Path.cwd() / p.base.OUT).exists()
print("ISOLATED_THROUGHPUT_PACKAGE_PASS; A40_GATE_REJECTED_LOCAL_CPU; ZERO_SCIENTIFIC_STEPS")
"""
package_check = subprocess.run(
    [sys.executable, "-X", "utf8", "-B", "-c", package_code, dependencies,
     delivery["bundle_manifest"]["sha256"]],
    cwd=extract,
    env=environment,
    capture_output=True,
    text=True,
)
assert package_check.returncode == 0, package_check.stdout + package_check.stderr

tests = subprocess.run(
    [sys.executable, "-X", "utf8", "-B",
     "deployment/stage1_v2_2_a40/test_throughput_r1.py"],
    cwd=extract,
    env=environment,
    capture_output=True,
    text=True,
)
assert tests.returncode == 0, tests.stdout + tests.stderr
assert "Ran 12 tests" in tests.stderr and "OK" in tests.stderr

ps_syntax = subprocess.run(
    ["powershell.exe", "-NoProfile", "-Command",
     "$e=$null; [void][System.Management.Automation.Language.Parser]::ParseFile("
     "(Resolve-Path '.\\deployment\\stage1_v2_2_a40\\run_a40_throughput_r1.ps1'),"
     "[ref]$null,[ref]$e); if($e.Count){$e | Out-String; exit 1}"],
    cwd=extract,
    capture_output=True,
    text=True,
)
assert ps_syntax.returncode == 0, ps_syntax.stdout + ps_syntax.stderr

report = {
    "status": "PASS",
    "scope": "LOCAL_NON_SCIENTIFIC_THROUGHPUT_PACKAGE_VERIFICATION",
    "bundle_sha256": delivery["bundle"]["sha256"],
    "bundle_manifest_sha256": delivery["bundle_manifest"]["sha256"],
    "archive_member_count": len(expected),
    "all_archive_member_hashes_verified": True,
    "standalone_code_and_data_closure": True,
    "isolated_extraction": str(extract),
    "throughput_tests_passed": 12,
    "powershell_syntax": "PASS",
    "local_cpu_rejected_by_A40_preflight": True,
    "scientific_training_steps": 0,
    "scientific_evaluations": 0,
    "final_target_labels_accessed": False,
    "actual_UNIVERSITY_A40_preflight": "NOT_RUN_LOCALLY_REQUIRES_TS9",
}
print(package_check.stdout.strip())
print(json.dumps(report, indent=2))
