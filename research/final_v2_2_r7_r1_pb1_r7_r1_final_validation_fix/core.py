"""Constants for the frozen R7 final-validation recovery."""
from pathlib import Path

from research.final_v2_2_r7_r1_pb1_r6_scoring_fix.core import *  # noqa: F401,F403

VALIDATION_REVISION = "final_v2_2_r7_r1_pb1_r7_r1_final_validation_fix"
VALIDATION_REPORT_SCHEMA = VALIDATION_REVISION + ".report.1"
EXPECTED_FINAL_MANIFEST_SHA256 = "b5a3d1cf3d29228e85ef4a2444a09ff0a1636862fc32975f4add847f50a514d0"
EXPECTED_FINAL_ARCHIVE_SHA256 = "37e9443a71029d2972c3b35262bb33b328060dda9b1a93486a186b68cddf2d55"
EXPECTED_R6_PACKAGE_SHA256 = "e98f8f7ab320c83f0638dd92636f36aea946656d9e8dc670a40614546ab854e7"
VALIDATION_PACKAGE_RELATIVE = Path(
    "deployment/final_v2_2_r7_r1_pb1_r7_r1_final_validation_fix/package_manifest_validation_fix.json"
)
VALIDATION_REPORT_RELATIVE = Path(
    "research/results/final_v2_2_r7_r1_pb1_r7_r1_final_validation_fix/final_validation_recovery_report.json"
)


def verify_validation_package(stage_root: Path, package_manifest: Path,
                              package_sha256: str) -> dict:
    stage_root = no_symlink_path(stage_root, require_file=False)
    expected = (stage_root / VALIDATION_PACKAGE_RELATIVE).absolute()
    supplied = no_symlink_path(package_manifest)
    if supplied != expected or sha256_file(supplied) != package_sha256:
        raise PermissionError("R7 validation-fix package identity mismatch")
    value = read_json(supplied)
    if (value.get("schema_version") != VALIDATION_REVISION + ".package.1" or
            value.get("status") != "PB1_R7_FINAL_VALIDATION_FIX_PACKAGE_FROZEN" or
            value.get("revision") != VALIDATION_REVISION or
            value.get("frozen_final_manifest_sha256") != EXPECTED_FINAL_MANIFEST_SHA256 or
            value.get("frozen_final_archive_sha256") != EXPECTED_FINAL_ARCHIVE_SHA256):
        raise PermissionError("R7 validation-fix package authority mismatch")
    for key in ("package_member_manifest", "executable_code_manifest",
                "implementation_contract", "synthetic_test_report"):
        verify_entry(stage_root, value[key])
    return entry(supplied, relative_to=stage_root)


