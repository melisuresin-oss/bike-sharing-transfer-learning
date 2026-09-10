"""R6 scoring-fix constants layered over the immutable R5 recovery."""
from pathlib import Path

from research.final_v2_2_r7_r1_pb1_r5_recovery.core import *  # noqa: F401,F403

SCORING_REVISION = "final_v2_2_r7_r1_pb1_r6_scoring_fix"
SCORING_PROVENANCE_SCHEMA = SCORING_REVISION + ".provenance.1"

EXPECTED_MATERIALIZED_TARGETS_SHA256 = "0f33130bbb0fe2e94c1f682b00be57078c7c2b2b8932bc1dcff1707a0404e29b"
EXPECTED_RECOVERY_PROVENANCE_SHA256 = "60ce6468f5cc9146af2b49fd77e11b6bd8764fc596efcdc675cab07c3cce23d7"
EXPECTED_R5_PACKAGE_SHA256 = "a3d550f5c2cc7abc24fa4a26436a64e56eda74a11082d546b30cf08e056e09f9"
EXPECTED_R5_ZIP_SHA256 = "f8d5bff3875559ba0a86f986eef3374b809fd76d4bbc9ef384af0d520b801f91"

R5_PACKAGE_RELATIVE = Path(
    "deployment/final_v2_2_r7_r1_pb1_r5_recovery/package_manifest_recovery.json"
)
SCORING_PACKAGE_RELATIVE = Path(
    "deployment/final_v2_2_r7_r1_pb1_r6_scoring_fix/package_manifest_scoring_fix.json"
)
REGISTERED_ZERO_REFERENCES = ("HA_SOURCE", "persistence", "seasonal_naive")


def verify_scoring_package(stage_root: Path, package_manifest: Path,
                           package_sha256: str) -> dict:
    stage_root = no_symlink_path(stage_root, require_file=False)
    expected_path = (stage_root / SCORING_PACKAGE_RELATIVE).absolute()
    supplied = no_symlink_path(package_manifest)
    if supplied != expected_path:
        raise PermissionError("noncanonical R6 scoring-package path")
    if sha256_file(supplied) != package_sha256:
        raise PermissionError("R6 scoring-package hash mismatch")
    value = read_json(supplied)
    if (value.get("schema_version") != SCORING_REVISION + ".package.1" or
            value.get("status") != "PB1_R6_SCORING_FIX_PACKAGE_FROZEN_NOT_EXECUTED" or
            value.get("revision") != SCORING_REVISION or
            value.get("original_r5_package_sha256") != EXPECTED_R5_PACKAGE_SHA256 or
            value.get("real_scoring_executed") is not False):
        raise PermissionError("R6 scoring-package authority mismatch")
    for key in ("package_member_manifest", "executable_code_manifest",
                "implementation_contract", "synthetic_test_report"):
        verify_entry(stage_root, value[key])
    return entry(supplied, relative_to=stage_root)

