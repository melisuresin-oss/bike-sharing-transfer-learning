from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REVISION = "final_v2_2_r7_r1_pb1"
AUTH_SCHEMA = REVISION + ".authorization.1"
TARGET_AUTHORITY_SCHEMA = REVISION + ".target_authority.1"
ACCESS_SCHEMA = REVISION + ".first_label_access.1"
MATERIALIZATION_SCHEMA = REVISION + ".materialization.1"
METRICS_SCHEMA = REVISION + ".metrics.1"
BOOTSTRAP_SCHEMA = REVISION + ".bootstrap.1"
COMPARISONS_SCHEMA = REVISION + ".comparisons.1"
VALIDATION_SCHEMA = REVISION + ".validation.1"
PAYLOAD_MANIFEST_SCHEMA = REVISION + ".frozen_payload.1"
FINAL_MANIFEST_SCHEMA = REVISION + ".final_result.1"
STAGING_SCHEMA = REVISION + ".staging.1"

R7_PACKAGE_SHA = "f9696f41966066a2f952c7f99a42a86efccd7f932a5dde5822eec8aab726b51c"
R7_ARCHIVE_SHA = "df5d9295aa1de6bed2bf665fc511be9ac541c98b5133ad876fd7c4b5081b503b"
R7_MEMBER_SHA = "5200402e91669f1a1d47628b38415749ae33e40a75d32a7730745e374affe1b1"
R7_CODE_SHA = "93448a81935ff6bb598086d7181eb7ef0d773958f54bc69d0fc79c2280263b3f"
R7_IMPLEMENTATION_SHA = "f49e4bcaca359ac1d516433378e8bf6905e93de35f3b0b1db798cda7f67cd42a"
JOB_SHA = "b415398fb627a89a46253bfed0089cf1e4d24b1ef5bc4847d77fa674f9aae908"
EVALUATION_SHA = "523b571672bcc27395d940dce8c4362ab2d21143c9590f756b6427ae694bb22e"
FINAL_DATA_SHA = "afffa1e539626839fd86f1f9a3f5e8aa7a7833b67dde43f839d9cf1ce527896a"
FINAL_PROTOCOL_MD_SHA = "c4a8fb1cc314cd93336269b8b1ee035771ce766c823e82476daac97e3e565f6c"
FINAL_PROTOCOL_JSON_SHA = "16617ca602e2d70cc621005a838dfb84d308b5f33488bac8b92d38fccea1d054"
GRL_MD_SHA = "9e96cf5262861ec83d722ec6db18f97c66526a6e493a4970062635a2e4e25843"
GRL_JSON_SHA = "9324c11efac316607c39dd67d436e2efe57dd3fcf3bf2b7d2002d78494438037"
PREDICTION_COMMITMENT_SHA = "4a0820294a3896ec2464b87a9fb7577cae12f9aa807343dc5a319f669c77c0a4"
PREDICTION_MANIFEST_SHA = "cb2f45cbdd6425d01d28ac0b36e41796c12f0ece3b29b5255c13da340dee0b87"
PHASE_A_ARCHIVE_SHA = "710596843b3c7d25154729ff3e34866c6caa3ea1092251e8f54c0d01e4f63e7e"
PHASE_A_ARCHIVE_BYTES = 199_326_267

TARGETS = (195, 199, 237, 617)
FINAL_SEEDS = (17, 29, 43, 71, 101)
BUDGETS = ("0", "1", "7", "30", "full")
HT = 1_683_817_200_000_000
HE = 1_689_451_200_000_000
HOUR = 3_600_000_000
EVAL_HOURS = 1565

R7_PATHS = {
    "r7_package_manifest": ("deployment/final_v2_2_r7_r1/package_manifest.json", R7_PACKAGE_SHA),
    "r7_member_manifest": ("deployment/final_v2_2_r7_r1/package_member_manifest.json", R7_MEMBER_SHA),
    "r7_code_manifest": ("research/results/final_v2_2_r7_r1_execution_seal/executable_code_manifest.json", R7_CODE_SHA),
    "r7_implementation_contract": ("research/results/final_v2_2_r7_r1_execution_seal/implementation_contract.json", R7_IMPLEMENTATION_SHA),
    "job_manifest": ("research/results/final_v2_2_execution_seal/final_job_manifest.json", JOB_SHA),
    "evaluation_manifest": ("research/results/final_v2_2_execution_seal/evaluation_manifest.json", EVALUATION_SHA),
    "final_data_manifest": ("processed/protocol_v2_2/FINAL_DATASET_MANIFEST.json", FINAL_DATA_SHA),
    "final_protocol_markdown": ("FINAL_EVALUATION_PROTOCOL_V2_2_AMENDMENT.md", FINAL_PROTOCOL_MD_SHA),
    "final_protocol_json": ("final_evaluation_protocol_v2_2.json", FINAL_PROTOCOL_JSON_SHA),
    "grl_clarification_markdown": ("FINAL_GRL_SOURCE_DOMAIN_CARDINALITY_CLARIFICATION_V2_2.md", GRL_MD_SHA),
    "grl_clarification_json": ("final_grl_source_domain_cardinality_v2_2.json", GRL_JSON_SHA),
}

PHASE_A_OUT = Path("research/results/final_v2_2_r7_r1_a40")
PB1_OUT = Path("research/results/final_v2_2_r7_r1_pb1")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def pretty_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def entry(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    result = {"path": str(path.resolve()) if relative_to is None else path.resolve().relative_to(relative_to.resolve()).as_posix(),
              "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def atomic_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"append-only publication already exists: {path}")
    pending = path.with_name(path.name + ".pending-" + os.urandom(8).hex())
    try:
        with pending.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(pending, path)
    finally:
        try:
            pending.unlink()
        except FileNotFoundError:
            pass
    return entry(path)


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    return atomic_bytes(path, pretty_bytes(value))


def canonical_relative(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("canonical repository-relative POSIX path required")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError("path traversal/substitution forbidden")
    return path


def no_symlink_path(path: Path, *, require_file: bool = True) -> Path:
    supplied = path.absolute()
    if not supplied.exists() or (require_file and not supplied.is_file()):
        raise FileNotFoundError(supplied)
    for part in (supplied, *supplied.parents):
        if part.exists() and part.is_symlink():
            raise PermissionError(f"symlink substitution forbidden: {part}")
    resolved = supplied.resolve(strict=True)
    if resolved != supplied:
        raise PermissionError(f"path substitution forbidden: {supplied}")
    return resolved


def bound_file(root: Path, relative: str) -> Path:
    rel = canonical_relative(relative)
    root = no_symlink_path(root, require_file=False)
    supplied = (root / rel).absolute()
    resolved = no_symlink_path(supplied)
    resolved.relative_to(root)
    return resolved


def verify_entry(root: Path, item: Mapping[str, Any]) -> Path:
    if set(item) < {"path", "bytes", "sha256"}:
        raise ValueError("incomplete file entry")
    path = bound_file(root, str(item["path"]))
    if path.stat().st_size != int(item["bytes"]) or sha256_file(path) != item["sha256"]:
        raise PermissionError(f"bound file changed: {item['path']}")
    return path


def deterministic_npz(arrays: Mapping[str, Any]) -> bytes:
    import numpy as np
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            value = io.BytesIO()
            np.lib.format.write_array(value, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o444) << 16
            archive.writestr(info, value.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def verify_fixed_file(root: Path, relative: str, expected_sha: str) -> dict[str, Any]:
    path = bound_file(root, relative)
    actual = sha256_file(path)
    if actual != expected_sha:
        raise PermissionError(f"frozen hash mismatch: {relative}")
    return entry(path, relative_to=root)

