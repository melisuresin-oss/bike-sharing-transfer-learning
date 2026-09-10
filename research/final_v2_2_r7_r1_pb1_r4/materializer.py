"""PB1 R4 materialization from same-handle verified stage-local snapshots."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable

import numpy as np

from research.final_v2_2_r7_r1_pb1 import authorization
from research.final_v2_2_r7_r1_pb1 import materializer as r3_materializer

from . import core


def _source_identity(value) -> dict:
    return {"st_dev": int(value.st_dev), "st_ino": int(value.st_ino),
            "st_size": int(value.st_size), "st_mtime_ns": int(value.st_mtime_ns),
            "st_ctime_ns": int(value.st_ctime_ns), "st_mode": int(value.st_mode)}


def _snapshot_relative(index: int, expected: dict) -> Path:
    token = hashlib.sha256(expected["path"].encode("utf-8")).hexdigest()[:24]
    return core.PB1_OUT / "verified_raw_snapshots" / (
        f"{index:02d}_{expected['role']}_{token}_{expected['sha256'][:16]}.parquet")


def _snapshot_bound_file(path: Path, expected: dict, snapshot_path: Path,
                         stage_root: Path,
                         after_open_hook: Callable[[Path], None] | None = None) -> dict:
    source = core.no_symlink_path(path)
    stage_root = core.no_symlink_path(stage_root, require_file=False)
    snapshot_path = snapshot_path.absolute()
    snapshot_path.relative_to(stage_root)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_path.exists():
        raise FileExistsError("verified snapshot already exists: " + str(snapshot_path))
    pending = snapshot_path.with_name(snapshot_path.name + ".pending-" + os.urandom(8).hex())
    digest = hashlib.sha256(); total = 0
    try:
        with source.open("rb") as handle, pending.open("xb") as output:
            before = os.fstat(handle.fileno())
            if after_open_hook is not None:
                after_open_hook(source)
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk); total += len(chunk); output.write(chunk)
            after = os.fstat(handle.fileno())
            output.flush(); os.fsync(output.fileno())
        before_identity, after_identity = _source_identity(before), _source_identity(after)
        if before_identity != after_identity:
            raise PermissionError("raw source changed while snapshotting")
        if total != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
            raise PermissionError("raw source identity mismatch: " + expected["path"])
        os.link(pending, snapshot_path)
    finally:
        try:
            pending.unlink()
        except FileNotFoundError:
            pass
    snapshot = core.entry(snapshot_path, relative_to=stage_root)
    if snapshot["bytes"] != expected["bytes"] or snapshot["sha256"] != expected["sha256"]:
        raise PermissionError("published snapshot identity mismatch")
    return {
        "source_relative_path": expected["path"],
        "registered_source_bytes": expected["bytes"],
        "registered_source_sha256": expected["sha256"],
        "role": expected["role"],
        "source_identity_before": before_identity,
        "source_identity_after": after_identity,
        "snapshot": snapshot,
        "verification_status": "PASS_SAME_HANDLE_STREAMED_VERIFIED_SNAPSHOT",
    }


def verify_raw_sources_after_authorization(capability: authorization.PhaseBCapability,
                                           authority_value: dict, raw_data_root: Path,
                                           after_open_hook: Callable[[Path], None] | None = None):
    if not isinstance(capability, authorization.PhaseBCapability):
        raise PermissionError("valid PB1 capability required")
    raw_data_root = core.no_symlink_path(raw_data_root, require_file=False)
    if str(raw_data_root) != capability.raw_data_root_canonical_path:
        raise PermissionError("raw root differs from authorization")
    stage_root = core.no_symlink_path(Path(capability.stage_root_canonical_path), require_file=False)
    trips, status, verified = [], [], []
    for index, expected in enumerate(authority_value["files"]):
        source = core.bound_file(raw_data_root, expected["path"])
        snapshot_path = stage_root / _snapshot_relative(index, expected)
        record = _snapshot_bound_file(source, expected, snapshot_path, stage_root, after_open_hook)
        snapshot = core.verify_entry(stage_root, record["snapshot"])
        verified.append(record)
        (trips if expected["role"] == "trips" else status).append(snapshot)
    if len(trips) != 6 or len(status) != 5:
        raise PermissionError("raw authority role count mismatch")
    return trips, status, verified


def _verified_snapshot_inputs(stage_root: Path, trips: list[Path], status_files: list[Path]) -> None:
    snapshot_root = core.no_symlink_path(stage_root / core.PB1_OUT / "verified_raw_snapshots",
                                         require_file=False)
    if len(trips) != 6 or len(status_files) != 5:
        raise PermissionError("materializer requires exactly eleven verified snapshots")
    for path in trips + status_files:
        core.no_symlink_path(path).relative_to(snapshot_root)


def _materialize_arrays(stage_root: Path, trips: list[Path], status_files: list[Path]):
    _verified_snapshot_inputs(stage_root, trips, status_files)
    return r3_materializer._materialize_arrays(stage_root, trips, status_files)


def materialize(authorization_path: Path, stage_root: Path, raw_data_root: Path,
                target_authority_path: Path, pb1_package_manifest: Path,
                pb1_package_sha: str) -> dict:
    capability = authorization.validate(authorization_path, stage_root, raw_data_root,
                                        target_authority_path, pb1_package_manifest, pb1_package_sha)
    stage_root = Path(capability.stage_root_canonical_path)
    output_root = stage_root / core.PB1_OUT
    event_path = output_root / "first_label_access.json"
    target_path = output_root / "materialized_targets.npz"
    provenance_path = output_root / "target_materialization_provenance.json"
    for path in (event_path, target_path, provenance_path):
        if path.exists():
            raise FileExistsError("repeat/conflicting Phase-B publication: " + str(path))
    authority_path = core.no_symlink_path(target_authority_path)
    authority_value = core.read_json(authority_path)
    if core.sha256_file(authority_path) != capability.target_authority_sha256:
        raise PermissionError("target authority changed after authorization")
    event = {
        "schema_version": core.ACCESS_SCHEMA,
        "status": "FIRST_HELD_OUT_BYTE_ACCESS_ABOUT_TO_BEGIN",
        "execution_id": capability.execution_id,
        "authorization": core.entry(core.no_symlink_path(authorization_path), relative_to=stage_root),
        "target_authority": core.entry(authority_path, relative_to=stage_root),
        "raw_data_root_canonical_path": capability.raw_data_root_canonical_path,
        "event_utc": core.utc(),
        "next_operation": "same-handle verified stage-local snapshot creation",
        "raw_bytes_read_before_this_event": False,
    }
    event_entry = core.atomic_json(event_path, event)
    trips, status_files, verified = verify_raw_sources_after_authorization(
        capability, authority_value, Path(capability.raw_data_root_canonical_path))
    arrays = _materialize_arrays(stage_root, trips, status_files)
    if set(map(int, np.unique(arrays["city_id"]))) != set(core.TARGETS):
        raise RuntimeError("materialized target-city set mismatch")
    if np.any(arrays["count"][arrays["label_valid"]] < 0) or np.any(arrays["count"][~arrays["label_valid"]] != -1):
        raise RuntimeError("label-valid/sentinel semantics mismatch")
    target_entry = core.atomic_bytes(target_path, core.deterministic_npz(arrays))
    provenance = {
        "schema_version": core.MATERIALIZATION_SCHEMA,
        "status": "PB1_R4_TARGETS_MATERIALIZED_FROM_VERIFIED_SNAPSHOTS",
        "execution_id": capability.execution_id,
        "authorization_sha256": capability.authorization_sha256,
        "target_authority_sha256": capability.target_authority_sha256,
        "first_label_access": event_entry,
        "verified_snapshots": verified,
        "parsed_snapshot_paths": [item["snapshot"]["path"] for item in verified],
        "original_raw_paths_parsed": False,
        "verified_snapshot_root_sha256": core.canonical_hash(verified),
        "materialized_targets": target_entry,
        "target_cities": list(core.TARGETS),
        "evaluation_hours_per_city": core.EVAL_HOURS,
        "scientific_summaries_emitted": False,
        "raw_rows_printed": False,
        "created_utc": core.utc(),
    }
    provenance_entry = core.atomic_json(provenance_path, provenance)
    return {"status": "PASS_PB1_R4_TARGET_MATERIALIZATION", "targets": target_entry,
            "provenance": provenance_entry, "verified_snapshots": 11,
            "raw_rows_printed": False, "scientific_summaries_emitted": False}
