"""Authorized held-out materialization. Importing this module opens no raw source."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable

import numpy as np

from research.v2_2.contract import Contract
from research.v2_2.history import StatusEvent, StatusIndex, coverage

from . import authorization, core


def _hash_bound_file(path: Path, expected: dict, after_open_hook: Callable[[Path], None] | None = None) -> dict:
    path = core.no_symlink_path(path)
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if after_open_hook is not None:
            after_open_hook(path)
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity != identity_after:
        raise PermissionError("raw source changed while hashing")
    if before.st_size != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
        raise PermissionError("raw source identity mismatch: " + expected["path"])
    return {"path": expected["path"], "bytes": before.st_size, "sha256": digest.hexdigest(),
            "role": expected["role"]}


def verify_raw_sources_after_authorization(capability: authorization.PhaseBCapability,
                                           authority_value: dict, raw_data_root: Path,
                                           after_open_hook: Callable[[Path], None] | None = None) -> tuple[list[Path], list[Path], list[dict]]:
    if not isinstance(capability, authorization.PhaseBCapability):
        raise PermissionError("valid PB1 capability required")
    raw_data_root = core.no_symlink_path(raw_data_root, require_file=False)
    if str(raw_data_root) != capability.raw_data_root_canonical_path:
        raise PermissionError("raw root differs from authorization")
    trips, status, verified = [], [], []
    for expected in authority_value["files"]:
        path = core.bound_file(raw_data_root, expected["path"])
        verified.append(_hash_bound_file(path, expected, after_open_hook))
        (trips if expected["role"] == "trips" else status).append(path)
    if len(trips) != 6 or len(status) != 5:
        raise PermissionError("raw authority role count mismatch")
    return trips, status, verified


def _parquet_list(paths: list[Path]) -> str:
    return "[" + ",".join("'" + str(path).replace("'", "''") + "'" for path in paths) + "]"


def _load_city_keys(stage_root: Path, manifest: dict, city_id: int):
    record = next(item for item in manifest["targets"] if item["city_id"] == city_id)
    key_path = core.verify_entry(stage_root, record["prediction_keys"])
    with np.load(key_path, allow_pickle=False) as data:
        if set(data.files) != {"city_id", "station_id", "forecast_origin_us"}:
            raise ValueError("prediction-key schema mismatch")
        city = np.asarray(data["city_id"], dtype=np.int64)
        station = np.asarray(data["station_id"], dtype=np.int64)
        origin = np.asarray(data["forecast_origin_us"], dtype=np.int64)
    if not (len(city) == len(station) == len(origin)) or np.any(city != city_id):
        raise ValueError("prediction-key city mismatch")
    if len(np.unique(origin)) != core.EVAL_HOURS or int(origin.min()) != core.HT or int(origin.max()) != core.HE - core.HOUR:
        raise ValueError("prediction-key final interval mismatch")
    if np.any(origin % core.HOUR):
        raise ValueError("unaligned prediction hour")
    opaque = (city.astype(np.uint64) << np.uint64(32)) | np.arange(len(city), dtype=np.uint64)
    return city, station, origin, opaque


def _materialize_arrays(stage_root: Path, trips: list[Path], status_files: list[Path]) -> dict[str, np.ndarray]:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("DuckDB 1.5.5 is required for PB1 Parquet materialization") from exc
    manifest = core.read_json(stage_root / core.R7_PATHS["final_data_manifest"][0])
    contract = Contract(root=stage_root)
    connection = duckdb.connect(":memory:")
    connection.execute("SET threads=2")
    rows = []
    try:
        trip_expr, status_expr = _parquet_list(trips), _parquet_list(status_files)
        for city_id in core.TARGETS:
            city, station, origin, opaque = _load_city_keys(stage_root, manifest, city_id)
            roster = contract.city(city_id)
            roster_ids = [item.station_id for item in roster]
            if set(map(int, np.unique(station))) != set(roster_ids):
                raise ValueError("prediction keys differ from sealed target cohort")
            ids = ",".join(map(str, sorted(roster_ids)))
            status_sql = f"""
                SELECT DISTINCT CAST(station_id AS BIGINT), epoch_us(to_timestamp(time))
                FROM read_parquet({status_expr})
                WHERE station_id IN ({ids}) AND time IS NOT NULL AND isfinite(time)
                ORDER BY 1,2
            """
            status_rows = [StatusEvent(int(sid), int(moment)) for sid, moment in connection.execute(status_sql).fetchall()]
            if not status_rows:
                raise RuntimeError("no status evidence for sealed target city")
            status_index = StatusIndex(contract, status_rows)
            adjudication = max(item.timestamp for item in status_rows if item.timestamp is not None)
            trip_sql = f"""
                SELECT CAST(station_id_start AS BIGINT),
                       epoch_us(date_trunc('hour',to_timestamp(time_start))),
                       count(*)::BIGINT
                FROM read_parquet({trip_expr})
                WHERE city_id={city_id}
                  AND station_id_start IN ({ids})
                  AND station_id_start IS NOT NULL AND isfinite(station_id_start)
                  AND station_id_start=trunc(station_id_start)
                  AND time_start IS NOT NULL AND isfinite(time_start)
                  AND time_start>={core.HT}/1000000.0
                  AND time_start<{core.HE}/1000000.0
                GROUP BY 1,2 ORDER BY 1,2
            """
            counts = {(int(sid), int(moment)): int(value) for sid, moment, value in connection.execute(trip_sql).fetchall()}
            coverage_by_hour = {}
            roster_index = {item.station_id: index for index, item in enumerate(roster)}
            for moment in sorted(map(int, np.unique(origin))):
                coverage_by_hour[moment] = coverage(contract, status_index, city_id, moment, adjudication).observed
            valid = np.zeros(len(city), dtype=np.bool_)
            values = np.full(len(city), -1, dtype=np.int64)
            for index, (sid, moment) in enumerate(zip(station, origin)):
                observed = bool(coverage_by_hour[int(moment)][roster_index[int(sid)]])
                valid[index] = observed
                if observed:
                    values[index] = counts.get((int(sid), int(moment)), 0)
            rows.append((city, station, origin, opaque, values, valid))
    finally:
        connection.close()
    return {
        "city_id": np.concatenate([item[0] for item in rows]).astype(np.int64),
        "station_id": np.concatenate([item[1] for item in rows]).astype(np.int64),
        "forecast_origin_us": np.concatenate([item[2] for item in rows]).astype(np.int64),
        "opaque_label_join_key": np.concatenate([item[3] for item in rows]).astype(np.uint64),
        "count": np.concatenate([item[4] for item in rows]).astype(np.int64),
        "label_valid": np.concatenate([item[5] for item in rows]).astype(np.bool_),
    }


def materialize(authorization_path: Path, stage_root: Path, raw_data_root: Path,
                target_authority_path: Path, pb1_package_manifest: Path,
                pb1_package_sha: str) -> dict:
    capability = authorization.validate(authorization_path, stage_root, raw_data_root,
                                        target_authority_path, pb1_package_manifest, pb1_package_sha)
    output_root = Path(capability.stage_root_canonical_path) / core.PB1_OUT
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
        "authorization": core.entry(core.no_symlink_path(authorization_path), relative_to=Path(capability.stage_root_canonical_path)),
        "target_authority": core.entry(authority_path, relative_to=Path(capability.stage_root_canonical_path)),
        "raw_data_root_canonical_path": capability.raw_data_root_canonical_path,
        "event_utc": core.utc(),
        "next_operation": "verify registered raw file hashes before parsing",
        "raw_bytes_read_before_this_event": False,
    }
    event_entry = core.atomic_json(event_path, event)
    trips, status_files, verified = verify_raw_sources_after_authorization(capability, authority_value,
                                                                           Path(capability.raw_data_root_canonical_path))
    arrays = _materialize_arrays(Path(capability.stage_root_canonical_path), trips, status_files)
    if set(map(int, np.unique(arrays["city_id"]))) != set(core.TARGETS):
        raise RuntimeError("materialized target-city set mismatch")
    if np.any(arrays["count"][arrays["label_valid"]] < 0) or np.any(arrays["count"][~arrays["label_valid"]] != -1):
        raise RuntimeError("label-valid/sentinel semantics mismatch")
    target_entry = core.atomic_bytes(target_path, core.deterministic_npz(arrays))
    provenance = {
        "schema_version": core.MATERIALIZATION_SCHEMA,
        "status": "PB1_TARGETS_MATERIALIZED_AFTER_AUTHORIZATION",
        "execution_id": capability.execution_id,
        "authorization_sha256": capability.authorization_sha256,
        "target_authority_sha256": capability.target_authority_sha256,
        "first_label_access": event_entry,
        "actual_raw_identities": verified,
        "materialized_targets": target_entry,
        "target_cities": list(core.TARGETS),
        "evaluation_hours_per_city": core.EVAL_HOURS,
        "scientific_summaries_emitted": False,
        "raw_rows_printed": False,
        "created_utc": core.utc(),
    }
    provenance_entry = core.atomic_json(provenance_path, provenance)
    return {"status": "PASS_PB1_TARGET_MATERIALIZATION", "targets": target_entry,
            "provenance": provenance_entry, "raw_rows_printed": False,
            "scientific_summaries_emitted": False}

