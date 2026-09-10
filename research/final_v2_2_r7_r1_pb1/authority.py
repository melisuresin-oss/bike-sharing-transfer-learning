from __future__ import annotations

import re
from pathlib import Path

from . import core


def authority_payload(final_data_manifest: Path) -> dict:
    final_data_manifest = core.no_symlink_path(final_data_manifest)
    if core.sha256_file(final_data_manifest) != core.FINAL_DATA_SHA:
        raise PermissionError("final-data manifest hash mismatch")
    manifest = core.read_json(final_data_manifest)
    raw = manifest.get("raw_bindings")
    if not isinstance(raw, dict) or set(raw) != {"dataset", "revision", "files"}:
        raise ValueError("registered raw_bindings schema mismatch")
    files = []
    for item in raw["files"]:
        if set(item) != {"path", "sha256", "bytes"}:
            raise ValueError("registered raw-file entry mismatch")
        rel = core.canonical_relative(item["path"]).as_posix()
        digest = item["sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or type(item["bytes"]) is not int or item["bytes"] <= 0:
            raise ValueError("invalid registered raw identity")
        files.append({"path": rel, "bytes": item["bytes"], "sha256": digest,
                      "role": "trips" if "/trips/" in f"/{rel}" else "station_status"})
    files.sort(key=lambda item: item["path"])
    if len(files) != 11 or sum(x["role"] == "trips" for x in files) != 6 or sum(x["role"] == "station_status" for x in files) != 5:
        raise ValueError("expected six trip and five station-status shards")
    if files != sorted(files, key=lambda item: item["path"]):
        raise AssertionError("deterministic order failure")
    return {
        "schema_version": core.TARGET_AUTHORITY_SCHEMA,
        "status": "PB1_METADATA_ONLY_MULTI_FILE_TARGET_AUTHORITY",
        "derivation": "sealed FINAL_DATASET_MANIFEST.raw_bindings only; no raw file opened or hashed",
        "final_data_manifest": {"path": "processed/protocol_v2_2/FINAL_DATASET_MANIFEST.json",
                                "bytes": final_data_manifest.stat().st_size,
                                "sha256": core.FINAL_DATA_SHA},
        "upstream": {"dataset": raw["dataset"], "revision": raw["revision"]},
        "ordering": "canonical_relative_path_ascending",
        "files": files,
        "raw_file_count": 11,
        "raw_content_opened": False,
        "raw_hashes_recomputed": False,
    }


def create(final_data_manifest: Path, output: Path) -> dict:
    payload = authority_payload(final_data_manifest)
    result = core.atomic_json(output, payload)
    return {"status": "PB1_TARGET_AUTHORITY_PUBLISHED", "target_authority": result,
            "raw_content_opened": False, "raw_hashes_recomputed": False}


def validate(path: Path, final_data_manifest: Path) -> dict:
    path = core.no_symlink_path(path)
    value = core.read_json(path)
    if value != authority_payload(final_data_manifest):
        raise PermissionError("target authority is not the deterministic metadata-only authority")
    return {"status": "PASS_PB1_TARGET_AUTHORITY", "target_authority": core.entry(path),
            "raw_content_opened": False}

