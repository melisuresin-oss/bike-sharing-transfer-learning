from __future__ import annotations

import zipfile
from pathlib import Path

from . import core


def _safe_extract(archive_path: Path, root: Path, *, expected_sha: str, expected_bytes: int | None = None) -> list[dict]:
    archive_path = core.no_symlink_path(archive_path)
    if expected_bytes is not None and archive_path.stat().st_size != expected_bytes:
        raise PermissionError("archive byte-size mismatch")
    if core.sha256_file(archive_path) != expected_sha:
        raise PermissionError("archive SHA-256 mismatch")
    entries = []
    with zipfile.ZipFile(archive_path) as archive:
        names = [x.filename for x in archive.infolist() if not x.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive member")
        for item in archive.infolist():
            if item.is_dir():
                continue
            rel = core.canonical_relative(item.filename).as_posix()
            destination = (root / rel).absolute()
            destination.resolve(strict=False).relative_to(root)
            if destination.exists():
                raise FileExistsError(f"staging collision: {rel}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = archive.read(item)
            if len(payload) != item.file_size:
                raise IOError("truncated archive member")
            core.atomic_bytes(destination, payload)
            entries.append({"path": rel, "bytes": len(payload), "sha256": core.sha256_bytes(payload)})
    return sorted(entries, key=lambda item: item["path"])


def stage(stage_root: Path, r7_archive: Path, phase_a_archive: Path) -> dict:
    stage_root = stage_root.absolute()
    if not stage_root.exists() or not stage_root.is_dir():
        raise FileNotFoundError("PB1 package must be extracted into a fresh stage root first")
    output = stage_root / core.PB1_OUT / "staging_manifest.json"
    if output.exists():
        raise FileExistsError("stage already sealed")
    r7_files = _safe_extract(r7_archive, stage_root, expected_sha=core.R7_ARCHIVE_SHA)
    phase_a_files = _safe_extract(phase_a_archive, stage_root, expected_sha=core.PHASE_A_ARCHIVE_SHA,
                                  expected_bytes=core.PHASE_A_ARCHIVE_BYTES)
    commitment = stage_root / core.PHASE_A_OUT / "predictions/prediction_commitment.json"
    prediction_manifest = stage_root / core.PHASE_A_OUT / "predictions/prediction_manifest.json"
    if core.sha256_file(commitment) != core.PREDICTION_COMMITMENT_SHA:
        raise PermissionError("staged commitment mismatch")
    if core.sha256_file(prediction_manifest) != core.PREDICTION_MANIFEST_SHA:
        raise PermissionError("staged prediction manifest mismatch")
    payload = {
        "schema_version": core.STAGING_SCHEMA,
        "status": "PB1_PHASE_A_STAGED_IMMUTABLY",
        "stage_root_canonical_path": str(stage_root.resolve()),
        "r7_archive": core.entry(r7_archive),
        "phase_a_archive": core.entry(phase_a_archive),
        "r7_files": r7_files,
        "phase_a_files": phase_a_files,
        "counts": {"r7_files": len(r7_files), "phase_a_files": len(phase_a_files)},
        "source_archives_modified": False,
        "held_out_sources_opened": False,
        "created_utc": core.utc(),
    }
    result = core.atomic_json(output, payload)
    return {"status": "PASS_PB1_STAGING", "staging_manifest": result,
            "phase_a_files": len(phase_a_files), "held_out_sources_opened": False}


def validate(stage_root: Path) -> dict:
    stage_root = core.no_symlink_path(stage_root, require_file=False)
    manifest_path = stage_root / core.PB1_OUT / "staging_manifest.json"
    manifest = core.read_json(core.no_symlink_path(manifest_path))
    if manifest.get("schema_version") != core.STAGING_SCHEMA or manifest.get("status") != "PB1_PHASE_A_STAGED_IMMUTABLY":
        raise PermissionError("staging manifest state mismatch")
    if manifest.get("stage_root_canonical_path") != str(stage_root):
        raise PermissionError("stage root substitution")
    if manifest.get("source_archives_modified") is not False or manifest.get("held_out_sources_opened") is not False:
        raise PermissionError("staging firewall mismatch")
    for name, expected_sha in (("r7_archive", core.R7_ARCHIVE_SHA), ("phase_a_archive", core.PHASE_A_ARCHIVE_SHA)):
        item = manifest[name]; path = core.no_symlink_path(Path(item["path"]))
        if path.stat().st_size != item["bytes"] or core.sha256_file(path) != expected_sha or item["sha256"] != expected_sha:
            raise PermissionError(name + " changed")
    for item in manifest["r7_files"] + manifest["phase_a_files"]:
        core.verify_entry(stage_root, item)
    if len(manifest["phase_a_files"]) != 4167:
        raise PermissionError("Phase-A member count mismatch")
    return {"status": "PASS_PB1_STAGED_PHASE_A_IMMUTABLE", "phase_a_files": 4167,
            "held_out_sources_opened": False}

