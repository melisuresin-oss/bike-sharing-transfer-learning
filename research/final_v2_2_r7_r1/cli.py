"""R7 Phase-A command surface. There is deliberately no Phase-B command."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from . import core

COMMANDS = ("package-check", "preflight", "dry-run", "validate-existing", "launch",
            "controller-status", "postrun-validate", "predict", "validate-predictions", "commit-predictions", "archive-phase-a")

def archive_phase_a(package_sha):
    from . import commitment, validation
    validation.postrun_validate(package_sha); commitment.verify(core.PREDICTION_COMMITMENT, package_sha)
    root=core.path(core.OUT); destination=root/"archives"/"final_v2_2_r7_r1_phase_a.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists(): raise FileExistsError("append-only Phase-A archive already exists")
    files=[p for p in root.rglob("*") if p.is_file() and destination not in p.parents and p != destination]
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for value in sorted(files): archive.write(value, value.relative_to(core.ROOT).as_posix())
    return {"status": "PHASE_A_ARCHIVED", "archive": {"path": destination.relative_to(core.ROOT).as_posix(),
        "bytes": destination.stat().st_size, "sha256": core.sha256_path(destination)},
        "phase_b_targets_included": False, "final_labels_accessed": False}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--package-sha256", required=True); parser.add_argument("--structural-only", action="store_true"); parser.add_argument("--controller-uuid")
    args=parser.parse_args(); command=args.command
    if command == "package-check":
        from .validation import package_check; result=package_check(args.package_sha256)
    elif command == "preflight":
        from .validation import preflight; result=preflight(args.package_sha256, strict=not args.structural_only)
    elif command == "dry-run":
        from .validation import dry_run; result=dry_run(args.package_sha256)
    elif command == "validate-existing":
        from .validation import validate_existing; result=validate_existing(args.package_sha256)
    elif command == "controller-status":
        from .controller import controller_status; result=controller_status()
    elif command == "launch":
        if args.structural_only: raise PermissionError("launch cannot use structural-only mode")
        from .controller import launch; result=launch(args.package_sha256,args.controller_uuid)
    elif command == "postrun-validate":
        from .validation import postrun_validate; result=postrun_validate(args.package_sha256)
    elif command == "predict":
        from .predictions import predict_all; result=predict_all(args.package_sha256)
    elif command == "validate-predictions":
        from .predictions import validate_predictions; result=validate_predictions(args.package_sha256)
    elif command == "commit-predictions":
        from .commitment import commit; result=commit(args.package_sha256)
    else: result=archive_phase_a(args.package_sha256)
    print(json.dumps(result, indent=2), flush=True)

if __name__ == "__main__": main()

