from __future__ import annotations

import argparse
import json
from pathlib import Path


def _paths(parser):
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--raw-data-root", required=True, type=Path)
    parser.add_argument("--target-authority", required=True, type=Path)
    parser.add_argument("--r4-package-manifest", required=True, type=Path)
    parser.add_argument("--r4-package-sha256", required=True)
    parser.add_argument("--recovery-package-manifest", required=True, type=Path)
    parser.add_argument("--recovery-package-sha256", required=True)
    parser.add_argument("--scoring-package-manifest", required=True, type=Path)
    parser.add_argument("--scoring-package-sha256", required=True)
    parser.add_argument("--authorization", required=True, type=Path)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("score", "validate-results", "freeze"):
        child = sub.add_parser(command)
        _paths(child)
    child = sub.add_parser("validate-final")
    child.add_argument("--stage-root", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "validate-final":
        from . import validator
        result = validator.validate_frozen(args.stage_root)
    else:
        common = (
            args.authorization, args.stage_root, args.raw_data_root,
            args.target_authority, args.r4_package_manifest,
            args.r4_package_sha256, args.recovery_package_manifest,
            args.recovery_package_sha256, args.scoring_package_manifest,
            args.scoring_package_sha256)
        if args.command == "score":
            from . import scorer
            result = scorer.score(*common)
        elif args.command == "validate-results":
            from . import validator
            result = validator.validate_results(*common)
        else:
            from . import freeze
            result = freeze.freeze(*common)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

