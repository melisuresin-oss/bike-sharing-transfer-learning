from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import core


def _paths(parser):
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--raw-data-root", required=True, type=Path)
    parser.add_argument("--target-authority", required=True, type=Path)
    parser.add_argument("--pb1-package-manifest", required=True, type=Path)
    parser.add_argument("--pb1-package-sha256", required=True)


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("create-target-authority"); p.add_argument("--final-data-manifest", required=True, type=Path); p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("stage"); p.add_argument("--stage-root", required=True, type=Path); p.add_argument("--r7-archive", required=True, type=Path); p.add_argument("--phase-a-archive", required=True, type=Path)
    p = sub.add_parser("create-authorization"); _paths(p); p.add_argument("--output", required=True, type=Path)
    for command in ("validate-authorization", "materialize", "score", "validate-results", "freeze"):
        p = sub.add_parser(command); _paths(p); p.add_argument("--authorization", required=True, type=Path)
    p = sub.add_parser("validate-final"); p.add_argument("--stage-root", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "create-target-authority":
        from . import authority
        result = authority.create(args.final_data_manifest, args.output)
    elif args.command == "stage":
        from . import staging
        result = staging.stage(args.stage_root, args.r7_archive, args.phase_a_archive)
    elif args.command == "create-authorization":
        from . import authorization
        result = authorization.create(args.stage_root, args.raw_data_root, args.target_authority, args.pb1_package_manifest, args.pb1_package_sha256, args.output)
    elif args.command == "validate-authorization":
        from . import authorization
        capability = authorization.validate(args.authorization, args.stage_root, args.raw_data_root, args.target_authority, args.pb1_package_manifest, args.pb1_package_sha256)
        result = {"status": "PASS_PB1_AUTHORIZATION", "authorization_sha256": capability.authorization_sha256,
                  "execution_id": capability.execution_id, "raw_content_opened": False}
    elif args.command == "materialize":
        from . import materializer
        result = materializer.materialize(args.authorization, args.stage_root, args.raw_data_root, args.target_authority, args.pb1_package_manifest, args.pb1_package_sha256)
    elif args.command == "score":
        from . import scorer
        result = scorer.score(args.authorization, args.stage_root, args.raw_data_root, args.target_authority, args.pb1_package_manifest, args.pb1_package_sha256)
    elif args.command == "validate-results":
        from . import validator
        result = validator.validate_results(args.authorization, args.stage_root, args.raw_data_root, args.target_authority, args.pb1_package_manifest, args.pb1_package_sha256)
    elif args.command == "freeze":
        from . import freeze
        result = freeze.freeze(args.authorization, args.stage_root, args.raw_data_root, args.target_authority, args.pb1_package_manifest, args.pb1_package_sha256)
    else:
        from . import validator
        result = validator.validate_frozen(args.stage_root)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
