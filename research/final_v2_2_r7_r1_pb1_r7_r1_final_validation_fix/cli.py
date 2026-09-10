from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate-final",))
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--validation-package-manifest", required=True, type=Path)
    parser.add_argument("--validation-package-sha256", required=True)
    args = parser.parse_args()
    from . import validator
    result = validator.validate_and_record(
        args.stage_root, args.validation_package_manifest,
        args.validation_package_sha256)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()


