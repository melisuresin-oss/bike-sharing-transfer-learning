from __future__ import annotations

import json

from research.governance.stage3_policy import verify_seal_bundle


def main() -> int:
    print(json.dumps(verify_seal_bundle(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
