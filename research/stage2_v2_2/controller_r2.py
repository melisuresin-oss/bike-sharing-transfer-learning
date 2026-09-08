"""Operational R2 controller with complete base-interface delegation."""
from __future__ import annotations

import argparse
import sys

from research.stage2_v2_2 import a40_r2 as r2
from research.stage2_v2_2 import controller as original


def _worker_command(stage: str, job_id: str, manifest_sha256: str,
                    preflight_path: str, workers: int) -> list[str]:
    return [
        sys.executable, "-X", "utf8", "-B", "-m", "research.stage2_v2_2.a40_r2", "job",
        "--manifest-sha256", manifest_sha256, "--workers", str(workers),
        "--stage", stage, "--job-id", job_id, "--preflight", preflight_path,
    ]


def _postrun_command(mode: str, stage: str, manifest_sha256: str, workers: int,
                     *, record: bool = False) -> list[str]:
    command = [
        sys.executable, "-X", "utf8", "-B", "-m", "research.stage2_v2_2.a40_r2", mode,
        "--manifest-sha256", manifest_sha256, "--workers", str(workers), "--stage", stage,
    ]
    if record:
        command.append("--record")
    return command


def launch(manifest_sha256: str, workers: int) -> None:
    r2.verify_package(manifest_sha256)
    prior_a40 = original.a40
    prior_worker = original._worker_command
    prior_postrun = original._postrun_command
    prior_r2_preflight = r2.preflight

    def preflight_and_retention_gate(overlay_sha256: str, count: int):
        result = prior_r2_preflight(overlay_sha256, count)
        r2.validate_existing(overlay_sha256)
        return result

    original.a40 = r2
    original._worker_command = _worker_command
    original._postrun_command = _postrun_command
    r2.preflight = preflight_and_retention_gate
    try:
        # This intentionally exercises original.launch -> r2.verify_package.
        original.launch(manifest_sha256, workers)
    finally:
        r2.preflight = prior_r2_preflight
        original._postrun_command = prior_postrun
        original._worker_command = prior_worker
        original.a40 = prior_a40


def main() -> None:
    parser = argparse.ArgumentParser(description="Operational R2 shared scheduler")
    parser.add_argument("mode", choices=("launch",))
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers != 4:
        raise RuntimeError("Operational R2 repair is frozen to four workers")
    with r2.execution_lock("stage2-stage2b-controller"):
        launch(args.manifest_sha256, args.workers)


if __name__ == "__main__":
    main()
