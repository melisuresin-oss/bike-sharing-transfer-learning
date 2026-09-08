"""Zero-science integration tests for the operational R2 controller interface."""
from __future__ import annotations

import ast
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tmp/neural_build/pydeps"), str(ROOT)]

from research.stage2_v2_2 import a40_r1 as r1
from research.stage2_v2_2 import a40_r2 as r2
from research.stage2_v2_2 import controller as original
from research.stage2_v2_2 import controller_r2
from research.stage2_v2_2 import core as c


class ControllerR2IntegrationTests(unittest.TestCase):
    def test_complete_original_controller_interface_is_available(self):
        required = (
            "verify_package", "preflight", "write_json_atomic", "utc",
            "THREADS_PER_WORKER", "entry", "output_path", "execution_lock",
            "MAX_WORKERS", "DEFAULT_WORKERS",
        )
        self.assertEqual([name for name in required if not hasattr(r2, name)], [])
        self.assertIs(r2.canonicalize_metadata, r1.canonicalize_metadata)
        self.assertIs(r2.validate_completed, r1.validate_completed)

    def _exercise_controller(self, *, jobs: int, failed_first: bool):
        writes = []
        commands = []
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            queue = []
            for index in range(jobs):
                stage = "stage2" if index % 2 == 0 else "stage2b"
                stage_number = 2 if stage == "stage2" else "2B"
                queue.append((stage, {"job_id": f"fixture-{index}", "stage": stage_number}))

            def fake_path(relative: str) -> Path:
                return temporary / relative

            def fake_entry(relative: str):
                return {"path": relative, "sha256": "e" * 64, "bytes": 0}

            def fake_write(relative: str, value):
                writes.append((relative, value))
                path = fake_path(relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")
                return fake_entry(relative)

            class FakeProcess:
                next_pid = 9000

                def __init__(self, command, **kwargs):
                    self.command = command
                    commands.append(command)
                    self.pid = FakeProcess.next_pid
                    FakeProcess.next_pid += 1
                    job_id = command[command.index("--job-id") + 1]
                    stage = command[command.index("--stage") + 1]
                    number = int(job_id.split("-")[-1])
                    self.returncode = 1 if failed_first and number == 0 else 0
                    if self.returncode == 0:
                        root = c.STAGE2_OUT if stage == "stage2" else c.STAGE2B_OUT
                        completed = fake_path(root + "completed/" + job_id + ".json")
                        completed.parent.mkdir(parents=True, exist_ok=True)
                        completed.write_text("fixture", encoding="utf-8")

                def poll(self):
                    return self.returncode

            package = Mock(return_value={})
            preflight = Mock(return_value={
                "record": {"resource_check": {}},
                "artifact": {"path": c.JOINT_OUT + "preflight/fixture.json",
                             "sha256": "p" * 64, "bytes": 1},
            })
            retained = Mock(return_value={"status": "PASS", "stage2_retained": 3,
                                          "stage2b_retained": 1})
            with ExitStack() as stack:
                stack.enter_context(patch.object(r2, "verify_package", package))
                stack.enter_context(patch.object(r2, "preflight", preflight))
                stack.enter_context(patch.object(r2, "validate_existing", retained))
                stack.enter_context(patch.object(r2, "write_json_atomic", side_effect=fake_write))
                stack.enter_context(patch.object(r2, "output_path", side_effect=fake_path))
                stack.enter_context(patch.object(r2, "entry", side_effect=fake_entry))
                stack.enter_context(patch.object(original, "_interleaved_jobs", return_value=queue))
                stack.enter_context(patch.object(original.c, "repository_path", side_effect=fake_path))
                stack.enter_context(patch.object(original.c, "sha256_file", return_value="h" * 64))
                popen = stack.enter_context(patch.object(subprocess, "Popen", side_effect=FakeProcess))
                run = stack.enter_context(patch.object(subprocess, "run", return_value=Mock(returncode=0)))
                error = None
                try:
                    controller_r2.launch("a" * 64, 4)
                except RuntimeError as caught:
                    error = caught
            return writes, commands, package, preflight, retained, popen, run, error

    def test_exact_previous_startup_path_reaches_dispatch_and_success_exit(self):
        writes, commands, package, preflight, retained, popen, run, error = \
            self._exercise_controller(jobs=1, failed_first=False)
        self.assertIsNone(error)
        self.assertGreaterEqual(package.call_count, 2)  # R2 entry plus original.launch.
        preflight.assert_called_once_with("a" * 64, 4)
        retained.assert_called_once_with("a" * 64)
        self.assertEqual(popen.call_count, 1)
        self.assertIn("research.stage2_v2_2.a40_r2", commands[0])
        exits = [value for path, value in writes if path.endswith(".exit.json")]
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0]["status"], "EXITED_ZERO_WITH_COMPLETION")
        self.assertEqual(exits[0]["exit_code"], 0)
        self.assertTrue(exits[0]["completed_record_present"])
        self.assertEqual(run.call_count, 4)  # two validators, then two freezes.

    def test_genuine_worker_failure_stops_before_fifth_dispatch(self):
        writes, commands, package, preflight, retained, popen, run, error = \
            self._exercise_controller(jobs=5, failed_first=True)
        self.assertIsNotNone(error)
        self.assertIn("Worker failure", str(error))
        self.assertEqual(popen.call_count, 4)
        self.assertNotIn("fixture-4", " ".join(" ".join(command) for command in commands))
        self.assertEqual(run.call_count, 0)
        self.assertTrue(any(path.endswith("controller_failed.json") for path, _ in writes))

    def test_existing_completion_skip_uses_corrected_r1_validation(self):
        job = {"stage": 2, "job_id": "retained"}
        with TemporaryDirectory() as directory:
            completed = Path(directory) / "completed.json"
            completed.write_text("fixture", encoding="utf-8")
            with patch.object(r1, "verify_overlay"), \
                 patch.object(r1, "_job", return_value=job), \
                 patch.object(r1.c, "repository_path", return_value=completed), \
                 patch.object(r1, "validate_completed", return_value={"status": "PASS"}) as validate, \
                 patch.object(r1.base, "run_job") as scientific_run:
                r1.run_job("stage2", "retained", r2.R1_OVERLAY_MANIFEST_SHA256,
                           "preflight.json", 4)
            validate.assert_called_once_with(job, r1.BASE_BUNDLE_MANIFEST_SHA256,
                                             recompute_metrics=False)
            scientific_run.assert_not_called()

    def test_r2_worker_delegates_to_r1_corrected_path(self):
        with patch.object(r2, "verify_package"), patch.object(r2.r1, "run_job") as run:
            r2.run_job("stage2", "fixture", "a" * 64, "preflight.json", 4)
        run.assert_called_once_with("stage2", "fixture", r2.R1_OVERLAY_MANIFEST_SHA256,
                                    "preflight.json", 4)

    def test_base_r1_and_scientific_hashes_remain_unchanged(self):
        expected = {
            "research/stage2_v2_2/a40.py": r1.ORIGINAL_A40_SHA256,
            "research/stage2_v2_2/controller.py": r1.ORIGINAL_CONTROLLER_SHA256,
            "research/stage2_v2_2/a40_r1.py": r2.R1_A40_SHA256,
            "research/stage2_v2_2/controller_r1.py": r2.R1_CONTROLLER_SHA256,
            c.STAGE2_CONTRACT: r1.STAGE2_CONTRACT_SHA256,
            c.STAGE2B_CONTRACT: r1.STAGE2B_CONTRACT_SHA256,
            c.STAGE2_JOB_MAP: r1.STAGE2_JOB_MAP_SHA256,
            c.STAGE2B_JOB_MAP: r1.STAGE2B_JOB_MAP_SHA256,
        }
        for relative, digest in expected.items():
            self.assertEqual(c.sha256_file(relative), digest, relative)

    def test_python310_and_powershell_boundaries(self):
        for relative in ("research/stage2_v2_2/a40_r2.py",
                         "research/stage2_v2_2/controller_r2.py"):
            ast.parse((ROOT / relative).read_text(encoding="utf-8"), feature_version=(3, 10))
        wrapper = (ROOT / "deployment/stage2_v2_2_a40/run_stage2_stage2b_operational_r2.ps1").read_text(encoding="utf-8")
        self.assertIn("research.stage2_v2_2.controller_r2", wrapper)
        self.assertIn("-WindowStyle Hidden", wrapper)
        self.assertIn("'--workers','4'", wrapper)

    def test_r2_provenance_records_zero_local_science_and_firewall(self):
        provenance = c.read_json("deployment/stage2_v2_2_a40/OPERATIONAL_R2_INCIDENT_PROVENANCE.json")
        self.assertEqual(provenance["classification"],
                         "OPERATIONAL_R2_CONTROLLER_INTERFACE_FIX_BEFORE_NEW_JOB_DISPATCH")
        self.assertFalse(provenance["scientific_training_performed_locally"])
        self.assertFalse(provenance["scientific_evaluation_performed_locally"])
        self.assertFalse(provenance["final_target_labels_accessed"])
        self.assertFalse(provenance["stage3_started"])
        self.assertFalse(provenance["stage4_started"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
