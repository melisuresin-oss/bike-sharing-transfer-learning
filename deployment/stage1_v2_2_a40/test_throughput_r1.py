"""Non-scientific execution-mechanics tests; never launches a worker."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tmp/neural_build/pydeps"), str(ROOT)]

from research.stage1_v2_2 import a40 as base
from research.stage1_v2_2 import a40_parallel as parallel


class ThroughputAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.jobs = base.c.read(base.JOBMAP)["jobs"]
        cls.controller_source = (ROOT / "research/stage1_v2_2/a40_parallel.py").read_text(encoding="utf-8")
        cls.worker_source = (ROOT / "research/stage1_v2_2/a40.py").read_text(encoding="utf-8")
        cls.wrapper_source = (ROOT / "deployment/stage1_v2_2_a40/run_a40_throughput_r1.ps1").read_text(encoding="utf-8")

    def test_01_operational_worker_limit_is_four(self) -> None:
        self.assertEqual(parallel.DEFAULT_WORKERS, 4)
        self.assertEqual(parallel.MAX_WORKERS, 4)
        self.assertEqual(parallel.THREADS_PER_WORKER, 2)

    def test_02_immutable_scientific_map_is_unchanged(self) -> None:
        self.assertEqual(base.c.sha(base.JOBMAP), base.JOBMAP_SHA)
        self.assertEqual(base.JOBMAP_SHA, "4709c222b799a9a2d4ef209ce80ae5ee859b5a818db5c1cb97df2954ffce3937")
        self.assertEqual(len(self.jobs), 48)
        self.assertEqual(sum(job["phase"] == "source" for job in self.jobs), 24)
        self.assertEqual(sum(job["phase"] == "adaptation" for job in self.jobs), 24)

    def test_03_every_fit_gets_a_fresh_python_worker_command(self) -> None:
        commands = [parallel._worker_command(job["id"], "m" * 64, "preflight.json") for job in self.jobs]
        self.assertEqual(len(commands), 48)
        self.assertEqual(len({tuple(command) for command in commands}), 48)
        for job, command in zip(self.jobs, commands, strict=True):
            self.assertIn("research.stage1_v2_2.a40", command)
            self.assertIn("job", command)
            self.assertEqual(command[-1], job["id"])

    def test_04_controller_uses_popen_not_reused_process_pool(self) -> None:
        tree = ast.parse(self.controller_source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        names = {ast.unparse(call.func) for call in calls}
        self.assertIn("subprocess.Popen", names)
        self.assertNotIn("ProcessPoolExecutor", self.controller_source)
        self.assertIn("CREATE_NO_WINDOW", self.controller_source)
        self.assertIn("CREATE_NEW_PROCESS_GROUP", self.controller_source)

    def test_05_source_wave_precedes_adaptation_wave(self) -> None:
        launch = next(node for node in ast.parse(self.controller_source).body if isinstance(node, ast.FunctionDef) and node.name == "launch")
        text = ast.get_source_segment(self.controller_source, launch) or ""
        first = text.index("source_jobs, phase=\"source\"")
        second = text.index("adaptation_jobs, phase=\"adaptation\"")
        self.assertLess(first, second)
        for job in self.jobs:
            if job["phase"] == "adaptation":
                self.assertIn(job["depends_on"], {item["id"] for item in self.jobs if item["phase"] == "source"})

    def test_06_job_namespaces_and_locks_are_collision_free(self) -> None:
        identifiers = [job["id"] for job in self.jobs]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertIn('execution_lock(a.job_id)', self.worker_source)
        self.assertIn('OUT+"attempts/"+job_id+"/"+uuid.uuid4().hex+"/"', self.worker_source)
        self.assertIn('OUT + "completed/" + job["id"] + ".json"', self.worker_source)

    def test_07_optimizer_and_rng_state_are_worker_local(self) -> None:
        self.assertIn("Fresh optimizer from update zero", self.worker_source)
        self.assertIn("np.random.default_rng", self.worker_source)
        self.assertIn("c.set_seed(task[\"seed\"])", self.worker_source)
        self.assertIn("partial_checkpoint_resumed", self.worker_source)
        self.assertIn("local_cpu_checkpoint_loaded", self.worker_source)

    def test_08_atomic_completion_and_controller_status_writes(self) -> None:
        self.assertIn('temp = p.with_name(p.name + ".pending-" + uuid.uuid4().hex)', self.worker_source)
        self.assertIn("os.fsync", self.worker_source)
        self.assertIn("temp.rename(p)", self.worker_source)
        self.assertIn('write(complete,record)', self.worker_source)
        self.assertIn('.exit.json", status)', self.controller_source)

    def test_09_detached_wrapper_and_file_redirection(self) -> None:
        self.assertIn("Start-Process", self.wrapper_source)
        self.assertIn("-WindowStyle Hidden", self.wrapper_source)
        self.assertIn("-RedirectStandardOutput", self.wrapper_source)
        self.assertIn("-RedirectStandardError", self.wrapper_source)
        self.assertIn("-PassThru", self.wrapper_source)
        self.assertNotIn("-Wait", self.wrapper_source)

    def test_10_worker_logs_and_concurrency_are_runtime_provenance(self) -> None:
        self.assertIn("runtime_provenance.json", self.controller_source)
        self.assertIn('"worker_count_is_scientific_hyperparameter": False', self.controller_source)
        self.assertIn('.stdout.log"', self.controller_source)
        self.assertIn('.stderr.log"', self.controller_source)
        self.assertIn("operational_worker_count", self.controller_source)

    def test_11_no_scientific_parameter_override_in_parallel_controller(self) -> None:
        forbidden_assignments = (
            "BATCH_SIZE =", "SOURCE_UPDATES =", "ADAPTATION_UPDATES =",
            "LEARNING_RATE =", "WEIGHT_DECAY =", "SEEDS =", "CANDIDATES =",
        )
        for assignment in forbidden_assignments:
            self.assertNotIn(assignment, self.controller_source)
        self.assertEqual(base.c.sha(base.JOBMAP), base.JOBMAP_SHA)

    def test_12_resource_basis_supports_initial_four_workers(self) -> None:
        payload = parallel._cache_payload_bytes()
        self.assertEqual(payload, 670747102)
        self.assertLess(payload * 4, 3 * 1024**3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
