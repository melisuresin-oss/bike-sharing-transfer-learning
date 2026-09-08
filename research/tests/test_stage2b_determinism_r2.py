from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from research.remote import stage2b_a40 as contract


class Stage2BR2StaticDeterminismTests(unittest.TestCase):
    def test_01_registered_workspace_value(self) -> None:
        self.assertEqual(contract.CUBLAS_WORKSPACE_CONFIG, ":4096:8")

    def test_02_workspace_is_set_automatically(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            self.assertEqual(contract.establish_cublas_workspace_config(), ":4096:8")
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")

    def test_03_conflicting_workspace_is_rejected(self) -> None:
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "conflicts with the sealed"):
                contract.establish_cublas_workspace_config()

    def test_04_runner_establishes_workspace_before_torch_import(self) -> None:
        path = contract.ROOT / "research/scripts/run_stage2b_remote_a40.py"
        source = path.read_text(encoding="utf-8")
        self.assertLess(source.index("establish_cublas_workspace_config()"), source.index("\n    import torch"))
        self.assertNotIn("set CUBLAS_WORKSPACE_CONFIG", source)

    def test_05_implementation_establishes_workspace_before_torch_import(self) -> None:
        path = contract.ROOT / "research/development/stage2b_vanilla_execution.py"
        source = path.read_text(encoding="utf-8")
        self.assertLess(source.index("establish_cublas_workspace_config()"), source.index("import torch"))

    def test_06_strict_preflight_calls_real_cuda_fixture(self) -> None:
        path = contract.ROOT / "research/scripts/preflight_stage2b_a40.py"
        source = path.read_text(encoding="utf-8")
        self.assertIn("cuda_determinism_fixture(device)", source)
        self.assertIn('configure_deterministic_device(17, "cuda")', source)
        self.assertNotIn("compute_metrics", source)
        self.assertNotIn("run_job(", source)

    def test_07_runner_allows_only_cuda(self) -> None:
        source = (contract.ROOT / "research/scripts/run_stage2b_remote_a40.py").read_text(encoding="utf-8")
        self.assertIn('choices=("cuda",)', source)

    def test_08_no_scientific_contract_change(self) -> None:
        rows = contract.build_job_rows()
        self.assertEqual(len(rows), 48)
        self.assertEqual({int(row["source_updates"]) for row in rows}, {12000})
        self.assertEqual({int(row["fine_tuning_fits"]) for row in rows}, {0})
        self.assertEqual({row["evaluation_regime"] for row in rows}, {"zero_shot"})

    def test_09_r0_and_r1_packages_are_preserved(self) -> None:
        expected = {
            "stage2b_execution_package_v2_1": "6cb1aaeadfe2648709a6746b3b54928cfaa43eaf2ad5e7a72beb1c7f56d57711",
            "stage2b_execution_package_v2_1_r1": "32d9bbe8f9df15bce968a7876df19f66b2951f8bee054a796fa2a0dbbc7f92c4",
        }
        for folder, digest in expected.items():
            path = contract.ROOT / "research/results" / folder / "stage2b_execution_package_manifest.json"
            self.assertEqual(contract.sha256_file(path), digest)

    def test_10_run_job_configures_before_model_build(self) -> None:
        path = contract.ROOT / "research/development/stage2b_vanilla_execution.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        run_job = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_job")
        text = ast.get_source_segment(path.read_text(encoding="utf-8"), run_job) or ""
        self.assertLess(text.index("configure_deterministic_device"), text.index("build_model"))

    def test_11_r2_matches_successful_stage2_flag_contract(self) -> None:
        canonical = (contract.ROOT / "research/development/stage2_graphgru_selection.py").read_text(encoding="utf-8")
        current = (contract.ROOT / "research/development/stage2b_vanilla_execution.py").read_text(encoding="utf-8")
        self.assertIn('CUBLAS_WORKSPACE_CONFIG", ":4096:8"', canonical)
        self.assertEqual(contract.CUBLAS_WORKSPACE_CONFIG, ":4096:8")
        for token in (
            "torch.backends.cuda.matmul.allow_tf32 = False",
            "torch.backends.cudnn.benchmark = False",
            "torch.backends.cudnn.deterministic = True",
            "torch.backends.cudnn.allow_tf32 = False",
            "set_seed(seed, deterministic=True)",
        ):
            self.assertIn(token, canonical)
            self.assertIn(token, current)


@unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "torch unavailable")
class Stage2BR2RuntimeDeterminismTests(unittest.TestCase):
    def test_12_cpu_probe_records_exact_flags_and_dtype_independent_policy(self) -> None:
        from research.development.stage2b_vanilla_execution import configure_deterministic_device
        device, settings = configure_deterministic_device(17, "cpu")
        self.assertEqual(device.type, "cpu")
        self.assertTrue(settings["deterministic_algorithms"])
        self.assertEqual(settings["cublas_workspace_config"], ":4096:8")
        self.assertFalse(settings["cuda_matmul_allow_tf32"])
        self.assertFalse(settings["cudnn_benchmark"])
        self.assertTrue(settings["cudnn_deterministic"])
        self.assertFalse(settings["cudnn_allow_tf32"])

    def test_13_forward_failure_precedes_optimizer_step(self) -> None:
        import torch
        from research.training.trainer import NeuralTrainer, OptimizerConfig, TrainerConfig

        class FailingModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))

            def forward(self, **_: object):
                raise RuntimeError("synthetic first-forward failure")

        model = FailingModel()
        trainer = NeuralTrainer(
            model,
            OptimizerConfig(name="AdamW", learning_rate=1e-3),
            TrainerConfig(seed=17, output_mode="log1p_target", checkpoint_mode="final"),
        )
        with self.assertRaisesRegex(RuntimeError, "synthetic first-forward"):
            trainer.train_step({"model_inputs": {}, "target": torch.ones(1), "mask": torch.ones(1, dtype=torch.bool)})
        self.assertEqual(trainer.step, 0)
        self.assertEqual(trainer.optimizer.state, {})

    @unittest.skipUnless(__import__("torch").cuda.is_available(), "strict CUDA fixture runs on UNIVERSITY_A40")
    def test_14_actual_cuda_matmul_fixture(self) -> None:
        from research.development.stage2b_vanilla_execution import configure_deterministic_device, cuda_determinism_fixture
        device, _ = configure_deterministic_device(17, "cuda")
        result = cuda_determinism_fixture(device)
        self.assertTrue(result["passed"])
        self.assertTrue(result["repeated_outputs_bitwise_equal"])
        self.assertFalse(result["training_or_evaluation_performed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
