from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from research.remote import stage2b_a40 as contract


class Stage2BContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = contract.load_and_verify_frozen_seal()
        cls.rows = contract.build_job_rows()

    def test_01_frozen_seal_hash(self) -> None:
        self.assertEqual(contract.sha256_file(contract.FROZEN_MANIFEST), contract.FROZEN_SEAL_SHA256)

    def test_02_frozen_upstream_decisions(self) -> None:
        self.assertEqual(self.manifest["frozen_stage1_decision"], "LOG1P_TARGET_MAE")
        self.assertEqual(self.manifest["frozen_stage2_decision_unchanged"], "GGRU_K04_H032_D00")

    def test_03_exact_grid(self) -> None:
        observed = {(r["config_id"], int(r["hidden_size"]), float(r["dropout"])) for r in self.rows}
        self.assertEqual(observed, set(contract.CONFIGURATIONS))

    def test_04_exact_job_count_and_uniqueness(self) -> None:
        self.assertEqual(len(self.rows), 48)
        self.assertEqual(len({r["job_key"] for r in self.rows}), 48)

    def test_05_marginal_counts(self) -> None:
        self.assertEqual(set(Counter(r["config_id"] for r in self.rows).values()), {12})
        self.assertEqual(set(Counter(r["pseudo_target_city_id"] for r in self.rows).values()), {12})
        self.assertEqual(set(Counter(r["seed"] for r in self.rows).values()), {16})

    def test_06_final_targets_derived_from_split(self) -> None:
        roles = contract.split_roles()
        self.assertEqual(set(roles["final"]), set(contract.FINAL_TARGETS))

    def test_07_pseudo_target_excluded_from_sources(self) -> None:
        for row in self.rows:
            target = int(row["pseudo_target_city_id"])
            sources = tuple(map(int, row["source_city_ids"].split(";")))
            self.assertNotIn(target, sources)
            self.assertEqual(len(sources), 7)
            self.assertFalse(set(sources) & set(contract.FINAL_TARGETS))

    def test_08_deterministic_seed_mapping(self) -> None:
        for row in self.rows:
            target, seed = int(row["pseudo_target_city_id"]), int(row["seed"])
            self.assertEqual(int(row["source_city_schedule_seed"]), contract.deterministic_seed("source-city-schedule", target, seed))
            self.assertEqual(int(row["source_anchor_sampling_seed"]), contract.deterministic_seed("source-anchor-sampling", target, seed))

    def test_09_registered_source_budget_only(self) -> None:
        for row in self.rows:
            self.assertEqual(int(row["source_updates"]), 12000)
            self.assertEqual(int(row["batch_size_city_hours"]), 16)
            self.assertEqual(row["optimizer"], "AdamW")
            self.assertEqual(float(row["learning_rate"]), 1e-3)
            self.assertEqual(float(row["global_gradient_clip"]), 1.0)
            self.assertEqual(row["early_stopping"], False)
            self.assertEqual(int(row["fine_tuning_fits"]), 0)

    def test_10_prediction_keys_precommitted(self) -> None:
        for row in self.rows:
            key = json.loads(row["prediction_key_contract_json"])
            self.assertTrue(key["precommitted_before_evaluation_labels_opened"])
            self.assertEqual(key["interval"], {"start": contract.HD, "end": contract.HF, "semantics": "UTC left-closed right-open"})
            self.assertEqual(key["mask"], "coverage_observed_12h == true")

    def test_11_paths_are_isolated(self) -> None:
        for row in self.rows:
            for key in ("expected_checkpoint", "expected_prediction", "expected_prediction_manifest", "expected_job_result"):
                self.assertTrue(row[key].startswith("research/results/stage2b_vanilla_fairness_v2_1_a40/"))
        contract.assert_output_isolation(require_empty=True)

    def test_12_firewall_rejects_final_and_stage4_content(self) -> None:
        forbidden = [
            contract.ROOT / "processed/protocol_v2_1/final_target/final_panel.parquet",
            contract.ROOT / "research/results/stage4_dann_development_v2_1/result.json",
        ]
        for path in forbidden:
            with self.assertRaises(PermissionError):
                contract.assert_final_firewall_path(path)
        with self.assertRaises(PermissionError):
            contract.assert_final_firewall_path(contract.ROOT / "processed/protocol_v2_1/manifests/budget_manifest.json")

    def test_13_no_scientific_artifacts_or_training(self) -> None:
        self.assertEqual(contract.scientific_artifacts(), [])
        if contract.AUTHORIZATION.is_file():
            auth = contract.load_authorization()
            self.assertFalse(auth["training_started"])
            self.assertFalse(auth["local_execution_authorized"])

    def test_14_no_sys_path_or_vendored_dependency_injection(self) -> None:
        files = [
            contract.ROOT / "research/development/stage2b_vanilla_execution.py",
            contract.ROOT / "research/scripts/preflight_stage2b_a40.py",
            contract.ROOT / "research/scripts/run_stage2b_remote_a40.py",
            contract.ROOT / "research/scripts/validate_stage2b_remote_a40.py",
        ]
        for path in files:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("sys.path.insert", source)
            self.assertNotIn("neural_build/pydeps", source)
            self.assertNotIn("neural_build\\pydeps", source)

    def test_15_resume_rejects_incomplete_or_tampered_job(self) -> None:
        self.assertFalse(contract.validate_resume_job({"status": "running"}, self.rows[0]))

    def test_16_generated_map_matches_frozen_run_plan(self) -> None:
        contract._validate_against_frozen_run_plan(self.rows)

    def test_17_exact_feature_contract_and_no_graph_feature(self) -> None:
        source = (contract.ROOT / "research/development/stage2b_vanilla_execution.py").read_text(encoding="utf-8")
        for lag in range(1, 25):
            self.assertIn('f"hist_lag_{lag:03d}_log1p"', source)
            self.assertIn('f"hist_lag_{lag:03d}_observed"', source)
        for feature in ("week_lag_168_log1p", "static_bike_racks_log1p", "calendar_local_hour_sin", "calendar_utc_offset_div_12"):
            self.assertIn(feature, source)
        self.assertNotIn('"adjacency"', source)

    def test_18_resume_requires_identity_and_artifact_hashes(self) -> None:
        row = self.rows[0]
        job = {
            "status": "completed", "job_key": row["job_key"], "config_id": row["config_id"],
            "pseudo_target_city_id": int(row["pseudo_target_city_id"]), "seed": int(row["seed"]),
            "source_updates_completed": 12000, "fine_tuning_fits": 0, "evaluation_regime": "zero_shot",
            "checkpoint": {"path": "x", "sha256": "ok"},
            "prediction": {"path": "y", "sha256": "ok"},
            "prediction_manifest": {"path": "z", "sha256": "ok"},
        }
        with patch.object(Path, "is_file", return_value=True), patch.object(contract, "sha256_file", return_value="ok"):
            self.assertTrue(contract.validate_resume_job(job, row))
            job["source_updates_completed"] = 11999
            self.assertFalse(contract.validate_resume_job(job, row))

    def test_19_numpy_and_cuda_provenance_are_strict_remote_gates(self) -> None:
        self.assertEqual(contract.REQUIRED_A40_RUNTIME["numpy"], "1.26.4")
        self.assertEqual(contract.REQUIRED_A40_RUNTIME["torch"], "2.0.1+cu118")
        self.assertEqual(contract.REQUIRED_A40_RUNTIME["gpu"], "NVIDIA A40")
        source = (contract.ROOT / "research/scripts/preflight_stage2b_a40.py").read_text(encoding="utf-8")
        self.assertIn('"module_paths"', source)
        self.assertIn("torch.cuda.is_available()", source)
        self.assertIn("forbidden vendored pydeps", source)

    def test_20_r2_records_aborted_r1_first_forward_attempt(self) -> None:
        incident = contract.read_json(contract.INCIDENT_RECORD)
        self.assertEqual(
            incident["REMOTE_STAGE2B_R1_ATTEMPT"],
            "ABORTED_FIRST_FORWARD_MISSING_CUBLAS_WORKSPACE_CONFIG",
        )
        self.assertEqual(incident["optimizer_updates_completed"], 0)
        self.assertFalse(incident["scientific_fit_completed"])
        self.assertFalse(incident["scientific_training_started"])
        self.assertFalse(incident["evaluation_started"])
        self.assertFalse(incident["scientific_results_generated"])
        self.assertFalse(incident["final_target_labels_accessed"])
        r1 = contract.ROOT / "research/results/stage2b_execution_package_v2_1_r1/stage2b_execution_package_manifest.json"
        self.assertEqual(contract.sha256_file(r1), "32d9bbe8f9df15bce968a7876df19f66b2951f8bee054a796fa2a0dbbc7f92c4")


@unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "torch unavailable in local validation runtime")
class Stage2BSyntheticNeuralTests(unittest.TestCase):
    def test_21_vanilla_shapes_and_no_graph_argument(self) -> None:
        import torch
        from research.development.stage2b_vanilla_execution import vanilla_model_config
        from research.models.model_factory import build_model
        model = build_model(vanilla_model_config(32, 0.0))
        output = model(
            x_hist=torch.zeros(5, 24, 2), m_hist=torch.zeros(5, 24, dtype=torch.bool),
            x_week=torch.zeros(5, 2), x_static=torch.zeros(5, 2), x_calendar=torch.zeros(5, 6),
        )
        self.assertEqual(tuple(output.prediction.shape), (5,))
        self.assertFalse(any("graph" in name or "adjacency" in name for name, _ in model.named_parameters()))

    def test_22_dropout_is_active_in_training_and_off_in_eval(self) -> None:
        import torch
        from research.development.stage2b_vanilla_execution import vanilla_model_config
        from research.models.model_factory import build_model
        torch.manual_seed(11)
        model = build_model(vanilla_model_config(32, 0.1))
        args = dict(x_hist=torch.ones(32, 24, 2), m_hist=torch.ones(32, 24, dtype=torch.bool), x_week=torch.ones(32, 2), x_static=torch.ones(32, 2), x_calendar=torch.ones(32, 6))
        model.train(); first = model(**args).prediction; second = model(**args).prediction
        self.assertFalse(torch.equal(first, second))
        model.eval(); third = model(**args).prediction; fourth = model(**args).prediction
        self.assertTrue(torch.equal(third, fourth))

    def test_23_checkpoint_roundtrip_and_optimizer_contract(self) -> None:
        import torch
        from research.development.stage2b_vanilla_execution import optimizer_config, vanilla_model_config
        from research.models.model_factory import build_model
        from research.training.checkpointing import CheckpointMetadata, load_checkpoint, save_checkpoint
        from research.training.trainer import build_optimizer
        config = vanilla_model_config(32, 0.0); model = build_model(config); optimizer = build_optimizer(model, optimizer_config())
        metadata = CheckpointMetadata(config.to_dict(), vars(optimizer_config()), None, "2.1", "f" * 64, "r" * 64, 17, 0, 12000, "log1p_target", "c" * 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.pt"; save_checkpoint(path, model, optimizer, metadata)
            restored = build_model(config); restored_optimizer = build_optimizer(restored, optimizer_config())
            loaded = load_checkpoint(path, restored, restored_optimizer, expected_feature_schema_sha256="f" * 64, expected_city_roster_sha256="r" * 64)
            self.assertEqual(loaded["step"], 12000)
            for left, right in zip(model.parameters(), restored.parameters(), strict=True):
                self.assertTrue(torch.equal(left, right))

    def test_24_cuda_is_the_only_runner_device(self) -> None:
        source = (contract.ROOT / "research/scripts/run_stage2b_remote_a40.py").read_text(encoding="utf-8")
        self.assertIn('choices=("cuda",)', source)
        self.assertNotIn('choices=("cuda", "cpu")', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
