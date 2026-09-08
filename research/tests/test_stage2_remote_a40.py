from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from research.development.stage1_scale_loss import FINAL_TARGETS, SOURCE_UPDATES
from research.development.stage2_graphgru_selection import (
    Stage2ArtifactRegistry,
    assert_batch_device,
    registered_stage2_grid,
    resolve_torch_device,
    stage2_model_config,
)
from research.models.common import NeuralModelConfig
from research.models.model_factory import build_model
from research.remote.cuda_fixture import fixed_graph_batch, run_fixed_training_fixture
from research.remote.stage2_a40 import (
    LOCAL_ABORTED_OUTPUT,
    REMOTE_OUTPUT,
    assert_remote_output_isolation,
    build_job_rows,
    load_frozen_stage2_plan,
    sha256_file,
)
from research.scripts.run_stage2_graphgru_selection import (
    ROOT,
    _artifact_inside_output,
    _read_completed_job,
    aggregate_results,
)
from research.training.checkpointing import (
    CheckpointMetadata,
    load_checkpoint,
    neural_code_hash,
    save_checkpoint,
)
from research.training.trainer import OptimizerConfig, build_optimizer


class Stage2RemoteA40Tests(unittest.TestCase):
    def test_cpu_device_path_and_tensor_colocation(self) -> None:
        result = run_fixed_training_fixture("cpu", updates=1)
        self.assertEqual(result["resolved_device"], "cpu")
        self.assertEqual(result["parameter_devices"], ["cpu"])
        self.assertTrue(np.isfinite(result["prediction"]).all())
        batch = fixed_graph_batch(torch.device("cpu"))
        assert_batch_device(batch, torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_device_path_is_fixed_seed_reproducible(self) -> None:
        first = run_fixed_training_fixture("cuda", seed=17, updates=4)
        second = run_fixed_training_fixture("cuda", seed=17, updates=4)
        self.assertEqual(first["resolved_device"], "cuda:0")
        self.assertEqual(first["parameter_devices"], ["cuda:0"])
        self.assertEqual(first["state_sha256"], second["state_sha256"])
        np.testing.assert_array_equal(first["losses"], second["losses"])
        np.testing.assert_array_equal(first["prediction"], second["prediction"])

    def test_model_graph_features_masks_and_targets_share_device(self) -> None:
        device = resolve_torch_device("cuda" if torch.cuda.is_available() else "cpu")
        config = NeuralModelConfig(
            model_type="graph_gru",
            hidden_size=8,
            dropout=0.0,
            output_mode="log1p_target",
        )
        model = build_model(config).to(device)
        batch = fixed_graph_batch(device)
        assert_batch_device(batch, device)
        self.assertEqual({parameter.device for parameter in model.parameters()}, {device})
        output = model(**batch["model_inputs"])
        self.assertEqual(output.prediction.device, device)

    def test_checkpoint_is_portable_between_cpu_and_available_target(self) -> None:
        config = NeuralModelConfig(
            model_type="graph_gru",
            hidden_size=8,
            dropout=0.0,
            output_mode="log1p_target",
        )
        cpu_model = build_model(config)
        optimizer_config = OptimizerConfig()
        cpu_optimizer = build_optimizer(cpu_model, optimizer_config)
        metadata = CheckpointMetadata(
            architecture_config=config.to_dict(),
            optimizer_config=asdict(optimizer_config),
            graph_config={"adjacency_hash_sha256": "graph"},
            protocol_version="2.1",
            feature_schema_sha256="feature",
            city_roster_sha256="roster",
            seed=17,
            epoch=0,
            step=0,
            scale_loss_mode="log1p_target",
            code_sha256=neural_code_hash(),
        )
        target = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        with tempfile.TemporaryDirectory() as directory:
            cpu_path = Path(directory) / "cpu.pt"
            save_checkpoint(cpu_path, cpu_model, cpu_optimizer, metadata)
            target_model = build_model(config).to(target)
            target_optimizer = build_optimizer(target_model, optimizer_config)
            load_checkpoint(
                cpu_path,
                target_model,
                target_optimizer,
                expected_feature_schema_sha256="feature",
                expected_city_roster_sha256="roster",
                expected_graph_sha256="graph",
            )
            self.assertEqual({item.device for item in target_model.parameters()}, {target})

            target_path = Path(directory) / "target.pt"
            save_checkpoint(target_path, target_model, target_optimizer, metadata)
            restored_cpu = build_model(config)
            load_checkpoint(
                target_path,
                restored_cpu,
                expected_feature_schema_sha256="feature",
                expected_city_roster_sha256="roster",
                expected_graph_sha256="graph",
            )
            self.assertEqual(
                {item.device for item in restored_cpu.parameters()}, {torch.device("cpu")}
            )

    def test_frozen_job_map_is_unique_144_of_144(self) -> None:
        rows = build_job_rows(load_frozen_stage2_plan())
        self.assertEqual(len(rows), 144)
        self.assertEqual({row["job_id"] for row in rows}, set(range(144)))
        self.assertEqual(len({row["job_key"] for row in rows}), 144)

    def test_remote_output_is_isolated_and_local_output_is_rejected(self) -> None:
        assert_remote_output_isolation(
            REMOTE_OUTPUT, require_no_scientific_results=True
        )
        with self.assertRaises(RuntimeError):
            assert_remote_output_isolation(
                LOCAL_ABORTED_OUTPUT, require_no_scientific_results=False
            )

    def test_local_aborted_artifact_cannot_enter_remote_validation(self) -> None:
        local_checkpoint = next(
            LOCAL_ABORTED_OUTPUT.glob("stage2_checkpoints/**/*.pt")
        )
        with self.assertRaises(RuntimeError):
            _artifact_inside_output(
                str(local_checkpoint), REMOTE_OUTPUT, "source checkpoint"
            )

    def test_local_result_cannot_enter_remote_aggregation(self) -> None:
        grid = registered_stage2_grid(ROOT / "DEVELOPMENT_SELECTION_PROTOCOL.md")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                aggregate_results(
                    [{"execution_environment": "LOCAL_CPU"}],
                    grid,
                    Path(directory),
                    {},
                    expected_execution_environment="UNIVERSITY_A40",
                )

    def test_resume_accepts_only_complete_hash_validated_remote_job(self) -> None:
        config = registered_stage2_grid(ROOT / "DEVELOPMENT_SELECTION_PROTOCOL.md")[0]
        task = {
            "configuration": config.as_dict(),
            "target_id": 532,
            "seed": 17,
            "execution_environment": "UNIVERSITY_A40",
            "device": "cuda",
            "task_signature_sha256": "task-signature",
        }
        job_id = "ggru_k04_h032_d00_target-532_seed-17"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            checkpoint = output / "stage2_checkpoints" / config.config_id / "source.pt"
            prediction = output / "stage2_predictions" / config.config_id / "prediction.npz"
            prediction_manifest_path = output / "stage2_prediction_manifests" / "job.json"
            job_path = output / "jobs" / f"{job_id}.json"
            for path, content in ((checkpoint, b"checkpoint"), (prediction, b"prediction")):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            evaluation = {
                "config_id": config.config_id,
                "pseudo_target_city_id": 532,
                "seed": 17,
                "regime": "zero_shot",
                "n": 3,
                "key_sha256": "keys",
                "target_sha256": "targets",
                "prediction_logical_sha256": "predictions",
                "prediction_artifact": str(prediction),
                "prediction_artifact_sha256": sha256_file(prediction),
                "prediction_artifact_bytes": prediction.stat().st_size,
                "predictions_finite": True,
                "predictions_nonnegative": True,
                "prediction_keys_unique": True,
                "prediction_keys_complete": True,
            }
            prediction_manifest = {
                key: evaluation[key]
                for key in (
                    "config_id",
                    "pseudo_target_city_id",
                    "seed",
                    "regime",
                    "n",
                    "key_sha256",
                    "target_sha256",
                    "prediction_logical_sha256",
                    "prediction_artifact_sha256",
                )
            }
            prediction_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            prediction_manifest_path.write_text(
                json.dumps(prediction_manifest), encoding="utf-8"
            )
            value = {
                "status": "completed",
                "job_id": job_id,
                "task_signature_sha256": "task-signature",
                "configuration": config.as_dict(),
                "pseudo_target_city_id": 532,
                "seed": 17,
                "execution_environment": "UNIVERSITY_A40",
                "device_requested": "cuda",
                "device_resolved": "cuda:0",
                "source_updates": SOURCE_UPDATES,
                "fine_tune_updates": 0,
                "regimes": ["zero_shot"],
                "source_city_ids": [467, 476, 619, 626, 658, 669, 734],
                "source_checkpoint": str(checkpoint),
                "source_checkpoint_sha256": sha256_file(checkpoint),
                "source_checkpoint_bytes": checkpoint.stat().st_size,
                "source_checkpoint_roundtrip_max_abs_difference": 0.0,
                "evaluation": evaluation,
                "prediction_manifest": str(prediction_manifest_path),
                "prediction_manifest_sha256": sha256_file(prediction_manifest_path),
            }
            job_path.parent.mkdir(parents=True, exist_ok=True)
            job_path.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(_read_completed_job(job_path, task, output), value)
            checkpoint.write_bytes(b"tampered")
            with self.assertRaises(RuntimeError):
                _read_completed_job(job_path, task, output)

    def test_final_label_firewall_remains_closed(self) -> None:
        registry = Stage2ArtifactRegistry()
        for city_id in FINAL_TARGETS:
            with self.assertRaises(RuntimeError):
                registry.station_manifest(city_id)


if __name__ == "__main__":
    unittest.main()
