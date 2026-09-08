import inspect
import unittest
from pathlib import Path

import torch

from research.data.manifests import ProtocolArtifactRegistry
from research.data.window_dataset import FinalStationInferenceLoader
from research.models.common import NeuralModelConfig
from research.models.model_factory import build_model
from research.training.trainer import NeuralTrainer, OptimizerConfig, TrainerConfig


ROOT = Path(__file__).resolve().parents[2]


class NeuralFinalLabelFirewallTests(unittest.TestCase):
    def test_neural_modules_do_not_name_or_import_sealed_labels(self):
        forbidden = ("SEALED_final_evaluation_labels", "final_labels/")
        for folder in ("models", "training", "graphs"):
            for path in (ROOT / "research" / folder).rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                for token in forbidden:
                    self.assertNotIn(token, text, path)

    def test_final_inference_has_x_and_keys_but_no_y(self):
        loader = FinalStationInferenceLoader(ProtocolArtifactRegistry(), 195)
        batch = loader.fetch(0, 4)
        self.assertIsNone(batch.y)
        self.assertEqual(batch.x_hist.shape, (4, 24, 2))
        self.assertEqual(len(batch.station_ids), 4)

    def test_trainer_rejects_label_free_batch(self):
        model = build_model(
            NeuralModelConfig(model_type="vanilla_gru", hidden_size=4, dropout=0.0)
        )
        trainer = NeuralTrainer(
            model,
            OptimizerConfig(),
            TrainerConfig(seed=1, output_mode="raw_count"),
        )
        x_hist = torch.zeros(2, 24, 2)
        batch = {
            "model_inputs": {
                "x_hist": x_hist,
                "m_hist": torch.zeros(2, 24, dtype=torch.bool),
                "x_week": torch.zeros(2, 2),
                "x_static": torch.zeros(2, 2),
                "x_calendar": torch.zeros(2, 6),
            },
            "target": None,
            "mask": torch.zeros(2, dtype=torch.bool),
        }
        with self.assertRaises(ValueError):
            trainer.train_step(batch)


if __name__ == "__main__":
    unittest.main()
