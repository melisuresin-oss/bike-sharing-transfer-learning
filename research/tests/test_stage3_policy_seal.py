from __future__ import annotations

import ast
import json
import unittest
from dataclasses import replace
from pathlib import Path

from research.governance.stage3_policy import (
    AUTHORITATIVE_INPUT_SHA256,
    FROZEN_UPSTREAM_SHA256,
    REGISTERED_BACKBONE,
    REGISTERED_POLICY,
    ROOT,
    SEAL_ROOT,
    Stage3FineTuningPolicy,
    policy_dict,
    sha256_file,
    validate_policy,
    verify_authoritative_inputs,
    verify_frozen_upstream,
    verify_manifest,
    verify_seal_bundle,
)


class Stage3PolicySealTests(unittest.TestCase):
    def test_stage1_and_stage2_decisions_are_hash_pinned_and_read_only(self) -> None:
        before = {
            path: (ROOT / path).read_bytes() for path in FROZEN_UPSTREAM_SHA256
        }
        verify_frozen_upstream()
        after = {path: (ROOT / path).read_bytes() for path in FROZEN_UPSTREAM_SHA256}
        self.assertEqual(before, after)
        for path, expected in FROZEN_UPSTREAM_SHA256.items():
            self.assertEqual(sha256_file(ROOT / path), expected)

    def test_selected_stage2_backbone_is_exact(self) -> None:
        self.assertEqual(
            REGISTERED_BACKBONE,
            {
                "config_id": "GGRU_K04_H032_D00",
                "model_type": "Graph-GRU",
                "graph_k": 4,
                "hidden_size": 32,
                "dropout": 0.0,
                "scale_loss_formulation": "LOG1P_TARGET_MAE",
            },
        )
        verify_frozen_upstream()

    def test_only_registered_update_budgets_are_accepted(self) -> None:
        self.assertEqual(REGISTERED_POLICY.update_budgets, (0, 100, 300, 600, 1200))
        validate_policy(REGISTERED_POLICY)
        with self.assertRaises(ValueError):
            validate_policy(replace(REGISTERED_POLICY, update_budgets=(0, 100, 301, 600, 1200)))

    def test_zero_update_budget_is_not_a_fine_tuning_fit(self) -> None:
        self.assertFalse(REGISTERED_POLICY.zero_budget_is_fine_tuning_fit)
        with self.assertRaises(ValueError):
            validate_policy(replace(REGISTERED_POLICY, zero_budget_is_fine_tuning_fit=True))

    def test_optimizer_state_reset_is_mandatory(self) -> None:
        self.assertTrue(REGISTERED_POLICY.reset_adamw_optimizer_state_before_adaptation)
        with self.assertRaises(ValueError):
            validate_policy(
                replace(
                    REGISTERED_POLICY,
                    reset_adamw_optimizer_state_before_adaptation=False,
                )
            )

    def test_early_stopping_is_impossible(self) -> None:
        self.assertFalse(REGISTERED_POLICY.early_stopping)
        with self.assertRaises(ValueError):
            validate_policy(replace(REGISTERED_POLICY, early_stopping=True))

    def test_authoritative_inputs_are_hash_pinned(self) -> None:
        verify_authoritative_inputs()
        for path, expected in AUTHORITATIVE_INPUT_SHA256.items():
            self.assertEqual(sha256_file(ROOT / path), expected)

    def test_manifest_closes_final_label_and_execution_firewalls(self) -> None:
        manifest = json.loads(
            (SEAL_ROOT / "stage3_finetuning_policy_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        verify_manifest(manifest)
        self.assertEqual(manifest["policy"], policy_dict())
        self.assertTrue(manifest["firewall"]["policy_seal_only"])
        self.assertFalse(manifest["firewall"]["final_target_labels_accessed"])
        self.assertFalse(manifest["firewall"]["training_triggered"])
        self.assertFalse(manifest["firewall"]["evaluation_triggered"])

    def test_seal_code_cannot_import_training_evaluation_or_data_modules(self) -> None:
        files = (
            ROOT / "research" / "governance" / "stage3_policy.py",
            ROOT / "research" / "scripts" / "verify_stage3_policy_seal.py",
        )
        forbidden = ("research.training", "research.evaluation", "research.data", "torch")
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            for imported in imports:
                self.assertFalse(imported.startswith(forbidden), (path, imported))

    def test_verification_is_read_only_and_does_not_trigger_execution(self) -> None:
        paths = tuple(SEAL_ROOT.iterdir())
        before = {path: path.read_bytes() for path in paths if path.is_file()}
        result = verify_seal_bundle()
        after = {path: path.read_bytes() for path in paths if path.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["stage3_training_started"])
        self.assertFalse(result["final_target_labels_accessed"])
        self.assertFalse(result["downstream_execution_plan_authorized"])

    def test_default_policy_is_immutable(self) -> None:
        with self.assertRaises(Exception):
            REGISTERED_POLICY.learning_rate = 1e-3  # type: ignore[misc]
        self.assertEqual(Stage3FineTuningPolicy(), REGISTERED_POLICY)


if __name__ == "__main__":
    unittest.main()
