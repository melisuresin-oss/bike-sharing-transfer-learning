from __future__ import annotations

import ast
import json
import math
import unittest

from research.data.budget_sampler import EqualCitySampler
from research.governance.stage4_adversarial_method_seal import (
    BACKBONE,
    FUTURE_SOFTWARE_GATES,
    LAMBDA_MAX_VALUES,
    METHOD_SPECIFICATION,
    PSEUDO_TARGETS,
    ROOT,
    SCHEDULES,
    SEAL_ROOT,
    SEEDS,
    SOURCE_UPDATES,
    UPSTREAM_SHA256,
    build_workload,
    sha256_file,
    specified_lambda,
    verify_seal_bundle,
    verify_upstream,
)


class Stage4AdversarialMethodSealTests(unittest.TestCase):
    def test_upstream_stages_and_audits_are_hash_pinned(self) -> None:
        verify_upstream()
        for relative, expected in UPSTREAM_SHA256.items():
            self.assertEqual(sha256_file(ROOT / relative), expected)

    def test_backbone_and_grid_are_exact(self) -> None:
        self.assertEqual(BACKBONE["config_id"], "GGRU_K04_H032_D00")
        self.assertEqual(BACKBONE["scale_loss_formulation"], "LOG1P_TARGET_MAE")
        self.assertEqual(LAMBDA_MAX_VALUES, (0.01, 0.10, 0.50))
        self.assertEqual(SCHEDULES, ("constant", "linear"))

    def test_exact_72_source_and_adaptation_workload(self) -> None:
        rows = build_workload()
        self.assertEqual(len(rows), 72)
        self.assertEqual(len({row["source_fit_id"] for row in rows}), 72)
        self.assertEqual(len({row["adaptation_fit_id"] for row in rows}), 72)
        self.assertTrue(all(row["adaptation_updates"] == 300 for row in rows))
        self.assertTrue(all(row["adaptation_history"] == "7_days" for row in rows))
        self.assertFalse(any(row["zero_shot_selection_evaluation"] for row in rows))

    def test_pseudo_target_is_excluded_from_sources_and_classes(self) -> None:
        for row in build_workload():
            sources = {int(value) for value in row["source_city_ids"].split(";")}
            classes = {int(value.split(":")[1]) for value in row["source_domain_class_map"].split(";")}
            self.assertEqual(len(sources), 7)
            self.assertEqual(sources, classes)
            self.assertNotIn(row["pseudo_target_city_id"], sources)

    def test_grl_objective_pooling_classifier_and_optimizer_contract(self) -> None:
        method = METHOD_SPECIFICATION
        self.assertFalse(method["canonical_source_target_dann_claim"])
        self.assertEqual(method["domain_ce_forward_weight"], 1.0)
        self.assertFalse(method["separate_alpha_coefficient"])
        self.assertEqual(method["grl_encoder_backward_multiplier"], "-lambda_t")
        self.assertEqual(method["forecast_head_domain_gradient"], "none")
        self.assertEqual(method["pooling"]["mask"], "current_hour_M_target_valid_station_coverage_mask_only")
        self.assertEqual(method["pooling"]["empty_mask_behavior"], "raise_error_no_denominator_clamp")
        self.assertFalse(method["pooling"]["M_hist_used"])
        self.assertEqual(method["classifier"]["layers"][2], "torch.nn.Dropout(0.10)")
        self.assertEqual(method["optimizer"]["type"], "single_joint_AdamW")
        self.assertEqual(method["optimizer"]["clip_scope"], "all_joint_optimizer_parameters")

    def test_lambda_schedule_boundaries_are_exact(self) -> None:
        for value in LAMBDA_MAX_VALUES:
            self.assertEqual(specified_lambda("linear", value, 0), 0.0)
            self.assertLess(specified_lambda("linear", value, 2998), value)
            self.assertEqual(specified_lambda("linear", value, 2999), value)
            self.assertEqual(specified_lambda("linear", value, 3000), value)
            for update in (0, 1, 2998, 2999, 3000, SOURCE_UPDATES - 1):
                self.assertEqual(specified_lambda("constant", value, update), value)

    def test_equal_city_sampler_contract_is_balanced(self) -> None:
        for target in PSEUDO_TARGETS:
            sources = tuple(city for city in (129, 194, 438, 467, 476, 532, 619, 658) if city != target)
            for seed in SEEDS:
                sequence = EqualCitySampler(sources, seed).sequence(SOURCE_UPDATES)
                counts = [sequence.count(city) for city in sources]
                self.assertEqual(sum(counts), SOURCE_UPDATES)
                self.assertLessEqual(max(counts) - min(counts), 1)

    def test_adaptation_diagnostics_and_future_gates_are_frozen(self) -> None:
        method = METHOD_SPECIFICATION
        self.assertEqual(method["adaptation"]["updates"], 300)
        self.assertTrue(method["adaptation"]["fresh_reset_adamw"])
        self.assertFalse(method["adaptation"]["domain_objective"])
        self.assertFalse(method["adaptation"]["discriminator_present"])
        self.assertIn("small_nonscientific_synthetic_grl_adversarial_optimization_test", FUTURE_SOFTWARE_GATES)
        self.assertAlmostEqual(1 / 7, 0.14285714285714285)
        self.assertAlmostEqual(math.log(7), 1.9459101490553132)

    def test_manifest_closes_implementation_and_execution_firewalls(self) -> None:
        manifest = json.loads((SEAL_ROOT / "stage4_adversarial_method_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["random_encoder_diagnostic"], "OMIT")
        self.assertFalse(manifest["firewall"]["stage4_implementation_authorized"])
        self.assertFalse(manifest["firewall"]["stage4_scientific_execution_authorized"])
        self.assertFalse(manifest["firewall"]["scientific_training_triggered"])
        self.assertFalse(manifest["firewall"]["final_target_labels_accessed"])
        self.assertEqual(len(manifest["unresolved_final_execution_blockers"]), 2)

    def test_seal_code_has_no_stage4_or_training_implementation_dependency(self) -> None:
        for path in (
            ROOT / "research" / "governance" / "stage4_adversarial_method_seal.py",
            ROOT / "research" / "scripts" / "verify_stage4_adversarial_method_seal.py",
        ):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            self.assertFalse(any(name.startswith(("torch", "research.training", "research.models")) for name in imports))

    def test_read_only_verifier_passes_without_mutation(self) -> None:
        paths = tuple(path for path in SEAL_ROOT.iterdir() if path.is_file())
        before = {path: path.read_bytes() for path in paths}
        result = verify_seal_bundle()
        after = {path: path.read_bytes() for path in paths}
        self.assertEqual(before, after)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["stage4_implementation_authorized"])


if __name__ == "__main__":
    unittest.main()
