from __future__ import annotations

import ast
import json
import unittest

from research.governance.stage2b_fairness_seal import (
    DROPOUTS,
    HIDDEN_SIZES,
    ROOT,
    SEAL_ROOT,
    SEEDS,
    SELECTION_RULE,
    UPSTREAM_SHA256,
    build_run_plan,
    quantize_primary,
    sha256_file,
    verify_seal_bundle,
    verify_upstream,
)


class Stage2BFairnessSealTests(unittest.TestCase):
    def test_upstream_decisions_and_audits_are_hash_pinned(self) -> None:
        verify_upstream()
        for relative, expected in UPSTREAM_SHA256.items():
            self.assertEqual(sha256_file(ROOT / relative), expected)

    def test_exact_grid_and_48_unique_jobs(self) -> None:
        rows = build_run_plan()
        self.assertEqual(len(rows), 48)
        self.assertEqual(len({row["job_id"] for row in rows}), 48)
        self.assertEqual({row["hidden_size"] for row in rows}, set(HIDDEN_SIZES))
        self.assertEqual({row["dropout"] for row in rows}, set(DROPOUTS))
        self.assertEqual({row["seed"] for row in rows}, set(SEEDS))
        self.assertEqual({row["model_family"] for row in rows}, {"genuine_vanilla_gru"})

    def test_every_fold_excludes_pseudo_target_and_has_no_adaptation(self) -> None:
        for row in build_run_plan():
            sources = {int(value) for value in row["source_city_ids"].split(";")}
            self.assertEqual(len(sources), 7)
            self.assertNotIn(row["pseudo_target_city_id"], sources)
            self.assertFalse(row["target_adaptation_fit"])
            self.assertTrue(row["zero_shot_evaluation"])
            self.assertEqual(row["status"], "SEALED_NOT_AUTHORIZED_FOR_EXECUTION")

    def test_selection_rule_and_half_up_quantization_are_exact(self) -> None:
        self.assertEqual(quantize_primary(1.23445), quantize_primary(1.23446))
        self.assertEqual(str(quantize_primary(1.23445)), "1.2345")
        self.assertEqual(SELECTION_RULE["tie_1"], "lower_population_standard_deviation_across_four_pseudo_target_seed_means")
        self.assertEqual(SELECTION_RULE["tie_2"], "smaller_hidden_size")
        self.assertEqual(SELECTION_RULE["tie_3"], "lower_dropout")

    def test_manifest_preserves_fairness_and_firewall_boundaries(self) -> None:
        manifest = json.loads((SEAL_ROOT / "stage2b_fairness_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["frozen_stage2_decision_unchanged"], "GGRU_K04_H032_D00")
        self.assertEqual(manifest["fairness_criterion"]["not_claimed_matched"], ["parameter_count", "wall_clock_compute", "hardware_cost"])
        self.assertFalse(manifest["firewall"]["stage2b_execution_authorized"])
        self.assertFalse(manifest["firewall"]["stage2b_training_triggered"])
        self.assertFalse(manifest["firewall"]["final_target_labels_accessed"])

    def test_seal_and_verifier_have_no_training_dependencies(self) -> None:
        for path in (
            ROOT / "research" / "governance" / "stage2b_fairness_seal.py",
            ROOT / "research" / "scripts" / "verify_stage2b_fairness_seal.py",
        ):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            self.assertFalse(any(name.startswith(("torch", "research.training", "research.evaluation")) for name in imports))

    def test_read_only_verifier_passes_without_mutation(self) -> None:
        paths = tuple(path for path in SEAL_ROOT.iterdir() if path.is_file())
        before = {path: path.read_bytes() for path in paths}
        result = verify_seal_bundle()
        after = {path: path.read_bytes() for path in paths}
        self.assertEqual(before, after)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["stage2b_execution_authorized"])


if __name__ == "__main__":
    unittest.main()
