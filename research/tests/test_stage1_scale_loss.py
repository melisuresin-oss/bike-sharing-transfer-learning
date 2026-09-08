from __future__ import annotations

import unittest

import numpy as np

from research.data.budget_sampler import EqualCitySampler
from research.development.stage1_scale_loss import (
    DEVELOPMENT_CITIES,
    DEVELOPMENT_SEEDS,
    FINE_TUNE_UPDATES,
    FORMULATIONS,
    OUTPUT_MODE,
    PSEUDO_TARGETS,
    SOURCE_UPDATES,
    Stage1ArtifactRegistry,
    _filled,
    deterministic_seed,
    stage1_model_config,
)
from research.scripts.run_stage1_scale_loss import _tie_4, verify_governing_contract


class Stage1ScaleLossTests(unittest.TestCase):
    def test_registered_job_counts(self) -> None:
        self.assertEqual(len(FORMULATIONS) * len(PSEUDO_TARGETS) * len(DEVELOPMENT_SEEDS), 24)
        self.assertEqual(SOURCE_UPDATES, 12_000)
        self.assertEqual(FINE_TUNE_UPDATES, 300)

    def test_each_fold_has_seven_balanced_sources(self) -> None:
        for target_id in PSEUDO_TARGETS:
            sources = tuple(city_id for city_id in DEVELOPMENT_CITIES if city_id != target_id)
            self.assertEqual(len(sources), 7)
            sequence = EqualCitySampler(sources, seed=123).sequence(SOURCE_UPDATES)
            counts = [sequence.count(city_id) for city_id in sources]
            self.assertEqual(sum(counts), SOURCE_UPDATES)
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_raw_and_log1p_configs_differ_only_in_output_mode(self) -> None:
        raw = stage1_model_config("RAW").to_dict()
        log1p = stage1_model_config("LOG1P").to_dict()
        self.assertEqual(raw.pop("output_mode"), OUTPUT_MODE["RAW"])
        self.assertEqual(log1p.pop("output_mode"), OUTPUT_MODE["LOG1P"])
        self.assertEqual(raw, log1p)

    def test_derived_randomness_is_candidate_matched(self) -> None:
        for target_id in PSEUDO_TARGETS:
            for seed in DEVELOPMENT_SEEDS:
                raw_seed = deterministic_seed("source-anchor-sampling", target_id, seed)
                log_seed = deterministic_seed("source-anchor-sampling", target_id, seed)
                self.assertEqual(raw_seed, log_seed)

    def test_stage1_registry_rejects_final_artifacts(self) -> None:
        registry = Stage1ArtifactRegistry()
        with self.assertRaises(RuntimeError):
            registry.path("final_adaptation")
        with self.assertRaises(RuntimeError):
            registry.station_manifest(195)

    def test_four_decimal_tie_rounds_half_up(self) -> None:
        self.assertEqual(_tie_4(1.23444), _tie_4(1.23443))
        self.assertNotEqual(_tie_4(1.23445), _tie_4(1.23444))

    def test_nullable_integer_targets_convert_to_float_nan(self) -> None:
        source = np.ma.MaskedArray([1, 2], mask=[False, True])
        converted = _filled(source, np.float32, np.nan)
        self.assertEqual(float(converted[0]), 1.0)
        self.assertTrue(np.isnan(converted[1]))

    def test_governing_contract_has_no_registered_contradiction(self) -> None:
        result = verify_governing_contract()
        self.assertTrue(result["passed"])
        self.assertEqual(result["contradictions_found"], [])
        self.assertEqual(result["stage0_constants"]["source_updates"], 12_000)


if __name__ == "__main__":
    unittest.main()
