from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.data.window_dataset import _array as canonical_loader_array
from research.development.stage2b_vanilla_execution import (
    _filled,
    build_vanilla_cache,
)
from research.remote.stage2b_a40 import ROOT


class Stage2BFilledRegressionTests(unittest.TestCase):
    def test_masked_int64_casts_before_nan_fill(self) -> None:
        source = np.ma.MaskedArray(
            np.asarray([4, 9], dtype=np.int64), mask=[False, True]
        )
        result = _filled(source, np.float32, np.nan)
        self.assertEqual(result.dtype, np.dtype(np.float32))
        self.assertEqual(float(result[0]), 4.0)
        self.assertTrue(np.isnan(result[1]))

    def test_multiple_masked_integer_entries(self) -> None:
        source = np.ma.MaskedArray(
            np.asarray([0, 7, 0, 11], dtype=np.int64),
            mask=[False, True, True, False],
        )
        result = _filled(source, np.float32, np.nan)
        np.testing.assert_array_equal(np.isnan(result), [False, True, True, False])
        np.testing.assert_array_equal(result[[0, 3]], np.asarray([0.0, 11.0], dtype=np.float32))

    def test_unmasked_integer_input(self) -> None:
        source = np.asarray([0, 2, 5], dtype=np.int64)
        result = _filled(source, np.float32, np.nan)
        self.assertEqual(result.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(result, np.asarray([0.0, 2.0, 5.0], dtype=np.float32))

    def test_float_input_with_nan_fill(self) -> None:
        source = np.asarray([1.5, np.nan, 0.0], dtype=np.float64)
        result = _filled(source, np.float32, np.nan)
        self.assertEqual(result.dtype, np.dtype(np.float32))
        self.assertEqual(float(result[0]), 1.5)
        self.assertTrue(np.isnan(result[1]))
        self.assertEqual(float(result[2]), 0.0)

    def test_mask_and_observed_zero_semantics_are_preserved(self) -> None:
        source = np.ma.MaskedArray(
            np.asarray([0, 0, 3], dtype=np.int64), mask=[False, True, False]
        )
        result = _filled(source, np.float32, np.nan)
        self.assertEqual(float(result[0]), 0.0)
        self.assertFalse(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))
        self.assertEqual(float(result[2]), 3.0)
        np.testing.assert_array_equal(np.isnan(result), np.ma.getmaskarray(source))

    def test_stage2b_matches_canonical_loader_conversion(self) -> None:
        source = np.ma.MaskedArray(
            np.asarray([0, 2, 8, 0], dtype=np.int64),
            mask=[False, True, False, True],
        )
        expected = canonical_loader_array(source, np.float32, np.nan)
        observed = _filled(source, np.float32, np.nan)
        np.testing.assert_array_equal(observed, expected)
        self.assertEqual(observed.dtype, expected.dtype)

    def test_other_cache_dtype_fill_pairs_are_safe(self) -> None:
        integer = np.ma.MaskedArray([3, 8], mask=[False, True], dtype=np.int64)
        boolean = np.ma.MaskedArray([True, True], mask=[False, True], dtype=bool)
        floating = np.ma.MaskedArray([2.5, 9.5], mask=[False, True], dtype=np.float64)
        np.testing.assert_array_equal(_filled(integer, np.int64, -1), [3, -1])
        np.testing.assert_array_equal(_filled(boolean, bool, False), [True, False])
        np.testing.assert_array_equal(_filled(floating, np.float32, 0.0), [2.5, 0.0])


class Stage2BActualCacheSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="stage2b_cache_smoke_")
        cls.cache_root = Path(cls.temporary.name)
        cls.metadata = build_vanilla_cache(cls.cache_root, city_ids=(129,))

    @classmethod
    def tearDownClass(cls) -> None:
        gc.collect()
        cls.temporary.cleanup()

    def test_actual_cache_build_completed_without_scientific_execution(self) -> None:
        metadata = self.metadata[129]
        self.assertFalse(metadata["contains_graph"])
        self.assertFalse(metadata["contains_final_target_data"])
        self.assertGreater(metadata["hour_count"], 0)
        self.assertGreater(metadata["station_count"], 0)

    def test_cached_tensors_match_canonical_stage1_cache_exactly(self) -> None:
        current = self.cache_root / "city_129"
        canonical = ROOT / "tmp/stage1_scale_loss_cache_v2_1/city_129"
        for name in (
            "timestamps_us", "station_ids", "x_hist", "x_week", "x_static",
            "x_calendar", "target", "coverage_mask",
        ):
            observed = np.load(current / f"{name}.npy", allow_pickle=False)
            expected = np.load(canonical / f"{name}.npy", allow_pickle=False)
            self.assertEqual(observed.dtype, expected.dtype, name)
            self.assertEqual(observed.shape, expected.shape, name)
            self.assertTrue(np.array_equal(observed, expected, equal_nan=True), name)

    def test_cache_build_did_not_create_prediction_or_metric_artifacts(self) -> None:
        names = {path.name.lower() for path in self.cache_root.rglob("*") if path.is_file()}
        self.assertFalse(any("prediction" in name or "mae" in name or "checkpoint" in name for name in names))


if __name__ == "__main__":
    unittest.main(verbosity=2)
