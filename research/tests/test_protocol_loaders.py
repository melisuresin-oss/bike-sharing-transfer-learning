from __future__ import annotations

import unittest

import numpy as np

from research.data.graph_dataset import GraphCityHourLoader
from research.data.manifests import ProtocolArtifactRegistry
from research.data.protocol_dataset import MODEL_FEATURE_COLUMNS
from research.data.window_dataset import FinalStationInferenceLoader, StationWindowLoader


class ProtocolLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = ProtocolArtifactRegistry()

    def test_station_tensor_contract(self) -> None:
        loader = StationWindowLoader(
            self.registry,
            532,
            start_utc="2023-02-15T22:00:00Z",
            end_utc="2023-02-16T22:00:00Z",
        )
        batch = loader.fetch(0, 8)
        self.assertEqual(batch.x_hist.shape, (8, 24, 2))
        self.assertEqual(batch.m_hist.shape, (8, 24))
        self.assertEqual(batch.x_week.shape, (8, 2))
        self.assertEqual(batch.x_static.shape, (8, 2))
        self.assertEqual(batch.x_calendar.shape, (8, 6))
        self.assertEqual(batch.y.shape, (8,))
        self.assertEqual(batch.m_target.shape, (8,))
        self.assertNotIn("city_id", MODEL_FEATURE_COLUMNS)
        self.assertNotIn("station_id", MODEL_FEATURE_COLUMNS)

    def test_graph_tensor_contract_and_node_order(self) -> None:
        loader = GraphCityHourLoader(
            self.registry,
            476,
            start_utc="2023-02-15T22:00:00Z",
            end_utc="2023-02-16T22:00:00Z",
        )
        first = loader.fetch(0, 3)
        second = loader.fetch(3, 3)
        n = self.registry.manifest["station_counts"]["476"]
        self.assertEqual(first.x_hist.shape, (3, 24, n, 2))
        self.assertEqual(first.m_hist.shape, (3, 24, n))
        self.assertEqual(first.x_week.shape, (3, n, 2))
        self.assertEqual(first.x_static.shape, (n, 2))
        self.assertEqual(first.x_calendar.shape, (3, 6))
        self.assertEqual(first.y.shape, (3, n))
        self.assertEqual(first.m_target.shape, (3, n))
        np.testing.assert_array_equal(first.station_ids, second.station_ids)

    def test_city_homogeneous_graph_batch(self) -> None:
        loader = GraphCityHourLoader(self.registry, 619)
        batch = loader.fetch(0, 2)
        self.assertEqual(batch.city_id, 619)
        self.assertEqual(len(np.unique(batch.station_ids)), loader.node_count)

    def test_final_inference_has_no_y(self) -> None:
        station = FinalStationInferenceLoader(self.registry, 195)
        self.assertIsNone(station.fetch(0, 4).y)
        graph = GraphCityHourLoader(self.registry, 195, final_inference=True)
        self.assertIsNone(graph.fetch(0, 2).y)


if __name__ == "__main__":
    unittest.main()
